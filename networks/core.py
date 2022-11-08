from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from fannypack.utils import Buddy
from torch import nn as nn

from datasets import DatasetConfig, VisData
from networks.models import ODENet
from utils.log_utils import log_basic, log_scalars
from utils.net_utils import (
    gaussian_log_prob,
    quadratic_matmul,
    safe_chol,
)


##########
# KALMAN #
##########


class KalmanEstimatorCell(ODENet, ABC):
    """Base abstract cell for Kalman filters."""

    def __init__(
        self,
        dynamics: nn.Module,
        observation_dynamics: Optional[nn.Module] = None,
        latent_dim: int = 2,
        observation_dim: int = 2,
        ctrl_dim: int = 0,
        initial_state: Optional[np.ndarray] = None,
        initial_variance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        const_var: bool = False,
        reparam: bool = False,
        regularizer: float = 1e-3,
    ) -> None:
        """Initialize an estimator cell.

        Parameters
        ----------
        dynamics : nn.Module
            System dynamics.
        observation_dynamics : nn.Module
            Observation dynamics.
        latent_dim : int
            Dimension of the state.
        observation_dim : int
            Dimension of the observation.
        ctrl_dim : int
            Dimension of the input.
        initial_state : Optional[np.ndarray]
            initial_state
        initial_variance : Optional[np.ndarray]
            initial_variance
        process_noise : Optional[np.ndarray]
            process_noise
        measurement_noise : Optional[np.ndarray]
            measurement_noise
        const_var : bool, default=False
            Variances are constant.
        reparam : bool, default=False
            Reparameterize to sample new states.
        regularizer: float, default=1e-3
            Covariance regularizer.
        """
        super(KalmanEstimatorCell, self).__init__()

        # set models
        self._dynamics = dynamics
        self._observation_dynamics = observation_dynamics

        # set common dimensions
        self._latent_dim = latent_dim
        self._observation_dim = observation_dim
        self._ctrl_dim = ctrl_dim

        self._reparam = reparam
        self._reg = regularizer

        # set initial parameters
        def _optional_tensor_decomposition(
            cov: Optional[np.ndarray], scale: float = 0.5, vtype: str = "process",
        ) -> torch.Tensor:
            if cov is None:
                if vtype == "process":
                    _V = torch.eye(latent_dim, device=self._device) * np.sqrt(scale)
                elif vtype == "observation":
                    _V = torch.eye(observation_dim, device=self._device) * np.sqrt(
                        scale
                    )
                else:
                    raise NotImplementedError
            else:
                _V = safe_chol(
                    torch.tensor(cov, dtype=torch.float, device=self._device)
                )
            if const_var:
                return _V
            else:
                return torch.nn.Parameter(_V)

        self._P0V = _optional_tensor_decomposition(
            initial_variance, scale=1, vtype="process"
        )
        self._QV = _optional_tensor_decomposition(
            process_noise, scale=0.01, vtype="process"
        )
        self._RV = _optional_tensor_decomposition(
            measurement_noise, scale=0.01, vtype="observation"
        )

        # initial latent state
        if initial_state is None:
            z0 = torch.zeros(latent_dim, dtype=torch.float)
        else:
            z0 = self._P0V.new_tensor(initial_state)
        self._z0 = torch.nn.Parameter(z0)

    # ---------- #
    # PROPERTIES #
    # ---------- #

    @property
    def Q(self) -> torch.Tensor:
        """Dynamics noise."""
        return (self._QV @ self._QV.T) * torch.eye(  # type: ignore
            self._latent_dim, device=self._device
        )

    @property
    def R(self) -> torch.Tensor:
        """Measurement noise."""
        return (self._RV @ self._RV.T) * torch.eye(  # type: ignore
            self._observation_dim, device=self._device
        )

    @property
    def P0(self) -> torch.Tensor:
        """Initial belief."""
        return (self._P0V @ self._P0V.T) * torch.eye(  # type: ignore
            self._latent_dim, device=self._device
        )

    @property
    def _device(self):
        return next(self.parameters()).device

    # --------- #
    # UTILITIES #
    # --------- #

    def device(self, z: torch.Tensor) -> torch.Tensor:
        """Sends a tensor to the device."""
        return z.to(self._device)

    def observation_dynamics(
        self, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Observe on latent state.

        If no obs model, observe on the first observation_dim number of
        latent states directly.

        Parameters
        ----------
        z : torch.Tensor, shape=(..., n)
            The state.
        """
        obs = self._observation_dynamics(z)
        #obs = torch.cat([obs1,obs2],-1)
        return obs


class KalmanEstimator(nn.Module, ABC):
    """Base abstract estimator for Kalman filters."""

    @property
    def config(self):
        """Estimator config."""
        return self._config
    
    def __init__(self, config: "KalmanEstimatorConfig") -> None:
        """Initializes the estimator."""
        super(KalmanEstimator, self).__init__()
        self._config = config

        self._is_smooth = config.is_smooth

        # tolerances for continuous-time estimation
        self._rtol = config.rtol
        self._atol = config.atol

        # loss-related hyperparameters
        self._burn_in = config.burn_in  # iterations of filter loss only for stability
        self._ramp_iters = config.ramp_iters  # ramping for logging
        self._dkl_anneal_iter = config.dkl_anneal_iter  # last dkl annealing iteration
        self._alpha = config.alpha  # reconstruction loss-weighting param
        self._beta = config.beta  # kl divergence loss-weighting param
        self._z_pred = config.z_pred  # flag for z prediction loss

    @property
    @abstractmethod
    def cell(self) -> KalmanEstimatorCell:
        """The cell associated with the estimator."""

    def get_initial_hidden_state(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """See parent class."""
        return self.cell.get_initial_hidden_state(batch_size, z0)

    def latent_to_observation(
        self,
        t: torch.Tensor,
        z_mu: torch.Tensor,
        z_cov: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        return self.cell.latent_to_observation(t, z_mu, z_cov)

    @abstractmethod
    def get_smooth(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return smoothed distributions."""

    def loss(
        self,
        batch_t: torch.Tensor,
        batch_y: torch.Tensor,
        batch_u: torch.Tensor,
        iteration: int,
        avg: bool = True,
        return_components: bool = False,
    ):
        """See parent class.

        New Parameters
        --------------
        return_components : bool, default=False
            Flag indicating whether to return loss components.
        """
        T, B = batch_y.shape[:2]

        # loss coefficients
        burn_in_coeff = min(1.0, iteration / self._burn_in)  # ramp up prediction weight
        anneal_coeff = min(1.0, iteration / self._dkl_anneal_iter)  # kl annealing

        z0_p = self.get_initial_hidden_state(B)
        z_mean, z_cov = self(
            batch_t, batch_y, batch_u, z0_p, return_hidden=True
        )
        y_mean, y_cov = self.latent_to_observation(batch_t, z_mean, z_cov)

        if not self._is_smooth:
            raise NotImplementedError
        z_mean_s, z_cov_s = self.get_smooth()

        # filter and kl loss
        # the order of the loss computations is important to preserve for LE-EKF!
        loss_dkl = self.kl_loss(z_mean_s, z_cov_s, batch_t, batch_u, avg=avg)
        loss_f = -gaussian_log_prob(y_mean, y_cov, batch_y)

        # smoothing/prediction loss
        y_mean_s, y_cov_s = self.latent_to_observation(batch_t, z_mean_s, z_cov_s)
        loss_s = -gaussian_log_prob(y_mean_s, y_cov_s, batch_y)
        loss_p = self.prediction_loss(
            z_mean_s, z_cov_s, batch_t, batch_y, batch_u, avg=avg
        )

        if avg:
            loss_f = torch.sum(loss_f) / (T * B)
            loss_s = torch.sum(loss_s) / (T * B)

        if return_components:
            return loss_f, loss_s, loss_p, loss_dkl
        else:
            return (
                self._alpha * loss_s
                + (1 - self._alpha) * burn_in_coeff * loss_p
                + self._beta * anneal_coeff * loss_dkl
            )

    def prediction_loss(
        self,
        z_mean: torch.Tensor,
        z_cov: torch.Tensor,
        batch_t: torch.Tensor,
        batch_y: torch.Tensor,
        batch_u: torch.Tensor,
        l2: bool = False,
        avg: bool = True,
    ) -> torch.Tensor:
        """Prediction loss computation.

        Parameters
        ----------
        z_mean : torch.Tensor, shape=(T, B, n)
            Latent means.
        z_cov : torch.Tensor, shape=(T, B, n)
            Latent covariances.
        batch_t : torch.Tensor, shape=(T)
            Times.
        batch_y : torch.Tensor, shape=(T, B, p)
            Observation trajectories.
        batch_u : torch.Tensor, shape=(T, B, m)
            Control inputs.
        l2 : bool
            Whether to use the l2 loss.
        avg : bool, default=True
            Flag indicating whether to average the loss.

        Returns
        -------
        torch.Tensor, shape=(1)
            Prediction loss.
        """
        T, B = batch_y.shape[:2]

        # take prediction loss over obs y or latent state z
        if not self._z_pred:
            y_mu_p, y_cov_p = self.predict(
                batch_t, z_mean[0], z_cov[0], batch_t, batch_u, return_hidden=False
            )

            if l2:
                loss_p = -((y_mu_p - batch_y) ** 2)
            else:
                loss_p = -gaussian_log_prob(y_mu_p, y_cov_p, batch_y)
        else:
            z_mu_p, z_cov_p = self.predict(
                batch_t, z_mean[0], z_cov[0], batch_t, batch_u, return_hidden=True,
            )
            z_mu_s, z_cov_s = self.get_smooth()  # use smoothed vals as targets

            if l2:
                loss_p = -((z_mu_p - z_mu_s) ** 2)
            else:
                loss_p = -gaussian_log_prob(z_mu_p, z_cov_p, z_mu_s)

        if avg:
            loss_p = torch.sum(loss_p) / (T * B)

        assert not torch.isnan(loss_p).any()
        return loss_p

    def kl_loss(
        self,
        z_mean: torch.Tensor,
        z_cov: torch.Tensor,
        batch_t: torch.Tensor,
        batch_u: torch.Tensor,
        avg: bool = True,
    ) -> torch.Tensor:
        """Compute KL divergence portion of the loss.

        Parameters
        ----------
        z_mean : torch.Tensor, shape=(T, B, n)
            The latent mean.
        z_cov : torch.Tensor, shape=(T, B, n)
            The latent covariance.
        batch_t : torch.Tensor, shape=(T)
            The times.
        batch_u : torch.Tensor, shape=(T, B, m)
            The control inputs.
        avg : bool, default=True
            Flag indicating whether to average the loss.

        Returns
        -------
        torch.Tensor
            KL divergence portion of the loss.
        """
        # jank implementation of kl divergence stuff
        # time invariant and assuming uniform dt.
        T, B, n = z_mean.shape

        hs_mean = z_mean[:-1].reshape(-1, n)  # first T-1 steps
        hs_cov = z_cov[:-1].reshape(-1, n, n)

        # repeated twice for API compliance, the second dim doesn't matter
        if batch_u.shape[-1] > 0:
            batch_u = batch_u[:-1].reshape(-1, batch_u[:-1].shape[-1]).repeat(2, 1, 1)
        else:
            batch_u = torch.zeros(2, (T - 1) * B, 0, device=self.cell._device)

        # simulate each point forward in time by one step
        times = z_mean.new_tensor([batch_t[0], batch_t[1]])  # assume constant dt
        z_mean_prior, z_cov_prior = self.predict(
            batch_t, hs_mean, hs_cov, times, batch_u, return_hidden=True
        )
        z_mean_prior = z_mean_prior[-1, ...].reshape(len(batch_t) - 1, B, n)
        z_cov_prior = z_cov_prior[-1, ...].reshape(len(batch_t) - 1, B, n, n)
        z_mean_prior = torch.cat([self.cell._z0.expand(1, B, n), z_mean_prior], dim=0)
        z_cov_prior = torch.cat([self.cell.P0.expand(1, B, n, n), z_cov_prior], dim=0)
        loss = kl_gaussian(z_mean, z_cov, z_mean_prior, z_cov_prior)
        if avg:
            loss = torch.sum(loss) / (T * B)
        return loss

    def log(self, buddy: Buddy, viz: VisData, filter_length: int = 1) -> None:
        """Logs data during training.

        Plots means of prediction and filter distributions versus viz data.

        TODO: eventually update these upstream functions with reasonable handling of
        conditional context. Main problem is that we expect the context appended to the
        data in the case of image data (KVAE) but not for stripped data. This means
        that cond is internally handled in some of the inherited log functions but must
        be externally passed here, which breaks the API a bit. We'll just handle this
        at a later time.

        Parameters
        ----------
        See parent class.
        """
        log_basic(
            self,
            buddy,
            viz,
            filter_length=filter_length,
            smooth=self._is_smooth,
            ramp_pred=True,
        )

        # log the loss components without any scaling effects applied during training
        # also apply ramped traj length to stabilize early EKF on long trajs
        iteration = buddy.optimizer_steps
        train_len = min((iteration // self._ramp_iters) + 2, len(viz.t))
        batch_t = viz.t[:train_len]
        batch_y = viz.y[:train_len]
        batch_u = viz.u[:train_len]
        loss_f, loss_s, loss_p, loss_dkl = self.loss(
            batch_t,
            batch_y,
            batch_u,
            buddy.optimizer_steps,
            avg=True,
            return_components=True,
        )

        log_scalars(
            buddy,
            {
                "Validation_F-Loss": loss_f.item(),
                "Validation_S-Loss": loss_s.item(),
                "Validation_P-Loss": loss_p.item(),
                "Validation_KL-Loss": loss_dkl.item(),
                "Traj_Length": train_len,
            },
            scope="Validation_KF",
        )


@dataclass(frozen=True)
class KalmanEstimatorConfig:
    """Kalman specific configuration parameters."""
    
    latent_dim: int
    ctrl_dim: int
    dataset: DatasetConfig
    dyn_hidden_units: int
    dyn_layers: int
    dyn_nonlinearity: nn.Module
    obs_hidden_units: int
    obs_layers: int
    obs_nonlinearity: nn.Module
    is_smooth: bool = True
    rtol: float = 1e-7
    atol: float = 1e-9
    ramp_iters: int = 1000
    burn_in: int = 100
    dkl_anneal_iter: int = 10000
    alpha: float = 0.5
    beta: float = 1.0
    z_pred: bool = False


# ------------ #
# LOSS HELPERS #
# ------------ #


def kl_gaussian(p_mu, p_cov, q_mu, q_cov):
    """TODO: replace calls with the one in utils."""
    mu_diff = q_mu - p_mu
    d = p_mu.shape[-1]
    # batched trace
    trace = torch.diagonal(torch.inverse(q_cov) @ p_cov, dim1=-2, dim2=-1).sum(-1)
    KL = 0.5 * (
        torch.logdet(q_cov)
        - torch.logdet(p_cov)
        - d
        + trace
        + quadratic_matmul(mu_diff, torch.inverse(q_cov))
    )
    return KL
