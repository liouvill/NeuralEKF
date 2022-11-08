from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn as nn


def batch_eye(
    eye_dim: int, batch_size: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create a batch of identity matrices.

    Parameters
    ----------
    eye_dim : int
        Desired eye dimension.
    batch_size : int
        Desired batch dimension.
    device : Optional[torch.device], default=None
        The device.

    Returns
    -------
    torch.Tensor, shape=(batch_size, eye_dim, eye_dim)
        Batched eye.
    """
    if device is None:
        eye = torch.eye(eye_dim).unsqueeze(0)  # shape=(1, eye_dim, eye_dim)
    else:
        eye = torch.eye(eye_dim, device=device).unsqueeze(0)
    return eye.expand([batch_size, eye_dim, eye_dim])


def dkl_diag_gaussian(
    mu_1: torch.Tensor,
    var_1: torch.Tensor,
    mu_2: Optional[torch.Tensor] = None,
    var_2: Optional[torch.Tensor] = None,
    log_flag: bool = False,
    avg: bool = True,
) -> torch.Tensor:
    """Computes the analytical KL divergence between two diagonal-covariance Gaussians.

    Consider two Gaussian distributions D_1 = N(mu_1, var_1) and D_2 = N(mu_2, var_2).
    This function will compute D_KL(D_1 || D_2). If the parameters of D_2 are none,
    then D_2 is assumed to be the standard normal distribution.

    Parameters
    ----------
    mu_1 : torch.Tensor, shape=(B, X)
        Mean of D_1.
    var_1 : torch.Tensor, shape=(B, X)
        Diagonal entries of covariance of D_1.
    mu_2 : torch.Tensor, shape=(B, X), default=None
        Mean of D_2. Optional.
    var_2 : torch.Tensor, shape=(B, X), default=None
        Diagonal entries of covariance of D_2. Optional.
    log_flag : bool, default=False
        Flag indicating whether the variances are passed as log-variances
    avg : bool, default=True
        Flag indicating whether to batch average the KL divergence.

    Returns
    -------
    dkl : torch.Tensor, shape=(1) or shape=(B, 1)
        The KL divergence.
    """
    B, X = mu_1.shape
    if log_flag:
        if mu_2 is None or var_2 is None:
            inner = -var_1 + torch.exp(var_1) + mu_1 ** 2
        else:
            mu_diff = mu_2 - mu_1
            inner = var_2 - var_1 + (torch.exp(var_1) + mu_diff ** 2) / torch.exp(var_2)
    else:
        if mu_2 is None or var_2 is None:
            inner = -torch.log(var_1) + var_1 + mu_1 ** 2

        else:
            inner = torch.log(var_2 / var_1) + (var_1 + (mu_2 - mu_1) ** 2) / var_2

    dkl = 0.5 * (torch.sum(inner, dim=-1) - X)
    if avg:
        dkl = torch.sum(dkl) / B
    return dkl


def gradient_norm(f: nn.Module) -> float:
    """Compute the gradient norm.

    Parameters
    ----------
    f : nn.Module
        A torch Module

    Returns
    -------
    float
        The gradient norm.
    """
    total_norm = 0.0
    for p in f.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def gaussian_log_prob(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    x: torch.Tensor,
    use_torch: bool = False,
) -> torch.Tensor:
    """Compute the log probability given the parameters of a gaussian.

    Parameters
    ----------
    mean : torch.Tensor, shape=(..., X)
        Mean of the gaussian.
    covariance : torch.Tensor, shape=(..., X, X) or shape=(..., X)
        Variance of the gaussian.
    x : torch.Tensor, shape=(..., X)
        The state being evaluated.

    Returns
    -------
    torch.Tensor, shape=(...)
        The log probability.
    """
    if use_torch:
        if mean.shape == covariance.shape:
            log_var = covariance
            exponential = (x - mean) ** 2 / (torch.exp(log_var) + 1e-6)
            quotient = np.log(2 * np.pi) + log_var
            return -0.5 * torch.sum(exponential + quotient, dim=-1)

        else:
            distribution = torch.distributions.MultivariateNormal(mean, covariance)
            torch_result = distribution.log_prob(x)
            return torch_result
    else:
        len_leading_dim = len(covariance.shape[:-2])
        dim = mean.shape[-1]
        shape = [1] * len_leading_dim + [dim, dim]
        tol = torch.eye(dim, device=mean.device).reshape(*shape) * 1e-6
        exponential = quadratic_matmul(x - mean, torch.inverse(covariance + tol))
        other_terms = torch.logdet(covariance + tol) + dim * np.log(2 * np.pi)
        log_p = -0.5 * (exponential + other_terms)
        return log_p


def jacobian(
    f: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
    out_dim: Optional[int] = None,
) -> torch.Tensor:
    """Compute the Jacobian of a function.

    Parameters
    ----------
    f: torch.nn.Module
        Function whose Jacobian will be computed.
    x : torch.Tensor, shape(B, X)
        The current state.
        B - Batch size
        X - State dim
    out_dim : Optional[int]
        The output dimension. Assumed to be equal to input dim by default.

    Returns
    -------
    torch.Tensor, shape=(B, X, X)
        The Jacobian of the state.
    """
    B, X = x.shape
    if out_dim is None:
        out_dim = X

    x = x.detach().clone()
    x = x.unsqueeze(1)  # (B, 1, X)
    x = x.repeat(1, out_dim, 1)  # (B, out_dim, X)
    x.requires_grad_(True)
    y = f(x.reshape(-1, X)).reshape(B, -1, out_dim)

    # mask is eye of shape (B, out_dim, out_dim)
    mask = batch_eye(out_dim, B).to(x.device)

    jac = torch.autograd.grad(y, x, mask, create_graph=True)
    return jac[0]


def lower_matrix_to_vector(lower: torch.Tensor) -> torch.Tensor:
    """Convert a lower triangular matrix to a vector.

    Parameters
    ----------
    lower : torch.Tensor
        lower

    Returns
    -------
    torch.Tensor

    """
    shape = lower.shape
    assert shape[-1] == shape[-2]
    lower_idx = torch.tril_indices(shape[-1], shape[-1])
    lower_flat = lower[..., lower_idx[0], lower_idx[1]]
    return lower_flat


def lower_vector_to_matrix(
    lower_flat: torch.Tensor, matrix_dim: Optional[int] = None
) -> torch.Tensor:
    """Convert a valid vector to a lower triangular matrix.

    Parameters
    ----------
    vector : torch.Tensor
        vector
    matix_dim : Optional[int]
        matix_dim

    Returns
    -------
    torch.Tensor

    """
    shape = lower_flat.shape

    if matrix_dim is None:
        # (N, N) matrix has L =  N * (N + 1) / 2 values in its
        # lower triangular region (including diagonals).

        # N = 0.5 * sqrt(8 * L + 1)**0.5 - 0.5
        matrix_dim = int(0.5 * (8 * shape[-1] + 1) ** 0.5 - 0.5)

    matrix_shape = shape[:-1] + (matrix_dim, matrix_dim)

    lower = torch.zeros(matrix_shape)
    lower_idx = torch.tril_indices(matrix_dim, matrix_dim)
    lower[..., lower_idx[0], lower_idx[1]] = lower_flat
    return lower


def phi(A: torch.Tensor) -> torch.Tensor:
    """Phi operator.

    See here for more details https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf

    Parameters
    ----------
    A: torch.Tensor, shape=(B, X, X)
        A batched matrix.

    Returns
    -------
    B: torch.Tensor, shape=(B, X, X)
        Phi(A)
    """
    B = torch.tril(A)
    B.diagonal(dim1=-2, dim2=-1)[:] /= 2  # type: ignore
    return B


def quadratic_matmul(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Matrix quadratic multiplication.

    Parameters
    ----------
    x : torch.Tensor, shape=(..., X)
        A batch of vectors.
    A : torch.Tensor, shape=(..., X, X)
        A batch of square matrices.

    Returns
    -------
    torch.Tensor, shape=(...,)
        Batched scalar result of quadratic multiplication.
    """
    assert x.shape[-1] == A.shape[-1] == A.shape[-2]

    x_T = x.unsqueeze(-2)  # shape=(..., 1, X)
    x_ = x.unsqueeze(-1)  # shape=(..., X, 1)
    quadratic = x_T @ A @ x_  # shape=(..., 1, 1)
    return quadratic.squeeze(-1).squeeze(-1)


def reg_psd(X: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    """Regularizes a batch of PSD matrix trajectories."""
    x_dim = X.shape[-1]
    return X + reg * torch.eye(x_dim, device=X.device)


def reparameterize_gauss(
    mu: torch.Tensor,
    cov_or_var: torch.Tensor,
    chol_flag: bool = False,
    log_flag: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Execute the reparameterization trick to sample from a batch of Gaussians.

    Parameters
    ----------
    mu : torch.Tensor, shape=(..., X)
        Means.
    cov_or_var : torch.Tensor, shape=(..., X, X) or (..., X)
        Covariance or variances.
    chol_flag : bool, default=False
        Flag indicating whether full covariances are passed as Cholesky factor.
    log_flag : bool, default=False
        Flag indicating whether diagonal variances are passed as log_var.
    device : Optional[torch.device], default=None
        Optional device specification.

    Returns
    -------
    x : torch.Tensor, shape=(..., X)
        Collection of samples.
    """
    if device is not None:
        eps = torch.randn_like(mu, device=device)
    else:
        eps = torch.randn_like(mu)
    if mu.shape[-1] == cov_or_var.shape[-1] == cov_or_var.shape[-2] and len(
        mu.shape
    ) + 1 == len(cov_or_var.shape):
        if chol_flag:
            x = mu + (cov_or_var @ eps.unsqueeze(-1)).squeeze(-1)
        else:
            # cov is a covariance matrix
            L = safe_chol(cov_or_var)
            x = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)
    elif mu.shape[-1] == cov_or_var.shape[-1]:
        # cov is flattened diagonal
        if log_flag:
            x = mu + torch.exp(0.5 * cov_or_var) * eps
        else:
            x = mu + torch.sqrt(cov_or_var) * eps
    else:
        raise NotImplementedError
    return x


def safe_chol(
    x: torch.Tensor, min_eigval: float = 1e-3, max_eigval_ratio: float = 1e7
) -> torch.Tensor:
    """Safe Cholesky decomposition function.

    This will catch any non-PD matrices in a batch, execute the eigendecomposition,
    adjust the offending eigenvalues, reconstruct the batch, then return the Cholesky
    factor appropriately while printing a warning.

    Parameters
    ----------
    x : torch.Tensor
        A collection of matrices on which the Cholesky decomposition will be executed.
    min_eigval : float, default=1e-3
        The minimum eigenvalue with which negative eigenvalues will be replaced.
    max_eigval_ratio : float, default=1e7
        The maximum ratio of eigenvalues in a failed Cholesky decomposition. Sometimes,
        the matrices are just poorly conditioned, so even if they are PD, the torch
        cholesky routine fails.

    Returns
    -------
    L : torch.Tensor
        The Cholesky factors of (potentially adjusted) x.
    """
    assert min_eigval > 0  # positive minimum eigenvalue
    if not torch.allclose(x, x.transpose(-1, -2), atol=1e-4):
        print(
            "safe_chol: Input matrices are not symmetric! Symmetrizing now."
            + "The largest offending difference is "
            + f"{torch.max(torch.abs(x - x.transpose(-1, -2))):.4f}."
        )
        x = 0.5 * (x + x.transpose(-1, -2))

    try:
        return torch.cholesky(x)
    except RuntimeError:
        # eigendecomposition
        _E, V = torch.symeig(x, eigenvectors=True)

        # produce warning about the largest offending eigenvalue
        offenders = _E[_E <= min_eigval]
        if len(offenders > 0):
            largest_bad_eigval = min(offenders)
        else:
            print("safe_chol: Conditioning error!")
            largest_bad_eigval = torch.tensor(float("nan"))

        print(
            "Cholesky factorization initially failed!"
            + " Covariance matrices minimally modified to be PD."
            + f" The largest offending eigenvalue is {largest_bad_eigval.item():.4f}"
        )

        # eigval replacement
        _E[_E <= min_eigval] = min_eigval

        # conditioning check
        E_shape = _E.shape
        _E = _E.reshape(-1, E_shape[-1])  # make _E 2D for clean indexing
        pre_inds = np.arange(_E.shape[0])
        min_vals = _E[pre_inds, torch.argmin(_E, dim=-1)]  # biggest val per batch
        ratios = _E / min_vals.unsqueeze(1)  # finding the ratios per batch
        ratios[ratios > max_eigval_ratio] = max_eigval_ratio  # adjusting
        _E = min_vals.unsqueeze(1) * ratios  # reconstruct better conditioned decomp
        _E = _E.reshape(*E_shape)  # change back to orig shape

        # batch reconstruction
        E = torch.diag_embed(_E)
        x_recon = V @ E @ V.transpose(-1, -2)
        return torch.cholesky(x_recon)
