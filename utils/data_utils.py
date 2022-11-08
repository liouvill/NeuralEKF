from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


class Sampler(ABC):
    """Abstract sampling class to sample random values."""

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Returns a single sample.

        Returns
        -------
        np.ndarray
            A single sample.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Returns a batch of samples.

        Returns
        -------
        np.ndarray
            A batch of samples.
        """
        raise NotImplementedError


@dataclass
class UniformSampler(Sampler):
    """Samples uniformly in the specified region."""

    lower_bound: np.ndarray
    upper_bound: np.ndarray

    def sample(self) -> np.ndarray:
        """See parent class."""
        return np.random.uniform(self.lower_bound, self.upper_bound)

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """See parent class."""
        return np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(batch_size, *self.lower_bound.shape),
        )


@dataclass
class GaussianSampler(Sampler):
    """Draws samples from a multivariate gaussian."""

    mean: Optional[np.ndarray]
    covariance: np.ndarray

    def sample(self) -> np.ndarray:
        """See parent class."""
        if self.mean is None:
            mean = np.zeros((self.covariance.shape[0]))
        else:
            mean = self.mean

        return np.random.multivariate_normal(mean, self.covariance)

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """See parent class."""
        if self.mean is None:
            mean = np.zeros((self.covariance.shape[0]))
        else:
            mean = self.mean
        return np.random.multivariate_normal(
            mean, self.covariance, size=batch_size
        ).astype(np.float32)


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a batch of data for training.

    Parameters
    ----------
    batch : Tuple[torch.Tensor, torch.Tensor]
        Batch of data comprised of the time and observtions.
    device : torch.device
        The device.

    Returns
    -------
    torch.Tensor, shape=(B, n)
        Inital observation.
    torch.Tensor, shape=(B, p)
        Time.
    torch.Tensor, shape=(T, B, n)
        Time series of observations.
    torch.Tensor, shape=(T, B, m)
        Time series of control inputs.
    """
    return (
        batch[1][:, 0, :].float().to(device),
        batch[0][0, :].float().to(device),
        batch[1].transpose(0, 1).float().to(device),
        batch[2].transpose(0, 1).float().to(device),
    )
