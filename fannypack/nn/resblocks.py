import abc
from typing import Callable, Dict

import torch
import torch.nn as nn


class Base(nn.Module, abc.ABC):
    """Base class for a generic residual block, with support for `"relu"`,
    `"leaky_relu"`, `"selu"`, and `"none"` activations.
    """

    _activation_types: Dict[str, Callable[..., nn.Module]] = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "none": nn.Identity,
    }
    """Dictionary of support activation types.
    """

    def __init__(self, activation: str = "relu", activations_inplace: bool = False):
        super().__init__()

        self.block1: nn.Module
        self.block2: nn.Module
        self.activation: nn.Module = self._activation_types[activation](
            inplace=activations_inplace
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """ResBlock forward pass."""
        residual = x
        x = self.block1(x)
        x = self.activation(x)
        x = self.block2(x)
        assert x.shape[0] == residual.shape[0]
        x += residual
        x = self.activation(x)
        return x


class Linear(Base):
    """Standard linear residual block."""

    def __init__(self, units: int, bottleneck_units: int = None, **resblock_base_args):
        super().__init__(**resblock_base_args)

        if bottleneck_units is None:
            bottleneck_units = units
        self.block1 = nn.Linear(units, bottleneck_units)
        self.block2 = nn.Linear(bottleneck_units, units)


class Conv2d(Base):
    """Standard convolutional residual block."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = None,
        kernel_size: int = 3,
        **resblock_base_args
    ):
        super().__init__(**resblock_base_args)

        if bottleneck_channels is None:
            bottleneck_channels = channels

        self.block1 = nn.Conv2d(
            channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.block2 = nn.Conv2d(
            bottleneck_channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )


__all__ = ["Base", "Linear", "Conv2d"]
