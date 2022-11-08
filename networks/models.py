from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from torch import nn as nn


class MLP(nn.Module):
    """A Multi-layer Perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_sizes: List[int],
        nonlinearity: nn.Module,
        dropout: Optional[float] = None,
        batchnorm: Optional[bool] = False,
    ) -> None:
        """Initialize the MLP. Activations are softpluses.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        output_dim : int
            Dimension of the output variable.
        hidden_layer_sizes : List[int]
            List of sizes of all hidden layers.
        nonlinearity : torch.nn.Module
            A the nonlinearity to use (must be a torch module).
        dropout : float
            Dropout probability if applied.
        batchnorm : bool
            Flag for applying batchnorm.
        """
        super(MLP, self).__init__()

        assert type(input_dim) == int
        assert type(output_dim) == int
        assert type(hidden_layer_sizes) == list
        assert all(type(n) is int for n in hidden_layer_sizes)

        # building MLP
        self._mlp = nn.Sequential()
        self._mlp.add_module("fc0", nn.Linear(input_dim, hidden_layer_sizes[0]))
        self._mlp.add_module("act0", nonlinearity)
        if batchnorm:
            self._mlp.add_module("bn0", nn.BatchNorm1d(hidden_layer_sizes[0]))
        if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
            self._mlp.add_module("do0", nn.Dropout(p=dropout))
        for i, (in_size, out_size) in enumerate(
            zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]), 1
        ):
            self._mlp.add_module(f"fc{i}", nn.Linear(in_size, out_size))
            self._mlp.add_module(f"act{i}", nonlinearity)
            if batchnorm:
                self._mlp.add_module("bn{i}", nn.BatchNorm1d(out_size))
            if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
                self._mlp.add_module("do{i}", nn.Dropout(p=dropout))
        self._mlp.add_module("fcout", nn.Linear(hidden_layer_sizes[-1], output_dim))

        # weight initialization
        for m in self._mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)
            
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape=(..., in_dim)
            Dimension of the input.

        Returns
        -------
        mlp_out : torch.Tensor, shape=(..., out_dim)
            Output tensor.
        """

        return self._mlp(torch.cat([x], dim=-1))


@dataclass
class MLPConfig:
    """Config dataclass for MLP."""

    input_dim: int
    output_dim: int
    hidden_layer_sizes: List[int]
    nonlinearity: nn.Module
    dropout: Optional[float] = None
    batchnorm: Optional[bool] = None

    def create(self) -> MLP:
        """Create a MLP from the config params."""
        return MLP(
            self.input_dim,
            self.output_dim,
            self.hidden_layer_sizes,
            self.nonlinearity,
            self.dropout,
            self.batchnorm,
        )


class ODENet(nn.Module, ABC):
    """Abstract class for a neural ODE."""

    def __init__(self) -> None:
        super(ODENet, self).__init__()

    @abstractmethod
    def forward(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Return the derivative of the state with respect to time.

        Parameters
        ----------
        time : torch.Tensor, shape=(1)
            The current time.
        state : torch.Tensor, shape(B, X)
            The current state.
            B - Batch size
            X - State dim

        Returns
        -------
        torch.tensor (B, X)
            Returns the derivative of the state with respect to time
        """
        raise NotImplementedError


class SimpleODENet(ODENet):
    """Simple ODE net."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: int,
        nonlinearity: Any,
        batchnorm: bool = False,
    ) -> None:
        """Initialize simple ODE net."""
        super(SimpleODENet, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_units),
            nonlinearity,
        ]

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nonlinearity)
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_units))

        layers.append(nn.Linear(hidden_units, output_dim))

        self._net = nn.Sequential(*layers)

        # weight initialization
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                # NOTE: fan_out or fan_in could be good. TODO: make a decision later.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)
            
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        t : torch.Tensor, shape=(1)
            Current time.
        x : torch.Tensor, shape=(B, n)
            State.
        u : Optional[torch.Tensor], shape=(B, m), default=None
            Control inputs.
        """
        # check for control input
        if u is None:
            u = x.new_tensor(np.zeros(x.shape[:-1] + (0,)))

        # compute the forward dynamics
        z = self._net(torch.cat([x, u], dim=-1))
        #z = torch.cat([z1,z2],-1)
        return z


@dataclass
class SimpleODENetConfig:
    """Config dataclass for simple ODE net."""

    input_dim: int
    output_dim: int
    hidden_layers: int
    hidden_units: int
    nonlinearity: nn.Module
    skip: bool = False
    batchnorm: bool = False

    def create(self) -> SimpleODENet:
        """Create a simple ODE net from the config params."""
        return SimpleODENet(
            self.input_dim,
            self.output_dim,
            self.hidden_layers,
            self.hidden_units,
            self.nonlinearity,
            self.batchnorm,
        )