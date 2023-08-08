import abc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast

import torch

if TYPE_CHECKING:
    from .._buddy import Buddy
    from ._checkpointing import _BuddyCheckpointing


@dataclass
class _BuddyOptimizerConfig:
    """Optimizer configuration.

    Along with the state dict of each optimizer, this is saved with each checkpoint.
    """

    global_steps: int
    """int: Total number of optimization steps taken. This is used for logging, LR schedulers, etc."""

    optimizer_type: str
    """str: Type of optimizer to used. Currently only `adam` or `adadelta`."""

    learning_rate_schedulers: Dict[str, Callable[[int], float]]
    """dict: Dictionary mapping optimizer names to LR scheduling functions."""


class _BuddyOptimizer(abc.ABC):
    """Buddy's optimization interface."""

    # Supported optimizer types
    # TODO: improve typing here; Type[torch.optim.Optimizer] would work but the
    # superclass constructor doesn't match the subclass constructors
    _OPTIMIZER_TYPES: Dict[str, Any] = {
        "adam": torch.optim.Adam,
        "adadelta": torch.optim.Adadelta,
    }

    def __init__(
        self, optimizer_type: str, optimizer_checkpoint_interval: float
    ) -> None:
        """Optimizer-specific setup."""
        # Assign our training configuration.
        self._optimizer_config = _BuddyOptimizerConfig(
            global_steps=0,
            optimizer_type=optimizer_type,
            learning_rate_schedulers={},
        )

        # Map from optimizer name to optimizers
        # These are constructed lazily!
        self._optimizer_dict: Dict[str, torch.optim.Optimizer] = {}

        # Autocheckpoint variables
        self._optimizer_checkpoint_interval: float = optimizer_checkpoint_interval
        self._optimizer_last_checkpoint_time: Optional[float] = None

        # Default learning rate
        self._optimizer_default_learning_rate: Optional[
            Union[float, Callable[[int], float]]
        ] = None

    def minimize(
        self,
        loss: torch.Tensor,
        optimizer_name: str = "primary",
        *,
        retain_graph: bool = False,
        checkpoint_interval: Optional[float] = None,
        clip_grad_max_norm: Optional[float] = None,
    ) -> None:
        """Compute gradients and use them to minimize a loss function."""
        model = cast("Buddy", self)._model
        assert model is not None, "No model attached!"

        # Get optimizer
        self._instantiate_optimizer(optimizer_name)
        optimizer: torch.optim.Optimizer = self._optimizer_dict[optimizer_name]

        # Update learning rate using scheduler if possible
        schedulers = self._optimizer_config.learning_rate_schedulers
        if optimizer_name in schedulers:
            self._set_learning_rate(
                schedulers[optimizer_name](self._optimizer_config.global_steps),
                optimizer_name,
            )

        # Take gradient step
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)  # type: ignore
        if clip_grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]["params"],
                max_norm=clip_grad_max_norm,
            )
        optimizer.step()

        # Update global step count
        self._optimizer_config.global_steps += 1

        # Autocheckpoint procedure
        if checkpoint_interval is None:
            checkpoint_interval = self._optimizer_checkpoint_interval

        # Disable autocheckpoint if interval is 0
        if checkpoint_interval == 0:
            return

        if self._optimizer_last_checkpoint_time is None:
            # First iteration
            self._optimizer_last_checkpoint_time = time.time()
        elif (
            time.time() - cast(float, self._optimizer_last_checkpoint_time)
            > self._optimizer_checkpoint_interval
        ):  # pragma: no cover
            # Checkpoint!
            cast("_BuddyCheckpointing", self).save_checkpoint()
            self._optimizer_last_checkpoint_time = time.time()

    def get_learning_rate(self, optimizer_name: str = "primary") -> float:
        """Gets an optimizer learning rate."""
        assert cast("Buddy", self)._model is not None, "No model attached!"
        assert optimizer_name in self._optimizer_dict

        # Return scheduled learning rate
        schedulers = self._optimizer_config.learning_rate_schedulers
        if optimizer_name in schedulers:
            return schedulers[optimizer_name](self.optimizer_steps)

        # Return raw learning rate
        # Currently, only one parameter group is supported
        optimizer = self._optimizer_dict[optimizer_name]
        assert len(optimizer.param_groups) == 1
        return optimizer.param_groups[0]["lr"]

    def set_learning_rate(
        self,
        value: Union[float, Callable[[int], float]],
        optimizer_name: str = "primary",
    ) -> None:
        """Sets an optimizer learning rate. Accepts either a floating point
        learning rate or a schedule function (int steps -> float LR).
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"
        schedulers = self._optimizer_config.learning_rate_schedulers
        if callable(value):
            assert isinstance(value(0), float)

            # Make sure optimizer is instantiated: if not, learning rate will be
            # overriden when it is
            self._instantiate_optimizer(optimizer_name)

            # Store scheduler
            schedulers[optimizer_name] = value
        else:
            # Set learning rate to a float
            assert isinstance(value, float)
            # Delete scheduler
            if optimizer_name in schedulers.keys():
                schedulers.pop(optimizer_name)

            # Set scalar learning rate
            self._set_learning_rate(value, optimizer_name)

    def set_default_learning_rate(
        self, value: Union[float, Callable[[int], float]]
    ) -> None:
        """Sets a default learning rate for new optimizers."""
        self._optimizer_default_learning_rate = value

    @property
    def optimizer_steps(self) -> int:
        """Read-only interface for # of steps taken by optimizer."""
        return self._optimizer_config.global_steps

    def _set_learning_rate(self, value: float, optimizer_name: str) -> None:
        """(Private) Sets an optimizer's learning rate."""

        self._instantiate_optimizer(optimizer_name)

        # Currently, only one parameter group is supported
        optimizer = self._optimizer_dict[optimizer_name]
        assert len(optimizer.param_groups) == 1
        optimizer.param_groups[0]["lr"] = value

    def _instantiate_optimizer(self, optimizer_name: str) -> None:
        """(Private) Instantiates an optimizer. Returns immediately if
        optimizer already exists.
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"
        if optimizer_name in self._optimizer_dict.keys():
            # Optimizer already exists: do nothing!
            return

        cast("Buddy", self)._print("Instantiating optimizer: ", optimizer_name)

        # Make sure we're creating a valid optimizer
        optimizer_type = self._optimizer_config.optimizer_type
        assert optimizer_type in self._OPTIMIZER_TYPES

        # Parameters
        optimizer_class = self._OPTIMIZER_TYPES[optimizer_type]

        # Construct optimizer
        self._optimizer_dict[optimizer_name] = optimizer_class(
            cast("Buddy", self).model.parameters()
        )

        # Set default learning rate
        if self._optimizer_default_learning_rate is not None:
            self.set_learning_rate(
                self._optimizer_default_learning_rate, optimizer_name=optimizer_name
            )
