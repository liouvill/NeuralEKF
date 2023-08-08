import abc
import contextlib
import pathlib
from typing import TYPE_CHECKING, Generator, List, Optional, Union, cast

import numpy as np
import torch.utils.tensorboard

from .. import _deprecation

if TYPE_CHECKING:
    from .._buddy import Buddy
    from ._optimizer import _BuddyOptimizer


class _BuddyLogging(abc.ABC):
    """Buddy's TensorBoard logging interface."""

    def __init__(self, log_dir: str) -> None:
        """Logging-specific setup.

        Args:
            log_dir (str): Path to save Tensorboard logs to.
        """
        self._log_dir = log_dir

        # Backwards-compatibility for deprecated API
        self.log = _deprecation.new_name_wrapper(
            "Buddy.log()", "Buddy.log_scalar()", self.log_scalar
        )
        self.log_model_grad_hist = _deprecation.new_name_wrapper(
            "Buddy.log_model_grad_hist()",
            "Buddy.log_grad_histogram()",
            self.log_grad_histogram,
        )
        self.log_model_weight_hist = _deprecation.new_name_wrapper(
            "Buddy.log_model_weight_hist()",
            "Buddy.log_parameters_histogram()",
            self.log_parameters_histogram,
        )

        # State variables for TensorBoard
        # Note that the writer is lazily instantiated; see below
        self._log_writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
        self._log_scopes: List[str] = []

    @contextlib.contextmanager
    def log_scope(self, scope: str) -> Generator[None, None, None]:
        """Returns a context manager that scopes log names.

        Example usage:

        ```
            with buddy.log_scope("scope"):
                # Logs to scope/loss
                buddy.log_scalar("loss", loss_tensor)
        ```

        Args:
            scope (str): Name of scope.
        """
        self.log_scope_push(scope)
        yield
        self.log_scope_pop(scope)

    def log_scope_push(self, scope: str) -> None:
        """Push a scope to log tensors into.

        Example usage:
        ```
            buddy.log_scope_push("scope")

            # Logs to scope/loss
            buddy.log_scalar("loss", loss_tensor)

            buddy.log_scope_pop("scope") # name parameter is optional

        Args:
            scope (str): Name of scope.
        ```
        """
        self._log_scopes.append(scope)

    def log_scope_pop(self, scope: str = None) -> None:
        """Pop a scope we logged tensors into. See `log_scope_push()`.

        Args:
            scope (str, optional): Name of scope. Needs to be the top one in the stack.
        """
        popped = self._log_scopes.pop()
        if scope is not None:
            assert popped == scope, f"{popped} does not match {scope}!"

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        dataformats: str = "CHW",
    ) -> None:
        """Convenience function for logging an image tensor for visualization in
        TensorBoard.

        Equivalent to:
        ```
        buddy.log_writer.add_image(
            buddy.log_scope_prefix(name),
            image,
            buddy.optimizer_steps,
            dataformats
        )
        ```

        Args:
            name (str): Identifier for Tensorboard.
            image (torch.Tensor or np.ndarray): Image to log.
            dataformats (str, optional): Dimension ordering. Defaults to "CHW".
        """
        # Add scope prefixes
        name = self.log_scope_prefix(name)

        # Log scalar
        optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps
        self.log_writer.add_image(
            name, image, global_step=optimizer_steps, dataformats=dataformats
        )

    def log_scalar(
        self, name: str, value: Union[torch.Tensor, np.ndarray, float]
    ) -> None:
        """Convenience function for logging a scalar for visualization in TensorBoard.

        Equivalent to:
        ```
        buddy.log_writer.add_scalar(
            buddy.log_scope_prefix(name),
            value,
            buddy.optimizer_steps
        )
        ```

        Args:
            name (str): Identifier for Tensorboard.
            value (torch.Tensor, np.ndarray, or float): Value to log.
        """
        # Add scope prefixes
        name = self.log_scope_prefix(name)

        # Log scalar
        optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps
        self.log_writer.add_scalar(name, value, global_step=optimizer_steps)

    def log_parameters_histogram(
        self, scope: str = "weights", *, ignore_zero_grad: bool = True
    ) -> None:
        """Log model weights into a histogram.

        Naming: with `scope` set to "weights", a parameter name "model.Linear.bias" will be
        logged to the tag `buddy.log_scope_prefix("weights/model/Linear/bias")`.

        Args:
            scope (str, optional): Scope to log gradients into. Defaults to "weights".
            ignore_zero_grad (bool, optional): Ignore parameters without gradients:
                decreases log sizes when only parts of models are being trained.
                Defaults to True.
        """
        optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps

        with self.log_scope(scope):
            for param_name, p in cast("Buddy", self).model.named_parameters():
                if ignore_zero_grad and p.grad is None:
                    continue

                param_name = param_name.replace(".", "/")
                self.log_writer.add_histogram(
                    tag=self.log_scope_prefix(param_name),
                    values=p.data.detach().cpu().numpy(),
                    global_step=optimizer_steps,
                )

    def log_grad_histogram(self, scope: str = "grad") -> None:
        """Log model gradients into a histogram. Should be called after
        `buddy.minimize()`.

        Naming: with `scope` set to "grad", a parameter name "model.Linear.bias" will be
        logged to the tag `buddy.log_scope_prefix("grad/model/Linear/bias")`.

        Args:
            scope (str, optional): Scope to log gradients into. Defaults to "grad".
        """
        optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps

        found_gradients = False
        with self.log_scope(scope):
            for param_name, p in cast("Buddy", self).model.named_parameters():
                if p.grad is None:
                    continue
                param_name = param_name.replace(".", "/")
                self.log_writer.add_histogram(
                    tag=self.log_scope_prefix(param_name),
                    values=p.grad.detach().cpu().numpy(),
                    global_step=optimizer_steps,
                )
                found_gradients = True

        assert found_gradients, "No gradients found!"

    def log_scope_prefix(self, name: str = "") -> str:
        """Get or apply the current log scope prefix.

        Example usage:
        ```
        print(buddy.log_scope_prefix()) # ""

        with buddy.log_scope("scope0"):
            print(buddy.log_scope_prefix("loss")) # "scope0/loss"

            with buddy.log_scope("scope1"):
                print(buddy.log_scope_prefix()) # "scope0/scope1/"
        ```

        Args:
            name (str, optional): Name to prepend a prefix to. Defaults to an empty string.

        Returns:
            str: Scoped log name, or scope prefix if input is empty.
        """
        if len(self._log_scopes) == 0:
            return name
        return "{}/{}".format("/".join(self._log_scopes), name)

    @property
    def log_writer(self) -> torch.utils.tensorboard.SummaryWriter:
        """Accessor for standard Tensorboard SummaryWriter. Instantiated lazily."""
        if self._log_writer is None:
            self._log_writer = torch.utils.tensorboard.SummaryWriter(
                pathlib.Path(self._log_dir) / cast("Buddy", self)._experiment_name
            )
        return self._log_writer
