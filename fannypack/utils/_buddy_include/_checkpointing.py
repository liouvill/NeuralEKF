import abc
import dataclasses
import pathlib
import signal
import warnings
from glob import escape
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import dill
import numpy as np
import torch

if TYPE_CHECKING:
    from .._buddy import Buddy
    from ._optimizer import _BuddyOptimizer


class _BuddyCheckpointing(abc.ABC):
    """Buddy's model checkpointing interface."""

    def __init__(self, checkpoint_dir: str, checkpoint_max_to_keep: int) -> None:
        """Checkpointing-specific setup."""
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_max_to_keep = checkpoint_max_to_keep

        # Find all unlabeled checkpoints for this experiment
        self._checkpoint_unlabeled_files, _ = self._find_checkpoints(
            checkpoint_dir=self._checkpoint_dir,
            experiment_name=cast("Buddy", self)._experiment_name,
            unlabeled_only=True,
        )

    def save_checkpoint(self, label: str = None) -> None:
        """Saves a checkpoint, which can optionally be labeled."""

        checkpoint_dir = pathlib.Path(self._checkpoint_dir)

        # Determine path to checkpoint file
        if label is not None:
            # Label explicitly specified!

            # Numerical labels are reserved for step counts (see below)
            label = cast(str, label)
            assert not label.isdigit()

            experiment_name = cast("Buddy", self)._experiment_name
            path = checkpoint_dir / f"{experiment_name}-{label}.ckpt"
        else:
            # Automatically decide label using global step count
            optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps
            path = (
                checkpoint_dir
                / f"{cast('Buddy', self)._experiment_name}-{optimizer_steps:016d}.ckpt"
            )

            if (
                self._checkpoint_unlabeled_files
                and path == self._checkpoint_unlabeled_files[-1]
            ):
                cast("Buddy", self)._print("Skipping redundant checkpoint save")
                return

        # Create directory if it doesn't exist yet
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        # Create state to save. This includes:
        # > Model state
        # > Optimizers
        # > Training steps
        # > Buddy configuration
        optimizer_states = {}

        for name, optimizer in cast("_BuddyOptimizer", self)._optimizer_dict.items():
            optimizer_states[name] = optimizer.state_dict()

        state = {
            "optimizer_config": dataclasses.asdict(
                cast("_BuddyOptimizer", self)._optimizer_config
            ),
            "optimizer_states": optimizer_states,
            "state_dict": cast("Buddy", self).model.state_dict(),
        }

        # Ignore SIGINT (eg ctrl+c) events while we save to disk...
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError as e:  # pragma: no cover
            # signal throws a ValueError if we're not in the main thread
            cast("Buddy", self)._print("Error while attaching SIGINT handler:", e)
            orig_handler = None

        # Checkpoint saving
        # > rename is POSIX-compliant and atomic
        tmp_path = checkpoint_dir / f"tmp-{np.random.randint(1e9)}.ckpt"
        torch.save(state, tmp_path, pickle_module=dill)
        tmp_path.rename(path)
        cast("Buddy", self)._print("Saved checkpoint to path:", path)

        # Restore SIGINT handler
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)

        # If unlabeled, add to list
        if label is None:
            self._checkpoint_unlabeled_files.append(path)

        # Prune checkpoint files
        while len(self._checkpoint_unlabeled_files) > self._checkpoint_max_to_keep:
            self._checkpoint_unlabeled_files.pop(0).unlink()

    def load_checkpoint_module(
        self,
        source: str,
        target: str = None,
        label: str = None,
        path: str = None,
        experiment_name: str = None,
    ) -> None:
        """Loads parameters from a specific child module within a checkpoint.
        By default, loads the checkpoint with the highest number of training
        iterations.

        Can also be specified via a label or file path.
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"

        if target is None:
            target = source

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)

        # Get possible target modules
        module_dict = dict(cast("Buddy", self).model.named_modules())
        assert target in module_dict.keys(), "Nonexistent target module!"

        # Build a state dict for this module only
        source_state_dict = {}
        key_prefix = ""
        if len(source) > 0:
            key_prefix = f"{source}."
        for key, value in checkpoint["state_dict"].items():
            if key.startswith(key_prefix):
                prefix_length = len(key_prefix)
                source_state_dict[key[prefix_length:]] = value

        # Load state dict
        missing, unexpected = module_dict[target].load_state_dict(source_state_dict)
        assert len(missing) == 0
        assert len(unexpected) == 0

        cast("Buddy", self)._print(f"Loaded module: {source} => {target}")

    def load_checkpoint_optimizer(
        self,
        source: str,
        target: str = None,
        label: str = None,
        path: str = None,
        experiment_name: str = None,
    ) -> None:
        """Loads state associated with a specific optimizer from a checkpoint.
        By default, loads the checkpoint with the highest number of training
        iterations.

        Can also be specified via a label or file path.
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"

        if target is None:
            target = source
        cast("_BuddyOptimizer", self)._instantiate_optimizer(target)

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)

        # Input validation
        assert (
            source in checkpoint["optimizer_states"].keys()
        ), "Nonexistent source optimizer!"
        assert (
            target in cast("_BuddyOptimizer", self)._optimizer_dict.keys()
        ), "Nonexistent target optimizer!"

        # Load optimizer state
        state_dict = checkpoint["optimizer_states"][source]
        cast("_BuddyOptimizer", self)._optimizer_dict[target].load_state_dict(
            state_dict
        )
        cast("Buddy", self)._print(f"Loaded optimizer: {source} => {target}")

    def load_checkpoint_optimizers(
        self, label=None, path=None, experiment_name=None
    ) -> None:
        """Loads all optimizer settings from a checkpoint. By default, loads
        the checkpoint with the highest number of training iterations.

        Can also be specified via a label or file path.
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)

        # Load optimizer state
        self._load_checkpoint_optimizers(checkpoint)

    def load_checkpoint(
        self, label: str = None, path: str = None, experiment_name: str = None
    ) -> None:
        """Loads a checkpoint. By default, loads the one with the highest
        number of training iterations.

        Can also be specified via a label or file path.
        """
        assert cast("Buddy", self)._model is not None, "No model attached!"

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)

        # Load optimizer state
        self._load_checkpoint_optimizers(checkpoint)

        # Load model parameters
        cast("Buddy", self).model.load_state_dict(checkpoint["state_dict"], strict=True)

        optimizer_steps = cast("_BuddyOptimizer", self).optimizer_steps
        cast("Buddy", self)._print("Loaded checkpoint at step:", optimizer_steps)
        return

    @property
    def checkpoint_labels(self) -> List[str]:
        """Accessor for listing available checkpoint labels.
        These should be saved as: `experiment_name-label.ckpt` in the
        `checkpoint_dir` directory.

        Returns:
            List[str]: Checkpoint labels, sorted alphabetically.
        """

        experiment_name = cast("Buddy", self)._experiment_name
        checkpoint_dir = pathlib.Path(self._checkpoint_dir)

        # Find all matching checkpoint files
        path_choices = filter(
            lambda path: path.stem.rpartition("-")[0] == experiment_name,
            checkpoint_dir.glob(f"{escape(experiment_name)}-*.ckpt"),
        )

        # Pull out labels
        output = []
        for choice in path_choices:
            label = choice.stem.partition("-")[2]
            output.append(label)

        # Sort output alphabetically and return
        output.sort()
        return output

    def _load_checkpoint_optimizers(self, checkpoint: Dict[str, Any]) -> None:
        # Load Buddy optimizer configuration
        optimizer_config = cast("_BuddyOptimizer", self)._optimizer_config
        for key, value in checkpoint["optimizer_config"].items():
            if hasattr(optimizer_config, key):
                assert type(value) == type(getattr(optimizer_config, key))
                setattr(optimizer_config, key, value)
            else:
                warnings.warn(f"Skipping missing configuration key: {key}={value}")

        # Instantiate optimizers & load state
        for name, state_dict in checkpoint["optimizer_states"].items():
            cast("_BuddyOptimizer", self)._instantiate_optimizer(name)
            cast("_BuddyOptimizer", self)._optimizer_dict[name].load_state_dict(
                state_dict
            )

    def _read_checkpoint_file(
        self,
        label: str = None,
        path: str = None,
        experiment_name: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Find a checkpoint to load.

        This is one of three options:
          1) The latest, based on step # (must have same experiment name)
          2) A file saved with a label (must have same experiment name)
          3) One specified by a path
        """

        # Determine path to checkpoint file
        if path is None and label is None:
            # Load latest unlabeled checkpoint

            # First, find all checkpoint paths
            paths, _ = self._find_checkpoints(
                checkpoint_dir=self._checkpoint_dir,
                experiment_name=cast("Buddy", self)._experiment_name
                if experiment_name is None
                else experiment_name,
            )
            if len(paths) == 0:
                raise FileNotFoundError("Missing checkpoint file")

            # The list of paths will be sorted by optimizer step count
            path = str(paths[-1])

        elif path is None and label is not None:
            # Load a labeled checkpoint
            if experiment_name is None:
                # Use our current experiment name by default
                experiment_name = cast("Buddy", self)._experiment_name
            path = str(
                pathlib.Path(self._checkpoint_dir) / f"{experiment_name}-{label}.ckpt"
            )
        elif path is not None:
            # Load a checkpoint by its location
            path = path
        else:
            assert (
                False
            ), "Too many arguments! Only one of (label, path) is supported at a time."

        # Load checkpoint dict
        checkpoint = torch.load(
            path, map_location=cast("Buddy", self)._device, pickle_module=dill
        )

        # Backwards-compatibility
        # This should eventually be removed :)
        renamed_fields = [
            ("config", "optimizer_config"),
            ("optimizers", "optimizer_states"),
        ]
        for old_name, new_name in renamed_fields:
            if old_name in checkpoint.keys():
                cast("Buddy", self)._print(
                    f"Legacy checkpoint field: {old_name} => {new_name}"
                )
                checkpoint[new_name] = checkpoint[old_name]
                checkpoint.pop(old_name)

        if "steps" in checkpoint.keys():
            cast("Buddy", self)._print("Legacy checkpoint field: steps")
            checkpoint["optimizer_config"]["global_steps"] = checkpoint["steps"]
            checkpoint.pop("steps")

        # Checkpoint file validation
        valid_keys = set(["optimizer_config", "optimizer_states", "state_dict"])
        for key in checkpoint.keys():
            assert key in valid_keys

        # Raise warning for optimizer type mismatches
        if (
            hasattr(self, "_optimizer_config")
            and checkpoint["optimizer_config"]["optimizer_type"]
            != cast("_BuddyOptimizer", self)._optimizer_config.optimizer_type
        ):
            warnings.warn("Checkpoint loading: overriding optimizer type.")

        if verbose:
            cast("Buddy", self)._print("Read checkpoint from path:", path)
        return checkpoint

    def _find_checkpoints(
        self, checkpoint_dir: str, experiment_name: str, unlabeled_only: bool = False
    ) -> Tuple[List[pathlib.Path], Dict[pathlib.Path, int]]:
        """(Private) Returns a list of all unlabeled checkpoints associated
        with this experiment, sorted from oldest to newest.
        """

        # Find all matching checkpoint files
        path_choices = filter(
            lambda path: path.stem.rpartition("-")[0] == experiment_name,
            pathlib.Path(checkpoint_dir).glob(f"{escape(experiment_name)}-*.ckpt"),
        )

        # Find checkpoint files + associated step counts
        paths = []
        step_counts = {}
        for choice in path_choices:
            steps = None
            string_steps = choice.stem.partition("-")[2]
            try:
                # Try unlabeled checkpoint
                step_counts[choice] = int(string_steps)
                paths.append(choice)
            except ValueError:
                # If label can't be cast to integer: this is a labeled checkpoint
                # Only incle unlabeled if unlabeled_only is False
                if not unlabeled_only:
                    paths.append(choice)

        # Add step counts for labeled checkpoints
        for path in paths:
            if path in step_counts:
                continue
            if unlabeled_only:
                # This condition should never be hit
                assert False

            steps = self._read_checkpoint_file(path=str(path), verbose=False)[
                "optimizer_config"
            ]["global_steps"]
            step_counts[path] = steps

        # Sort output by steps
        paths.sort(key=lambda path: step_counts[path])

        # Return paths & steps
        return paths, step_counts
