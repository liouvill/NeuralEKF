import argparse
import glob
import os

from ._subcommand import Subcommand
from ._utils import BuddyPaths


class RenameSubcommand(Subcommand):
    """Rename a Buddy experiment."""

    subcommand: str = "rename"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "source",
            type=str,
            help="Current name of experiment, as printed by `$ buddy list`.",
            metavar="source",  # Set metavar => don't show choices in help menu
            choices=paths.find_experiments().experiment_names,
        )
        parser.add_argument("dest", type=str, help="New name of experiment.")

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get old, new experiment names
        old_experiment_name = args.source
        new_experiment_name = args.dest

        # Validate that new experiment name doesn't exist
        new_checkpoint_files = glob.glob(
            os.path.join(
                paths.checkpoint_dir, f"{glob.escape(new_experiment_name)}-*.ckpt"
            )
        )
        if len(new_checkpoint_files) != 0:
            raise RuntimeError(
                f"Checkpoints already exist for destination name: {new_experiment_name}"
            )
        if os.path.exists(paths.get_log_dir(new_experiment_name)):
            raise RuntimeError(
                f"Logs already exist for destination name: {new_experiment_name}"
            )
        if os.path.exists(paths.get_metadata_file(new_experiment_name)):
            raise RuntimeError(
                f"Metadata already exist for destination name: {new_experiment_name}"
            )

        # Move checkpoint files
        checkpoint_paths = paths.find_checkpoints(old_experiment_name)
        print(f"Found {len(checkpoint_paths)} checkpoint files")
        for path in checkpoint_paths:
            # Get new checkpoint path
            prefix = os.path.join(paths.checkpoint_dir, f"{old_experiment_name}-")
            suffix = ".ckpt"
            assert path.startswith(prefix)
            assert path.endswith(suffix)
            label = path[len(prefix) : -len(suffix)]
            new_path = os.path.join(
                paths.checkpoint_dir, f"{new_experiment_name}-{label}.ckpt"
            )

            # Move checkpoint
            print(f"> Moving {path} to {new_path}")
            os.rename(path, new_path)

        # Move metadata
        metadata_path = paths.get_metadata_file(old_experiment_name)
        if os.path.exists(metadata_path):
            new_path = paths.get_metadata_file(new_experiment_name)
            print(f"Moving {metadata_path} to {new_path}")
            os.rename(metadata_path, new_path)
        else:
            print("No metadata found")

        # Move logs
        log_path = paths.get_log_dir(old_experiment_name)
        if os.path.exists(log_path):
            new_path = paths.get_log_dir(new_experiment_name)
            print(f"Moving {log_path} to {new_path}")
            os.rename(log_path, new_path)
        else:
            print("No logs found")
