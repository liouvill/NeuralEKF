import argparse
import dataclasses
import os
import shutil

import simple_term_menu
import termcolor
from pygments import formatters, highlight, lexers

from ._subcommand import Subcommand
from ._utils import BuddyPaths

_TRASH_DIR = "./_trash/"


class DeleteSubcommand(Subcommand):
    """Delete one or more Buddy experiments.

    If no experiment names are passed in, displays a menu.
    """

    subcommand: str = "delete"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "experiment_name",
            type=str,
            help="Name of experiment, as printed by `$ buddy list`.",
            metavar="experiment_name",  # Set metavar => don't show choices in help menu
            choices=paths.find_experiments().experiment_names | {""},
            nargs="*",
            default="",  # Need default to prevent unhashable list error
        )
        parser.add_argument(
            "--forever",
            action="store_true",
            help=f"Delete experiment forever: if unset, move files into `{_TRASH_DIR}`.",
        )

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get experiment name
        experiment_names = args.experiment_name

        if len(experiment_names) > 0:
            # Delete specified experiments
            for name in experiment_names:
                try:
                    cls._delete_experiment(name, args.forever, paths)
                except RuntimeError as e:
                    print(f"Encountered error for {name}")
                    print(e)
        else: #  pragma: no cover
            print("Navigate: j/k \t Select: <CR>")
            print(
                termcolor.colored(
                    "There will be no confirmation message.", "red", attrs=["bold"]
                )
            )
            while True:
                delete_options = sorted(paths.find_experiments().experiment_names)
                if len(delete_options) == 0:
                    print("No experiments to delete!")
                    return

                # Show menu
                menu = simple_term_menu.TerminalMenu(
                    delete_options,
                    preview_command=lambda name: cls.get_pretty_metadata(name, paths),
                    preview_size=0.75,
                )
                experiment_name = menu.show()

                # Quit on Ctrl+C, Esc, etc
                if experiment_name is None:
                    return

                # Delete experiments
                cls._delete_experiment(
                    delete_options[experiment_name], args.forever, paths
                )
                paths.clear_cache()

    @classmethod
    def _delete_experiment(
        cls, experiment_name: str, forever: bool, paths: BuddyPaths
    ) -> None:
        # If we're just moving an experiment, check that it doesn't exist already
        if not forever:
            new_checkpoint_files = dataclasses.replace(
                paths, checkpoint_dir=os.path.join(_TRASH_DIR, paths.checkpoint_dir)
            ).find_checkpoints(experiment_name)
            if len(new_checkpoint_files) != 0:
                raise RuntimeError(
                    "Checkpoints for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )
            if os.path.exists(
                os.path.join(_TRASH_DIR, paths.log_dir, f"{experiment_name}")
            ):
                raise RuntimeError(
                    "Logs for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )
            if os.path.exists(
                os.path.join(_TRASH_DIR, paths.metadata_dir, f"{experiment_name}.yaml")
            ):
                raise RuntimeError(
                    "Metadata for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )

        # Delete checkpoint files
        checkpoint_paths = paths.find_checkpoints(experiment_name)
        print(f"Found {len(checkpoint_paths)} checkpoint files")
        for path in checkpoint_paths:
            cls._delete(path, forever)

        # Delete metadata
        metadata_path = os.path.join(paths.metadata_dir, f"{experiment_name}.yaml")
        if os.path.exists(metadata_path):
            cls._delete(metadata_path, forever)
        else:
            print("No metadata found")

        # Delete logs
        log_path = os.path.join(paths.log_dir, f"{experiment_name}")
        if os.path.exists(log_path):
            cls._delete(log_path, forever)
        else:
            print("No logs found")

    @staticmethod
    def _delete(path: str, forever: bool) -> None:
        assert os.path.exists(path)

        if forever:
            # Delete file/directory forever
            if os.path.isdir(path):
                print(f"Deleting {path} (recursive)")
                shutil.rmtree(path)
            elif os.path.isfile(path):
                print(f"Deleting {path}")
                os.remove(path)
            else:
                assert False, "Something went wrong"
        else:
            # Move files/directory to a new path
            new_path = os.path.join(_TRASH_DIR, path)

            # Create trash directory if it doesn't exist yet
            directory = os.path.dirname(new_path)
            if not os.path.isdir(directory):
                os.makedirs(directory)

            # Move file/directory to trash
            print(f"Moving {path} to {new_path}")
            os.rename(path, new_path)

    @staticmethod
    def get_pretty_metadata(experiment_name: str, paths: BuddyPaths) -> str:
        try:
            with open(paths.get_metadata_file(experiment_name), "r") as f:
                return highlight(
                    f.read().strip(),
                    lexers.YamlLexer(),
                    formatters.TerminalFormatter(),
                )
        except FileNotFoundError:
            return termcolor.colored("No metadata", "blue")
