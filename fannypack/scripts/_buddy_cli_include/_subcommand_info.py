import argparse
import os

import beautifultable
import termcolor
from pygments import formatters, highlight, lexers

import fannypack

from ._subcommand import Subcommand
from ._utils import BuddyPaths, format_size, get_size


class InfoSubcommand(Subcommand):
    """Print info about a Buddy experiment: checkpoints, metadata, etc."""

    subcommand: str = "info"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "experiment_name",
            type=str,
            help="Name of experiment, as printed by `$ buddy list`.",
            metavar="experiment_name",  # Set metavar => don't show choices in help menu
            choices=paths.find_experiments().experiment_names,
        )

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get experiment name
        experiment_name = args.experiment_name
        print(experiment_name)

        # Generate dynamic-width table
        try:
            terminal_columns = int(os.popen("stty size", "r").read().split()[1])
        except IndexError:
            # stty size fails when run from outside proper terminal (eg in tests)
            terminal_columns = 100
        table = beautifultable.BeautifulTable(
            maxwidth=min(100, terminal_columns),
            default_alignment=beautifultable.ALIGN_LEFT,
        )
        table.set_style(beautifultable.STYLE_BOX_ROUNDED)

        def add_table_row(label, value):
            table.rows.append([termcolor.colored(label, attrs=["bold"]), value])

        # Constant for "not applicable" fields
        NA = termcolor.colored("N/A", "red")

        # Find checkpoint files
        checkpoint_paths = paths.find_checkpoints(experiment_name)

        # Display size, labels of checkpoints
        if len(checkpoint_paths) > 0:
            checkpoint_total_size = 0
            checkpoint_labels = []
            buddy = fannypack.utils.Buddy(experiment_name, verbose=False)
            checkpoint_paths, steps = buddy._find_checkpoints(
                paths.checkpoint_dir, args.experiment_name
            )
            for checkpoint_path in checkpoint_paths:
                prefix = os.path.join(paths.checkpoint_dir, f"{experiment_name}-")
                suffix = ".ckpt"
                assert str(checkpoint_path).startswith(prefix)
                assert str(checkpoint_path).endswith(suffix)
                label = str(checkpoint_path)[len(prefix) : -len(suffix)]

                checkpoint_labels.append(f"{label} (steps: {steps[checkpoint_path]})")
                checkpoint_total_size += get_size(str(checkpoint_path))

            add_table_row(
                "Total checkpoint size",
                termcolor.colored(format_size(checkpoint_total_size), "green"),
            )
            add_table_row(
                "Average checkpoint size",
                termcolor.colored(
                    format_size(checkpoint_total_size / len(checkpoint_paths)), "green"
                ),
            )
            add_table_row("Checkpoint labels", "\n".join(checkpoint_labels))
        else:
            add_table_row("Total checkpoint size", NA)
            add_table_row("Average checkpoint size", NA)
            add_table_row("Checkpoint labels", "")

        # Display log file size
        log_path = paths.get_log_dir(args.experiment_name)
        if os.path.exists(log_path):
            #  _delete(log_path, args.forever)
            add_table_row(
                "Log size", termcolor.colored(format_size(get_size(log_path)), "green")
            )
        else:
            add_table_row("Log size", NA)

        # Display metadata + metadata size
        metadata_path = paths.get_metadata_file(args.experiment_name)
        if os.path.exists(metadata_path):
            add_table_row(
                "Metadata size",
                termcolor.colored(format_size(get_size(metadata_path)), "green"),
            )
            with open(metadata_path, "r") as f:
                add_table_row(
                    "Metadata",
                    highlight(
                        f.read().strip(),
                        lexers.YamlLexer(),
                        formatters.TerminalFormatter(),
                    ),
                )
        else:
            add_table_row("Metadata size", NA)
            add_table_row("Metadata", NA)

        # Print table
        print(table)
