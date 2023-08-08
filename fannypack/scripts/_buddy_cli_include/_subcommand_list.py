import argparse
import datetime
import os

import beautifultable
import termcolor

from ._subcommand import Subcommand
from ._utils import BuddyPaths, format_size, get_size


class ListSubcommand(Subcommand):
    """Get & summarize existing Buddy experiments."""

    subcommand: str = "list"
    table_column_headers = [
        "Name",
        "Ckpts",
        "Logs",
        "Meta",
        "Timestamp",
    ]

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "--sort-by",
            type=str,
            choices=[h.lower() for h in cls.table_column_headers[:-1]],
            help="Sort experiment table by column. Defaults to timestamp.",
        )

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        results = paths.find_experiments(verbose=True)

        # Generate dynamic-width table
        try:
            terminal_columns = int(os.popen("stty size", "r").read().split()[1])
        except IndexError:
            # stty size fails when run from outside proper terminal (eg in tests)
            terminal_columns = 100
        table = beautifultable.BeautifulTable(maxwidth=min(100, terminal_columns))
        table.set_style(beautifultable.STYLE_BOX_ROUNDED)
        table.rows.separator = ""

        # Add bolded headers
        table.columns.header = [
            termcolor.colored(h, attrs=["bold"]) for h in cls.table_column_headers
        ]

        # Constant for "not applicable" fields
        NA = termcolor.colored("N/A", "red")

        # Add experiment rows, oldest to newest
        sorted_experiment_names = sorted(
            results.experiment_names,
            key=lambda n: results.timestamps.get(n, 0.0),
        )
        for name in sorted_experiment_names:
            # Get checkpoint count
            checkpoint_count = 0
            if name in results.checkpoint_counts:
                checkpoint_count = results.checkpoint_counts[name]

            # Get timestamp
            timestamp = ""
            if name in results.timestamps:
                timestamp = datetime.datetime.fromtimestamp(
                    results.timestamps[name]
                ).strftime(
                    "%b %d, %y @ %-H:%M" if terminal_columns > 100 else "%y-%m-%d"
                )

            # Add row for experiment
            table.rows.append(
                [
                    name,
                    checkpoint_count,
                    termcolor.colored(
                        format_size(get_size(paths.get_log_dir(name)), short=True),
                        "green",
                    )
                    if name in results.log_experiments
                    else NA,
                    termcolor.colored(
                        format_size(
                            get_size(paths.get_metadata_file(name)), short=True
                        ),
                        "green",
                    )
                    if name in results.metadata_experiments
                    else NA,
                    timestamp,
                ]
            )

        # Print table, sorted by name
        print(f"Found {len(results.experiment_names)} experiments!")
        if args.sort_by is not None:
            # Sort-by field: lowercase -> index
            table.sort(
                list(map(lambda s: s.lower(), cls.table_column_headers)).index(
                    args.sort_by
                )
            )
        print(table)
