# PYTHON_ARGCOMPLETE_OK

# To print options, install `fannypack` via pip and run `$ buddy --help`

"""CLI interface for experiment management via Buddy.

By default, searches for checkpoints in `./checkpoints/`, logs in `./logs/`, and
metadata in `./metadata/`. To override these values, set the BUDDY_CHECKPOINT_DIR,
BUDDY_LOG_DIR, and BUDDY_METADATA_DIR environment variables.
"""
import argparse
import os
from typing import Dict, Type

import argcomplete

from ._buddy_cli_include import BuddyPaths, Subcommand, subcommand_registry


def main() -> None:
    # Pull paths from environment
    paths = BuddyPaths(
        checkpoint_dir=os.environ.get("BUDDY_CHECKPOINT_DIR", default="checkpoints/"),
        log_dir=os.environ.get("BUDDY_LOG_DIR", default="logs/"),
        metadata_dir=os.environ.get("BUDDY_METADATA_DIR", default="metadata/"),
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(
        prog="buddy",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Separate parsers for subcommands
    subparsers = parser.add_subparsers(
        # required=True, <= Not supported in Python 3.6
        dest="subcommand",
        help="Get help by running `$ buddy {subcommand} --help`.",
    )

    # Add subcommands
    subcommand_map: Dict[str, Type[Subcommand]] = {}
    for S in subcommand_registry:
        subparser = subparsers.add_parser(
            S.subcommand, help=S.__doc__, description=S.__doc__
        )
        S.add_arguments(parser=subparser, paths=paths)
        subcommand_map[S.subcommand] = S

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Run subcommand
    if args.subcommand is None:
        print(parser.format_help())
        exit(2)
    else:
        subcommand_map[args.subcommand].main(args=args, paths=paths)


if __name__ == "__main__":
    main()
