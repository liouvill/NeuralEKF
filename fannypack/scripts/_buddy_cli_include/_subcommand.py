import abc
import argparse
from typing import List, Type

from ._utils import BuddyPaths

subcommand_registry: List[Type["Subcommand"]] = []


class Subcommand(abc.ABC):
    """Subcommand interface: defines arguments, runtime routine."""

    subcommand: str
    helptext: str

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers a subcommand."""
        super().__init_subclass__(**kwargs)
        subcommand_registry.append(cls)

    @classmethod
    @abc.abstractmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        raise NotImplementedError
