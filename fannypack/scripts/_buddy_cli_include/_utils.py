import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, cast


def _listdir(path: str) -> List[str]:
    """Helper for listing files in a directory"""
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []


@dataclass(frozen=True)
class FindOutput:
    """Output of `find_experiments(...)`."""

    experiment_names: Set[str]
    checkpoint_counts: Dict[str, int]
    log_experiments: Set[str]
    metadata_experiments: Set[str]
    timestamps: Dict[str, float]


def get_size(path: str) -> int:
    """Returns the size in bytes of the file or directory located at a path."""
    if os.path.isfile(path):
        return os.stat(path).st_size
    elif os.path.isdir(path):
        return sum(
            [
                os.stat(p).st_size
                for p in glob.glob(os.path.join(path, "**/*"), recursive=True)
            ]
        )
    else:
        return 0


def format_size(size: float, short: bool = False):
    """Converts a size in bytes to a human-readable string."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    unit_index = 0
    while size > 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if short:
        units = ["B", "K", "M", "G", "T", "P"]
        return f"{size:.0f}{units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


@dataclass
class BuddyPaths:
    """Dataclass for storing paths to experiment files."""

    checkpoint_dir: str
    log_dir: str
    metadata_dir: str

    _find_output_cache: Optional[FindOutput] = None

    def find_checkpoints(self, experiment_name: str):
        """Finds checkpoints associated with an experiment."""
        # Glob for checkpoint files
        checkpoint_files = glob.glob(
            os.path.join(self.checkpoint_dir, f"{glob.escape(experiment_name)}-*.ckpt")
        )

        # Filter further with rpartition (for handling experiment names with hyphens)
        return list(
            filter(
                lambda path: path.rpartition("-")[0].endswith(experiment_name),
                checkpoint_files,
            )
        )

    def get_log_dir(self, experiment_name: str):
        """Returns a path to an experiment's log directory."""
        return os.path.join(self.log_dir, experiment_name)

    def get_metadata_file(self, experiment_name: str):
        """Returns a path to an experiment's metadata file."""
        return os.path.join(self.metadata_dir, f"{experiment_name}.yaml")

    def find_experiments(self, verbose: bool = False) -> FindOutput:
        """Helper for listing experiments"""

        # Return cached results
        if self._find_output_cache is not None:
            return cast(FindOutput, self._find_output_cache)

        # Print helper
        def _print(*args, **kwargs):
            if not verbose:
                return
            print(*args, **kwargs)

        # Last modified: checkpoints and metadata only
        # > We could also do logs, but seems high effort?
        timestamps: Dict[str, float] = {}

        # Count checkpoints for each experiment
        checkpoint_counts: Dict[str, int] = {}
        for file in _listdir(self.checkpoint_dir):
            # Remove .ckpt suffix
            if file[-5:] != ".ckpt":
                _print(f"Skipping malformed checkpoint filename: {file}")
                continue
            trimmed = file[:-5]

            # Get experiment name
            name, hyphen, _label = trimmed.rpartition("-")
            if hyphen != "-":
                _print(f"Skipping malformed checkpoint filename: {file}")
                continue

            # Update tracker
            if name not in checkpoint_counts.keys():
                checkpoint_counts[name] = 0
            checkpoint_counts[name] += 1

            # Update timestamp
            mtime = os.path.getmtime(os.path.join(self.checkpoint_dir, file))
            if name not in timestamps.keys() or mtime > timestamps[name]:
                timestamps[name] = mtime

        # Get experiment names from metadata files
        metadata_experiments = set()
        for file in _listdir(self.metadata_dir):
            # Remove .yaml suffix
            if file[-5:] != ".yaml":
                _print(f"Skipping malformed metadata filename: {file}")
                continue
            name = file[:-5]
            metadata_experiments.add(name)

            # Update timestamp
            mtime = os.path.getmtime(os.path.join(self.metadata_dir, file))
            if name not in timestamps.keys() or mtime > timestamps[name]:
                timestamps[name] = mtime

        # Get experiment names from log directories
        log_experiments = set()
        for experiment_name in _listdir(self.log_dir):
            path = os.path.join(self.log_dir, experiment_name)
            if len(_listdir(path)) > 0:
                # Log files exist
                log_experiments.add(experiment_name)
            else:
                # Clean up empty directory
                os.rmdir(path)

        # Get all experiments
        experiment_names = (
            set(checkpoint_counts.keys()) | log_experiments | metadata_experiments
        )

        # Update global find_output cache
        self._find_output_cache = FindOutput(
            experiment_names=experiment_names,
            checkpoint_counts=checkpoint_counts,
            log_experiments=log_experiments,
            metadata_experiments=metadata_experiments,
            timestamps=timestamps,
        )
        return self._find_output_cache

    def clear_cache(self):
        """Clears the experiment list cache."""
        self._find_output_cache = None
