from typing import Any, Dict, Iterable, List, Optional, Union, overload

import h5py
import numpy as np


class TrajectoriesFile(Iterable):
    """An interface for reading/writing trajectories via h5py.

    Each TrajectoriesFile represents an iterable list of trajectories, where trajectores
    are stored as dictionaries that map `str` keys to `np.ndarray` contents.

    Example usage (read):
    ```
    with TrajectoriesFile('test.hdf5') as traj_file:

        for traj in traj_file:
            print(traj.keys()) # list of keys
            print(traj['some-key-name']) # numpy array
    ```

    Example usage (write):
    ```
    traj_file = TrajectoriesFile('test.hdf5', read_only=False)

    traj_file.add_meta({'label': 5})
    traj_file.add_timestep({'a': 1, 'b': 2})
    traj_file.add_timestep({'a': 3, 'b': 4})

    with traj_file:
        traj_file.complete_trajectory()

    print(len(traj_file)) # 1 trajectory!

    with traj_file:
        print(traj_file[0]['label']) # 5
        print(traj_file[0]['a']) # [1, 3]
        print(traj_file[0]['b']) # [2, 4]
    ```

    Note that some operations -- ones that require interfacing with the filesytem --
    need to be called within a `with` statement.

    Args:
        path (str): File path for this trajectory file.
        convert_doubles (bool): Convert doubles to floats to shrink files.
        read_only (bool, optional): Open file in read-only mode.
        compress (bool, optional): Reduce filesize w/ gzip.
        verbose (bool, optional): Enable debug prints.
    """

    def __init__(
        self,
        path: str,
        convert_doubles: bool = True,
        read_only: bool = True,
        compress: bool = True,
        verbose: bool = True,
    ):
        assert path[-5:] == ".hdf5", "Missing file extension!"

        # Meta
        self._path: str = path
        self._convert_doubles: bool = convert_doubles
        self._read_only: bool = read_only
        self._compress: bool = compress
        self._verbose: bool = verbose

        # Maps content key => content
        self._content_dict: Dict[str, Any] = {}

        # Number of timesteps in current trajectory
        self._current_trajectory_timesteps: int = 0

        # Count the number of trajectories that already exist
        self._trajectory_prefix: str = "trajectory"
        with self._h5py_file() as f:
            self._print("Loading trajectory from file:", f)
            if len(f.keys()) > 0:
                prefix_length = len(self._trajectory_prefix)
                ids = [int(k[prefix_length:]) for k in f.keys()]
                self._trajectory_count = max(ids) + 1
            else:
                self._trajectory_count = 0

            self._print("Existing trajectory count:", self._trajectory_count)

        assert type(self._trajectory_count) == int

        # File object
        self._file: Optional[h5py.File] = None

    def __enter__(self):
        """Automatic file opening, for use in `with` statements."""
        if self._file is None:
            self._print("Opening file...")
            self._file = self._h5py_file()
        return self

    def __exit__(self, *unused):
        """Automatic file closing, for use in `with` statements."""
        if self._file is not None:
            self._print("Closing file...")
            self._file.close()
            self._file = None

    @overload
    def __getitem__(self, index: slice) -> List[Dict[str, Union[np.ndarray, str]]]:
        ...

    @overload
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        ...

    def __getitem__(self, index):
        """Accessor for reading trajectories held by this file.
        Must be called with the TrajectoriesFile object in a `with` statement.

        Args:
            index (int or slice): Trajectory or trajectories to read.

        Returns:
            dict or list: If indexing with an int, returns a (str->value) dict
            containing data collected at each timestep of our trajectory. If indexing
            with a slice, returns a list of these dictionaries.
        """
        assert self._file is not None, "Not called in with statement!"

        # Slice into file
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        # Index checks
        assert isinstance(index, int)
        if index < 0 and index >= -len(self):
            # Negative indexing
            index = index % len(self)
        elif index >= len(self) or index < -len(self):
            # For use as a standard Python iterator
            raise IndexError

        traj_key = self._trajectory_prefix + str(index)
        assert traj_key in self._file.keys()

        # Copy values to numpy array
        output: Dict[str, Union[np.ndarray, str]] = {}
        for key, value in self._file[traj_key].items():
            # Conversion
            converted_value = np.array(value)

            # Numpy strings => native strings
            if converted_value.dtype.type is np.string_:
                # Decode
                output[key] = bytes(converted_value).decode("utf-8")
            else:
                output[key] = converted_value

        return output

    def __iter__(self):
        """Iterable __iter__() interface."""
        self._iter_index = 0
        return self

    def __next__(self) -> Dict[str, Union[np.ndarray, str]]:
        """Iterable __next__() interface."""
        try:
            output = self[self._iter_index]
            self._iter_index += 1
            return output
        except IndexError:
            pass
        raise StopIteration

    def __setitem__(self, index: int, item: Dict[str, Union[np.ndarray, str]]):
        """Assignment operation for modifying or mutating trajectories.
        Must be called with the TrajectoriesFile object in a `with` statement.

        Args:
            index (int): Trajectory #.
            item (dict): A (str->value) map, as would be returned by __getitem__().
        """
        assert self._file is not None, "Not called in with statement!"

        # Input type validation
        assert type(item) == dict

        # Index checks
        assert type(index) == int
        if index < 0 and index >= -len(self):
            # Negative indexing
            index = index % len(self)
        elif index >= len(self) or index < -len(self):
            # For use as a standard Python iterator
            raise IndexError

        traj_key = self._trajectory_prefix + str(index)
        group = self._file[traj_key]

        # Delete anything that's already in this trajectory
        for key in group:
            del group[key]

        # Populate the trajectory
        for key, value in item.items():
            # Convert content to a numpy array
            if type(value) == str:
                data = np.string_(value)
            else:
                data = np.asarray(value)

            # Compress floats
            if data.dtype == np.float64 and self._convert_doubles:
                data = data.astype(np.float32)

            # Disable compression for scalars
            if len(data.shape) == 0:
                group.create_dataset(key, data=data)
            else:
                group.create_dataset(
                    key,
                    data=data,
                    compression="lzf" if self._compress else None,
                )

    def __len__(self) -> int:
        """Returns the number of recorded trajectories."""
        return self._trajectory_count

    def resize(self, count: int):
        """Expand or contract our TrajectoriesFile."""
        assert self._file is not None, "Not called in with statement!"

        if self._trajectory_count <= count:
            # Expand
            for index in range(self._trajectory_count, count):
                traj_key = self._trajectory_prefix + str(index)
                self._file.create_group(traj_key)

        elif self._trajectory_count > count:
            # Contract
            for index in range(count, self._trajectory_count):
                traj_key = self._trajectory_prefix + str(index)
                del self._file[traj_key]

        self._trajectory_count = count

    def get_all(self, key: str) -> list:
        """Get contents associated with a key from all trajectories.

        Args:
            key (str): Content identifier.
        Returns:
            list: List of contents. First index is trajectory #.
        """
        assert self._file is not None, "Not called in with statement!"

        # Copy values to numpy array
        output = []
        for index in range(self._trajectory_count):
            traj_key = self._trajectory_prefix + str(index)
            assert traj_key in self._file.keys()
            value = self._file[traj_key][key]

            # Conversion
            value = np.array(value)
            assert type(value) == np.ndarray

            # Numpy strings => native strings
            if value.dtype.type is np.string_:
                # Decode
                value = bytes(value).decode("utf-8")

            # Add to output
            output.append(value)
        return output

    def add_timestep(self, content: Dict[str, np.ndarray]) -> None:
        """Add a timestep to the current trajectory.

        Args:
            content (dict): Map from timestep keys (str) to values (np.ndarray).
        """
        for key, value in content.items():
            if key not in self._content_dict:
                self._content_dict[key] = []

            assert type(self._content_dict[key]) == list
            self._content_dict[key].append(np.copy(value))

        # Increment length
        self._current_trajectory_timesteps += 1

    def add_meta(self, content: Dict[str, np.ndarray]) -> None:
        """Add some metadata to the current trajectory.

        Args:
            content (dict): Map from metadata keys (str) to values (np.ndarray).
        """
        for key, value in content.items():
            assert key not in self._content_dict.keys()
            if type(value) == str:
                self._content_dict[key] = value
            else:
                self._content_dict[key] = np.copy(value)

    def abandon_trajectory(self) -> None:
        """Abandon the current trajectory."""
        self._print("Abandoning trajectory")
        self._content_dict = {}
        self._current_trajectory_timesteps = 0

    def complete_trajectory(self) -> None:
        """Write the current trajectory to disk, and mark the start of a new
        trajectory.
        Must be called with the TrajectoriesFile object in a `with` statement.

        The next call to `add_timestep()` will be time 0 of the next trajectory.
        """
        assert self._file is not None, "Not called in with statement!"

        if self._content_dict == {}:
            self._print("Empty trajectory; skipping complete_trajectory()")
            return

        # Check length, print debug message
        self._print(
            "Completing trajectory!" f"(length={self._current_trajectory_timesteps})"
        )

        # Make space for an extra trajectory
        self.resize(self._trajectory_count + 1)

        # Copy contents in!
        self[-1] = self._content_dict

        # Reset state
        self._content_dict = {}
        self._current_trajectory_timesteps = 0
        self._print("Existing trajectory count:", self._trajectory_count)

    def clear(self) -> None:
        """Clear the contents of the TrajectoriesFile."""
        assert self._file is not None, "Not called in with statement!"

        for traj_key in self._file.keys():
            del self._file[traj_key]

    def _h5py_file(self, mode: str = None) -> h5py.File:
        """Private helper for creating h5py file objects."""
        if mode is None:
            mode = "r" if self._read_only else "a"

        return h5py.File(self._path, mode=mode, libver="latest")

    def _print(self, *args, **kwargs):
        """Private helper for logging."""
        # Only print in verbose mode
        if self._verbose:
            identifier = self._path
            if len(identifier) > 20:
                identifier = f"...{identifier[-17:]}"

            args = list(args)
            args[0] = f"[TrajectoriesFile-{identifier}] {args[0]}"
            print(*args, **kwargs)
