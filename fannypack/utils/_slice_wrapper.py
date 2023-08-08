from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import torch

import fannypack

# Valid raw types that we can wrap
_raw_types = set([list, tuple, np.ndarray, torch.Tensor])

# Generic types
MapOutputType = TypeVar("MapOutputType")
WrappedType = TypeVar(
    "WrappedType",
    bound=Union[
        List,
        Tuple,
        torch.Tensor,
        np.ndarray,
        Dict[Any, List],
        Dict[Any, Tuple],
        Dict[Any, torch.Tensor],
        Dict[Any, np.ndarray],
    ],
)


class SliceWrapper(Iterable, Generic[WrappedType]):
    """A wrapper class for creating a unified interface for slicing and manipulating:
    - Lists
    - Tuples
    - Torch tensors
    - Numpy arrays
    - Dictionaries containing a same-length group of any of the above

    This makes it easy to read, slice, and index into blocks of data organized into
    dictionaries.

    Nominally:
    ```
    dataset = SliceWrapper({
        "features": features,
        "labels": labels,
    })

    train_count = 100
    train_dataset = dataset[:train_count]
    val_dataset = dataset[train_count:]
    ```
    would be equivalent to:
    ```
    train_count = 100
    train_dataset = {
        "features": features[:train_count],
        "labels": b[:train_count],
    }
    val_dataset = {
        "features": features[train_count:],
        "labels": b[train_count:],
    }
    ```

    For convenience, a transparent interface is provided for iterables that are directly
    wrapped. Thus:
    ```
    SliceWrapper([1, 2, 3])[::-1]
    ```
    would return:
    ```
    [1, 2, 3][::-1]
    ```
    """

    def __init__(self, data: WrappedType):
        self.data: WrappedType = data
        """list, tuple, torch.Tensor, np.ndarray, or dict: Wrapped data."""

        # Sanity checks
        if type(self.data) == dict:
            # Cast for type-checking
            data_dict = cast(dict, self.data)

            # Every value in the dict should have the same length & type
            content_length = None
            content_type = None
            for value in data_dict.values():
                assert content_length is None or len(value) == content_length
                assert content_type is None or type(value) == content_type
                content_length = len(value)
                content_type = type(value)
                assert content_type in _raw_types
        else:
            # Non-dictionary inputs
            assert type(data) in _raw_types, "Unsupported datatype!"

        # Backwards-compatibility
        def convert_to_numpy():  # pragma: no cover
            self.data.update(self.map(np.asarray))

        self.convert_to_numpy = fannypack.utils.deprecation_wrapper(
            "SliceWrapper.convert_to_numpy() is deprecated -- please use "
            "the functional SliceWrapper.map() interface instead!",
            convert_to_numpy,
        )

    def __getitem__(self, index: Any) -> Any:
        """Unified interface for indexing into our wrapped object; shorthand for
        `SliceWrapper.map(lambda v: v[index])`.

        For wrapped dictionaries, this returns a new (un-wrapped) dictionary with the
        index applied value-wise.
        Thus:
        ```
        SliceWrapper({
            "a": a,
            "b": b,
        })[index]
        ```
        would return:
        ```
        {
            "a": a[index],
            "b": b[index],
        }
        ```

        For iterables that are directly wrapped, this is equivalent to evaluating
        `data[index]`.
        Thus..
        ```
        SliceWrapper([1, 2, 3])[::-1]
        ```
        would return:
        ```
        [1, 2, 3][::-1]
        ```

        Args:
            index (Any): Index. Can be a slice, tuple, boolean array, etc.

        Returns:
            Any: Indexed value. See overall function docstring.
        """
        return cast(WrappedType, self.map(lambda v: v[index]))

    def __len__(self) -> int:
        """Unified interface for evaluating the length of a wrapped object.

        Equivalent to `SliceWrapper.shape[0]`.

        Returns:
            int: Length of wrapped object.
        """
        return self.shape[0]

    def __iter__(self):
        """Iterable __iter__() interface."""
        self._iter_index = 0
        return self

    def __next__(self):
        """Iterable __next__() interface."""
        try:
            output = self[self._iter_index]
            self._iter_index += 1
            return output
        except IndexError:
            pass
        raise StopIteration

    def append(self, other: Any) -> None:
        """Append to the end of our data object.

        Only supported for wrapped lists and dictionaries containing lists.

        For wrapped lists, this is equivalent to `data.append(other)`.

        For dictionaries, `other` should be a dictionary.
        Behavior example:
        ```
        # Data before append
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}

        # Value of other
        {"a": 5, "b": 3}

        # Data after append
        {"a": [1, 2, 3, 4, 5], "b": [5, 6, 7, 8, 3]}
        ```

        Args:
            other (Any): Object to append.
        """
        if type(self.data) == dict:
            assert type(other) == dict, "Appended object must be a dictionary"

            # Cast for type-checking
            data_dict = cast(dict, self.data)
            other_dict = cast(dict, other)

            for key, value in other_dict.items():
                if key in data_dict.keys():
                    assert (
                        type(data_dict[key]) == list
                    ), "Append is only supported for wrapped lists"
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
        elif type(self.data) is list:
            cast(List, self.data).append(other)
        else:
            assert False, "Append is only supported for wrapped lists"

    def extend(self, other: WrappedType) -> None:
        """Extend to the end of our data object.

        Only supported for wrapped lists and dictionaries containing lists.

        For wrapped lists, this is equivalent to `data.extend(other)`.

        For dictionaries, `other` should be a dictionary.
        Behavior example:
        ```
        # Data before extend
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}

        # Value of other
        {"a": [5], "b": [3]}

        # Data after extend
        {"a": [1, 2, 3, 4, 5], "b": [5, 6, 7, 8, 3]}
        ```

        Args:
            other (dict or list): Object to append.
        """
        if type(self.data) == dict:
            assert type(other) == dict

            # Cast for type-checking
            data_dict = cast(dict, self.data)
            other_dict = cast(dict, other)

            for key, value in other_dict.items():
                if key in data_dict.keys():
                    assert (
                        type(data_dict[key]) == list
                    ), "Extend is only supported for wrapped lists"
                    data_dict[key].extend(value)
                else:
                    data_dict[key] = value
        elif type(self.data) == list:
            cast(list, self.data).extend(other)
        else:
            assert False, "Extend is only supported for wrapped lists"

    @overload
    def map(
        self: Union[
            "SliceWrapper[Dict[Any, List]]",
            "SliceWrapper[Dict[Any, Tuple]]",
            "SliceWrapper[Dict[Any, torch.Tensor]]",
            "SliceWrapper[Dict[Any, np.ndarray]]",
        ],
        function: Callable[[Any], MapOutputType],
    ) -> Dict[Any, MapOutputType]:
        pass

    @overload
    def map(
        self: Union[
            "SliceWrapper[List]",
            "SliceWrapper[Tuple]",
            "SliceWrapper[torch.Tensor]",
            "SliceWrapper[np.ndarray]",
        ],
        function: Callable[[Any], MapOutputType],
    ) -> MapOutputType:
        pass

    @overload
    def map(
        self, function: Callable[[Any], MapOutputType]
    ) -> Union[MapOutputType, Dict[Any, MapOutputType]]:
        pass

    def map(self, function):
        """Apply a function to all iterables within our wrapped data object.

        For iterables that are directly wrapped (eg lists), this is equivalent to
        evaluating:
        ```
        slice_wrapper: SliceWrapper[List]
        function(slice_wrapper.data)
        ```

        For dictionaries, `function` is applied value-wise.
        Thus, an input of:
        ```
        SliceWrapper({
            "a": [1, 2, 3],
            "b": [2, 4, 5],
        })
        ```
        would return:
        ```
        {
            "a": function([1, 2, 3]),
            "b": function([2, 4, 5]),
        }
        ```

        Args:
            function (Callable): Function to map.
        """
        if type(self.data) == dict:
            # Cast for type-checking
            data_dict = cast(dict, self.data)

            # Construct output
            mapped_data = {}
            for key, value in data_dict.items():
                mapped_data[key] = function(value)
            return mapped_data
        elif type(self.data) in _raw_types:
            return function(self.data)
        else:
            assert False, f"Unsupported data type: {type(self.data)}"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Unified interface for polling the shape of our wrapped object.

        For lists and tuples, this evaluates to `(len(data),)`.

        For Numpy arrays and torch tensors, we get `data.shape`.

        For dictionaries, we return a tuple containing all shared dimensions between
        our wrapped values, starting from the leftmost dimension.

        Args:

        Returns:
            Tuple[int, ...]:
        """
        if type(self.data) == dict:
            # Cast for type-checking
            data_dict = cast(dict, self.data)

            # Find longest shared shape prefix
            output: Tuple[int, ...]
            first = True
            for value in data_dict.values():
                shape = self._shape_helper(value)
                if first:
                    output = shape
                    first = False
                    continue

                for i in range(min(len(output), len(shape))):
                    if output[i] != shape[i]:
                        output = output[:i]
                        break

            return tuple(output)
        elif type(self.data) in _raw_types:
            return self._shape_helper(self.data)
        else:
            assert False, "Unsupported datatype!"

    @staticmethod
    def _shape_helper(data) -> Tuple[int, ...]:
        """Computes the shape of an object. `data.shape` for tensors and Numpy arrays,
        `(len(data))` for lists and tuples.
        """
        if type(data) in (torch.Tensor, np.ndarray):
            # Return full shape
            return data.shape
        elif type(data) in (list, tuple):
            # Return 1D shape
            return (len(data),)
        else:
            assert False, "Unsupported datatype!"
