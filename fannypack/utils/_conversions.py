import functools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import torch

# Note that we define two container TypeVars:
# - A bound container, whose output type exactly matches the input type
#       -> For moving across devices, for example, a List[torch.Tensor] will always
#          return a List[torch.Tensor]
# - And a constrained container, whose output type is only guaranteed to match at the
#   top-level
#       -> For converting NumPy -> Torch, a List[np.ndarray] will return a
#          List[torch.Tensor]; this is really difficult to represent in Python, so we
#          just mark the output as a List[Any]

BoundContainer = TypeVar("BoundContainer", bound=Union[Dict, List, Tuple])
Key = TypeVar("Key")

_ConstrainedContainer = TypeVar("_ConstrainedContainer", Dict, List, Tuple)
_InputType = TypeVar("_InputType", np.ndarray, torch.Tensor)
_OutputType = TypeVar("_OutputType", np.ndarray, torch.Tensor)


@overload
def _convert_recursive(
    x: _InputType,
    convert: Callable[[_InputType], _OutputType],
    input_type: Type,
) -> _OutputType:
    ...


@overload
def _convert_recursive(
    x: _ConstrainedContainer,
    convert: Callable[[_InputType], _OutputType],
    input_type: Type,
) -> _ConstrainedContainer:
    ...


def _convert_recursive(x, convert, input_type):
    """Private conversion helper. Recursively calls a conversion function on inputs
    within a nested set of containers.
    """

    # Conversion base case
    if isinstance(x, input_type):
        x = cast(_InputType, x)
        return convert(x)

    # Convert containers: bind arguments to helper function
    convert_recursive = functools.partial(
        _convert_recursive, convert=convert, input_type=input_type
    )

    # Convert dictionaries of values
    if isinstance(x, dict):
        x = cast(dict, x)
        return dict(zip(x.keys(), map(convert_recursive, x.values())))

    # Convert lists of values
    if isinstance(x, list):
        x = cast(list, x)
        return list(map(convert_recursive, x))

    # Convert tuples of values
    if isinstance(x, tuple):
        x = cast(tuple, x)
        if hasattr(x, "_fields"):  # NamedTuple
            return type(x)(*map(convert_recursive, x))
        else:
            return tuple(map(convert_recursive, x))

    # Unsupported input types
    assert False, f"Unsupported datatype {type(x)}!"


@overload
def to_device(
    x: torch.Tensor, device: torch.device, detach: bool = False
) -> torch.Tensor:
    ...


@overload
def to_device(
    x: BoundContainer, device: torch.device, detach: bool = False
) -> BoundContainer:
    ...


def to_device(x, device, detach=False):
    """Move a torch tensor, list, tuple (standard or named), or dict of tensors to a
    different device. Recursively applied for nested containers.

    Args:
        x (torch.Tensor, list, tuple (standard or named), or dict): Tensor or container
            of tensors to move.
        device (torch.device): Target device.
        detach (bool, optional): If set, detaches tensors after moving. Defaults to
            False.

    Returns:
        torch.Tensor, list, tuple (standard or named), or dict: Output, type will mirror
        input.
    """

    def convert(x):
        if detach:
            x = x.detach()
        return x.to(device)

    return _convert_recursive(x, convert=convert, input_type=torch.Tensor)


# This a lot of boilerplate, but seems impossible to reproduce same level of specificity
# with TypeVars


@overload
def to_torch(
    x: np.ndarray,
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> torch.Tensor:
    ...


@overload
def to_torch(
    x: List[np.ndarray],
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> List[torch.Tensor]:
    ...


@overload
def to_torch(
    x: List,
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> List:
    ...


@overload
def to_torch(
    x: Tuple[np.ndarray, ...],
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> Tuple[torch.Tensor, ...]:
    ...


@overload
def to_torch(
    x: Tuple,
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> Tuple:
    ...


@overload
def to_torch(
    x: Dict[Key, np.ndarray],
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> Dict[Key, torch.Tensor]:
    ...


@overload
def to_torch(
    x: Dict[Key, Any],
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> Dict[Key, Any]:
    ...


def to_torch(
    x,
    device="cpu",
    convert_doubles_to_floats=True,
):
    """Converts a NumPy array, list, tuple (standard or named), or dict of NumPy arrays
    for use in PyTorch.  Recursively applied for nested containers.

    Args:
        x (np.ndarray, list, tuple (standard or named), or dict): Array or container of
            arrays to convert to torch tensors.
        device (torch.device, optional): Torch device to create tensors on. Defaults to
            `"cpu"`.
        convert_doubles_to_floats (bool, optional): If set, converts 64-bit floats to
            32-bit. Defaults to True.

    Returns:
        torch.Tensor, list, tuple (standard or named), or dict: Output, type will mirror input.
    """

    def convert(x: np.ndarray) -> torch.Tensor:
        output = torch.from_numpy(x)
        if x.dtype == np.float64 and convert_doubles_to_floats:
            output = output.float()
        output = output.to(device)
        return output

    return _convert_recursive(x, convert=convert, input_type=np.ndarray)


@overload
def to_numpy(x: torch.Tensor) -> np.ndarray:
    ...


@overload
def to_numpy(x: List[torch.Tensor]) -> List[np.ndarray]:
    ...


@overload
def to_numpy(x: List) -> List:
    ...


@overload
def to_numpy(x: Tuple[torch.Tensor, ...]) -> Tuple[np.ndarray, ...]:
    ...


@overload
def to_numpy(x: Tuple) -> Tuple:
    ...


@overload
def to_numpy(x: Dict[Key, torch.Tensor]) -> Dict[Key, np.ndarray]:
    ...


@overload
def to_numpy(x: Dict[Key, Any]) -> Dict[Key, Any]:
    ...


def to_numpy(x):
    """Converts a tensor, list, tuple (standard or named), or dict of tensors for use in
    Numpy. Recursively applied for nested containers.

    Args:
        x (torch.Tensor, list, tuple (standard or named), or dict): Tensor or container
            of tensors to convert to NumPy.

    Returns:
        np.ndarray, list, tuple (standard or named), or dict: Output, type will mirror input.
    """

    def convert(x: torch.Tensor) -> np.ndarray:
        output = x.detach().cpu().numpy()
        return output

    return _convert_recursive(x, convert=convert, input_type=torch.Tensor)
