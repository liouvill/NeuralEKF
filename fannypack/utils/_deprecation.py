import warnings
from typing import Callable, Optional, TypeVar, cast

CallableType = TypeVar("CallableType", bound=Callable)


def deprecation_wrapper(message: str, function_or_class: CallableType) -> CallableType:
    """Creates a wrapper for a deprecated function or class. Prints a warning
    the first time a function or class is called.

    Args:
        message (str): Warning message.
        function_or_class (CallableType): Function or class to wrap.

    Returns:
        CallableType: Wrapped function/class.
    """

    warned = False

    def curried(*args, **kwargs):  # pragma: no cover
        nonlocal warned
        if not warned:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            warned = True
        return function_or_class(*args, **kwargs)

    return cast(CallableType, curried)


def new_name_wrapper(
    old_name: str, new_name: str, function_or_class: CallableType
) -> CallableType:
    """Creates a wrapper for a renamed function or class. Prints a warning the first
    time a function or class is called with the old name.

    Args:
        old_name (str): Old name of function or class. Printed in warning.
        new_name (str): New name of function or class. Printed in warning.
        function_or_class (CallableType): Function or class to wrap.

    Returns:
        CallableType: Wrapped function/class.
    """
    return deprecation_wrapper(
        f"{old_name} is deprecated! Use {new_name} instead.", function_or_class
    )
