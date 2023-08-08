import contextlib
import time
from typing import Generator

import termcolor

_stopwatch_nest_counter: int = 0


@contextlib.contextmanager
def stopwatch(label: str = "block") -> Generator[None, None, None]:
    """Helper for printing the runtime of a block of code.

    Example:
    ```
    with fannypack.utils.stopwatch("name"):
        time.sleep(1.0)
    ```

    Args:
        label (str): Label for block that's running.

    Returns:
        Generator: Context manager to place in `with` statement.
    """
    start_time = time.time()

    def print_red(*args, **kwargs):
        if len(args) > 0:
            global _stopwatch_nest_counter
            prefix = ("    " * _stopwatch_nest_counter) + f"[stopwatch: {label}] "
            args = (termcolor.colored(prefix + args[0], color="yellow"),) + args[1:]
        print(*args, **kwargs)

    print_red("Starting!")

    global _stopwatch_nest_counter
    _stopwatch_nest_counter += 1
    yield
    _stopwatch_nest_counter -= 1

    print_red(
        f"Completed in {termcolor.colored(str(time.time() - start_time) + ' seconds', attrs=['bold'])}"
    )
