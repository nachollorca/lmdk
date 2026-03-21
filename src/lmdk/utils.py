"""General-purpose utility helpers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any


def return_if_exception(func: Callable) -> Callable:
    """Decorator that catches exceptions and returns them instead of raising.

    Args:
        func: The function to wrap.

    Returns:
        A wrapped version that returns any raised ``Exception`` as a value.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any | Exception:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e

    return wrapper


def parallelize_function(
    function: Callable,
    params_list: list[dict[str, Any]],
    max_workers: int = 10,
    catch_exceptions: bool = False,
) -> list[Any]:
    """Execute *function* in parallel for each parameter set using threads.

    Args:
        function: The function to call for each parameter set.
        params_list: A list of keyword-argument dicts, one per call.
        max_workers: Maximum number of concurrent threads.
        catch_exceptions: If ``True``, failed calls return the
            ``Exception`` instance instead of propagating it.

    Returns:
        A list of results in the same order as *params_list*.

    Examples:
        >>> def add(a, b):
        ...     return a + b

        >>> params_list = [
        ...     {"a": 1, "b": 2},
        ...     {"a": 10, "b": 20},
        ...     {"a": 100, "b": 200},
        ... ]

        >>> parallelize_function(add, params_list)
        [3, 30, 300]
    """
    if catch_exceptions:
        function = return_if_exception(function)

    results: list[Any] = [None] * len(params_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_indices = {
            executor.submit(function, **params): index for index, params in enumerate(params_list)
        }
        for future in as_completed(futures_to_indices):
            index = futures_to_indices[future]
            results[index] = future.result()
    return results
