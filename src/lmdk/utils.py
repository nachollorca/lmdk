"""General-purpose utility helpers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
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


def render_template(
    template: str | None = None,
    path: str | Path | None = None,
    *args,
    **kwargs,
) -> str:
    """Renders a Jinja2 template from a string or a file path.

    Ensures that double curly braces in the input are removed to avoid rendering issues.
    All string variables are stripped of leading/trailing whitespace.

    Args:
        template (str): The Jinja2 template string.
        path (str or Path): The path to a template file.
        *args: Positional arguments to pass to the template.
        **kwargs: Keyword arguments to pass to the template.

    Returns:
        str: The rendered template string.

    Raises:
        ValueError: If neither template nor path is provided, or if both are provided.
    """
    if template is not None and path is not None:
        raise ValueError("Provide either 'template' or 'path', not both.")
    if template is None and path is None:
        raise ValueError("Must provide either 'template' or 'path'.")

    from jinja2 import Template  # lazy load

    template_content = Path(path).read_text() if path is not None else template

    processed_args = [
        arg.replace("{{", "").replace("}}", "").strip() if isinstance(arg, str) else arg
        for arg in args
    ]
    processed_kwargs = {
        k: v.replace("{{", "").replace("}}", "").strip() if isinstance(v, str) else v
        for k, v in kwargs.items()
    }
    assert template_content is not None
    jinja_template = Template(template_content)
    return jinja_template.render(*processed_args, **processed_kwargs)
