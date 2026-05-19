"""Observation hook for capturing completions made inside a code block.

This is useful when a caller wraps user code that ends up calling
:func:`lmdk.complete` and wants to inspect the resulting
:class:`CompletionResponse` (and its originating :class:`CompletionRequest`)
without forcing the user code to return it.

The mechanism is a :mod:`contextvars`-scoped recorder. While a call site is
inside an :func:`observe` block, every non-streaming completion produced by
``lmdk.complete`` is appended to the active observer. Streaming calls are
ignored — there is no single response to record.

Example:
    >>> with observe() as obs:
    ...     answer = my_function_that_calls_complete()
    >>> obs.responses[0].latency
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from lmdk.datatypes import CompletionResponse


class CompletionObserver:
    """Collects :class:`CompletionResponse` objects produced while active.

    Responses are appended in the order ``lmdk.complete`` returns them. The
    originating :class:`CompletionRequest` is always attached to
    ``response.request`` when an observer is active, regardless of the
    ``return_request`` flag — observers usually need it (for the rendered
    prompt, generation kwargs, etc.).
    """

    def __init__(self) -> None:
        self.responses: list[CompletionResponse] = []

    def _record(self, response: CompletionResponse) -> None:
        self.responses.append(response)


_active_observer: ContextVar[CompletionObserver | None] = ContextVar(
    "lmdk_active_observer", default=None
)


@contextmanager
def observe() -> Iterator[CompletionObserver]:
    """Capture every completion produced by ``lmdk.complete`` inside the block.

    Yields a fresh :class:`CompletionObserver` whose ``responses`` list is
    populated as completions finish. Observers nest: an inner ``observe``
    block shadows the outer one for the duration of its scope, so inner
    completions land on the inner observer only.

    The recorder is stored in a :class:`~contextvars.ContextVar`, so each
    thread / async task sees its own active observer. Completions made on
    threads spawned inside the block do **not** automatically inherit the
    observer unless the caller propagates the context explicitly.
    """
    observer = CompletionObserver()
    token = _active_observer.set(observer)
    try:
        yield observer
    finally:
        _active_observer.reset(token)


def _current_observer() -> CompletionObserver | None:
    """Return the active observer, or ``None`` if no ``observe`` block is open."""
    return _active_observer.get()
