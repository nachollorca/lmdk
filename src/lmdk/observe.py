"""Observation hook for capturing completions made inside a code block.

This is useful when a caller wraps user code that ends up calling
:func:`lmdk.complete` and wants to inspect the resulting
:class:`CompletionRequest` / :class:`CompletionResponse` pairs without
forcing the user code to return them.

The mechanism is a :mod:`contextvars`-scoped recorder. While a call site is
inside an :func:`observe` block, every non-streaming completion produced by
``lmdk.complete`` is appended to the active observer as a
:class:`CompletionRecord`. Streaming calls are ignored — there is no single
response to record.

Example:
    >>> with observe() as obs:
    ...     answer = my_function_that_calls_complete()
    >>> obs.records[0].response.latency
    >>> obs.records[0].request.prompt
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from lmdk.datatypes import CompletionRequest, CompletionResponse


@dataclass(frozen=True)
class CompletionRecord:
    """A single (request, response) pair captured by an :func:`observe` block."""

    request: CompletionRequest
    response: CompletionResponse


class CompletionObserver:
    """Collects :class:`CompletionRecord` entries produced while active.

    Records are appended in the order ``lmdk.complete`` returns them.
    """

    def __init__(self) -> None:
        self.records: list[CompletionRecord] = []

    def _record(self, request: CompletionRequest, response: CompletionResponse) -> None:
        self.records.append(CompletionRecord(request=request, response=response))


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
