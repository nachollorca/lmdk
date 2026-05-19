"""Internal hook seam dispatched around each non-streaming completion call.

This module gives :mod:`lmdk.core` a single place to plug cross-cutting
concerns (telemetry, observation, future cost tracking, etc.) without growing
ad-hoc branches in ``_complete_model``. Each concern lives in its own module
and is composed here.

Currently composed:

- :func:`lmdk.telemetry.traced_completion` — OpenTelemetry span + metrics,
  opt-in via the ``LMDK_TELEMETRY`` environment variable.
- :func:`lmdk.observe._current_observer` — user-facing ``observe()`` block
  that records :class:`CompletionResponse` objects for inspection.

To add another concern, extend :func:`completion_lifecycle` rather than
threading new arguments through ``_complete_model``.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager

from lmdk.datatypes import CompletionRequest, CompletionResponse
from lmdk.observe import _current_observer
from lmdk.telemetry import traced_completion


@contextmanager
def completion_lifecycle(
    provider_name: str,
    model_id: str,
    request: CompletionRequest,
    fallback_index: int,
) -> Generator[Callable[[CompletionResponse], None]]:
    """Open all hooks for a single non-streaming completion attempt.

    Yields a ``record`` callable that ``_complete_model`` invokes once with
    the final :class:`CompletionResponse`. Exceptions raised inside the block
    propagate through ``traced_completion`` so the telemetry span captures
    them.
    """
    observer = _current_observer()
    with traced_completion(provider_name, model_id, request, fallback_index) as telemetry:

        def record(response: CompletionResponse) -> None:
            telemetry.record_response(response)
            if observer is not None:
                observer._record(request, response)

        yield record
