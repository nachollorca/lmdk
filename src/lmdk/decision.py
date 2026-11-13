"""Contains the main logic to call decision (encoder) model APIs."""

from collections.abc import Sequence
from typing import Any

from lmdk.datatypes import DecisionBatch, DecisionRequest, DecisionResponse, Question
from lmdk.errors import AllModelsFailedError
from lmdk.provider import resolve_model
from lmdk.utils import parallelize_function


def decide(
    model: str | list[str],
    state: str | dict[str, Any],
    questions: dict[str, Question],
    calling_service: str | None = None,
) -> DecisionResponse:
    """Ask a decision model for the probability of each label of each question.

    Every question is evaluated independently against the same ``state`` in one request.
    System One servers render the whole ``state`` once per question, so its tokens are
    billed per question and must fit the per-question context limit. To score many
    independent items, use :func:`decide_batch` (one request per item) instead of
    putting them all in one ``state``.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"typesafe:jev-latest"``)
            or a list of identifiers to try in order as fallbacks.
        state: The content the questions are about. A dict keeps its structure, so
            questions can reference fields by path (e.g. ```policy.refund_window```).
        questions: Named questions; answers come back under the same names.
        calling_service: Optional name of the service calling the function.

    Returns:
        A ``DecisionResponse`` with ``{question: {label: probability}}`` and metadata.

    Raises:
        AllModelsFailedError: If every model in the list fails.
    """
    models = [model] if isinstance(model, str) else model

    errors: dict[str, Exception] = {}
    for m in models:
        try:
            _, model_id, provider = resolve_model(m)
            request = DecisionRequest(
                model_id=model_id,
                state=state,
                questions=questions,
                calling_service=calling_service,
            )
            return provider.decide(request)
        except Exception as exc:
            errors[m] = exc

    if len(errors) == 1:
        raise next(iter(errors.values()))
    raise AllModelsFailedError(errors)


def decide_batch(
    model: str | list[str],
    state_list: Sequence[str | dict[str, Any]],
    questions: dict[str, Question],
    max_workers: int = 10,
    calling_service: str | None = None,
) -> DecisionBatch:
    """Ask the same questions about each state in parallel, one request per state.

    This is the shape for scoring a collection: items don't compete with each
    other, so every request sees only its own state, and the caller ranks the
    answers in code. Each state is dispatched to :func:`decide` concurrently.

    Args:
        model: Provider-prefixed model identifier, or a list to try in order.
        state_list: One state per item to decide about.
        questions: Named questions, the same for every state.
        max_workers: Maximum number of concurrent threads.
        calling_service: Optional name of the service calling the function.

    Returns:
        A :class:`DecisionBatch` whose ``results`` has one entry per state, in the
        same order as *state_list*: a ``DecisionResponse`` on success or the
        ``Exception`` raised on failure.
    """
    shared_kwargs: dict[str, Any] = {
        "model": model,
        "questions": questions,
        "calling_service": calling_service,
    }
    params_list = [{**shared_kwargs, "state": state} for state in state_list]
    results = parallelize_function(
        function=decide,
        params_list=params_list,
        max_workers=max_workers,
        catch_exceptions=True,
    )
    return DecisionBatch(results=results)
