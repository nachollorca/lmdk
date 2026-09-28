"""Contains the main logic to call decision (encoder) model APIs."""

from typing import Any

from lmdk.datatypes import DecisionRequest, DecisionResponse, Question
from lmdk.errors import AllModelsFailedError
from lmdk.provider import resolve_model


def decide(
    model: str | list[str],
    state: str | dict[str, Any],
    questions: dict[str, Question],
    calling_service: str | None = None,
) -> DecisionResponse:
    """Ask a decision model for the probability of each label of each question.

    Every question is evaluated independently against the same ``state`` in one request.

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
