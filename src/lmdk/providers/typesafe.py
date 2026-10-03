"""Provider for TypeSafe's System One API (``POST /v1/systemone``).

Each :class:`~lmdk.datatypes.Question` maps to one of TypeSafe's question types:

- labels that are a yes/no (or true/false) pair -> ``noul``
- ``is_ordered=True`` -> ``score`` (levels in label order)
- anything else -> ``choice``

Answers are mapped back to ``{label: probability}`` for every type.

See https://docs.typesafe.ai/api.
"""

from typing import Any

from lmdk.datatypes import DecisionRequest, DecisionResponse, Question
from lmdk.provider import Provider

_BASE_URL = "https://api.typesafe.ai"
_YES = {"yes", "true"}
_NO = {"no", "false"}


def _noul_labels(labels: dict[str, str]) -> tuple[str, str] | None:
    """Return ``(yes_label, no_label)`` when labels are a yes/no pair, else ``None``."""
    if len(labels) != 2:
        return None
    yes = next((k for k in labels if k.lower() in _YES), None)
    no = next((k for k in labels if k.lower() in _NO), None)
    return (yes, no) if yes and no else None


def _to_wire(question: Question) -> dict[str, Any]:
    """Map a ``Question`` to a System One question object."""
    if noul := _noul_labels(question.labels):
        yes, no = noul
        criteria = {"true": question.labels[yes], "false": question.labels[no]}
        return {"type": "noul", "instructions": question.text, "criteria": criteria}
    if question.is_ordered:
        levels = [{"label": k, "description": v} for k, v in question.labels.items()]
        return {"type": "score", "instructions": question.text, "criteria": levels}
    return {"type": "choice", "instructions": question.text, "criteria": question.labels}


def _from_wire(question: Question, answer: dict[str, Any]) -> dict[str, float]:
    """Map a System One answer back to ``{label: probability}``."""
    if answer["type"] == "noul":
        # a noul answer only comes back for a yes/no question, so this is never None
        yes, no = _noul_labels(question.labels)  # ty: ignore[not-iterable]
        return {yes: answer["noul"], no: 1 - answer["noul"]}
    if answer["type"] == "score":
        labels = list(question.labels)
        return {labels[int(i)]: p for i, p in answer["probabilities"].items()}
    return answer["probabilities"]


class TypesafeProvider(Provider):
    """Provider for TypeSafe's hosted System One models (e.g. ``typesafe:jev-latest``)."""

    required_env = "TYPESAFE_API_KEY"

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        return {"Authorization": f"Bearer {credentials['TYPESAFE_API_KEY']}"}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Return ``(model, base_url)``. Overridden by self-hosted providers."""
        return model_id, _BASE_URL

    @classmethod
    def _send_decision_request(
        cls, request: DecisionRequest, credentials: dict[str, str]
    ) -> DecisionResponse:
        model, base_url = cls._parse_model_id(request.model_id)
        body = {
            "model": model,
            "state": request.state,
            "questions": {name: _to_wire(q) for name, q in request.questions.items()},
        }
        response = cls._make_request(
            f"{base_url}/v1/systemone", json=body, headers=cls._build_auth_headers(credentials)
        )
        data = response.json()
        usage = data.get("usage") or {}
        return DecisionResponse(
            probabilities={
                name: _from_wire(q, data["answers"][name]) for name, q in request.questions.items()
            },
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
