"""The System One wire format (``/v1/systemone``), addressed via ``@<address>``.

System One is the decision-model (encoder) API used by TypeSafe's Jev and
Fastino's GLiDE, and by self-hosted servers that copy it (e.g. ``laya-serve``).
Each :class:`~lmdk.datatypes.Question` maps to one of its question types:

- labels that are a yes/no (or true/false) pair -> ``noul``
- ``is_ordered=True`` -> ``score`` (levels in label order)
- anything else -> ``choice``

Answers are mapped back to ``{label: probability}`` for every type.

The bare shape is addressed explicitly through the model identifier's ``@``
suffix::

    decide("systemone:english@192.168.10.51:8100", state=..., questions=...)

The address is mandatory for the bare shape and ``http://`` is assumed when it
has no scheme. Servers that gate access with a token may set the optional
``LOCAL_API_KEY`` environment variable, which is sent as a ``Bearer``
credential. The ``typesafe`` and ``fastino`` vendors subclass this shape with
the hosted default locations and their own API keys.

See https://docs.typesafe.ai/api.
"""

from typing import Any

from lmdk.datatypes import DecisionRequest, DecisionResponse, Question
from lmdk.providers.shapes._base import ShapeProvider

_YES = {"yes", "true"}
_NO = {"no", "false"}


def _noul_labels(labels: dict[str, str]) -> tuple[str, str] | None:
    """Return ``(yes_label, no_label)`` when labels are a yes/no pair, else ``None``."""
    if len(labels) != 2:
        return None
    yes = next((k for k in labels if k.lower() in _YES), None)
    no = next((k for k in labels if k.lower() in _NO), None)
    return (yes, no) if yes and no else None


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


class SystemoneProvider(ShapeProvider):
    """Provider for any endpoint speaking the System One API (``/v1/systemone``)."""

    path = "/v1/systemone"

    @classmethod
    def _score_criteria(cls, labels: dict[str, str]) -> list[Any]:
        """Score criteria as an ordered list of ``{label, description}`` objects."""
        return [{"label": k, "description": v} for k, v in labels.items()]

    @classmethod
    def _to_wire(cls, question: Question) -> dict[str, Any]:
        """Map a ``Question`` to a System One question object."""
        if noul := _noul_labels(question.labels):
            yes, no = noul
            criteria = {"true": question.labels[yes], "false": question.labels[no]}
            return {"type": "noul", "instructions": question.text, "criteria": criteria}
        if question.is_ordered:
            return {
                "type": "score",
                "instructions": question.text,
                "criteria": cls._score_criteria(question.labels),
            }
        return {"type": "choice", "instructions": question.text, "criteria": question.labels}

    @classmethod
    def _send_decision_request(
        cls, request: DecisionRequest, credentials: dict[str, str]
    ) -> DecisionResponse:
        model, location = cls._parse_model_id(request.model_id)
        body = {
            "model": model,
            "state": request.state,
            "questions": {name: cls._to_wire(q) for name, q in request.questions.items()},
        }
        response = cls._make_request(
            cls._build_url(location), json=body, headers=cls._build_auth_headers(credentials)
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
