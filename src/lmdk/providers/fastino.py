"""Provider for Fastino's GLiDE models (``POST https://api.fastino.ai/v1/systemone``).

GLiDE speaks the same System One wire protocol as TypeSafe's Jev, so this reuses
:class:`~lmdk.providers.typesafe.TypesafeProvider` and only changes the endpoint,
credentials, model id, Score criteria shape, and retry behavior:

    ``fastino:glide``

``glide`` is mapped to the hosted ``fastino/GLiDE`` model; any other model id is
sent as-is (e.g. the id of a GLiDE model you deployed).

Differences from TypeSafe per https://docs.fastino.ai/inference/systemone:

- auth is ``X-API-Key: $FASTINO_API_KEY``
- Score ``criteria`` is an ordered array of level descriptions, not label/description objects
- retry ``425`` (model warming), ``429`` and ``503`` with bounded backoff
- use a read timeout of at least 300 seconds
"""

from typing import Any

from lmdk.providers.typesafe import TypesafeProvider

_BASE_URL = "https://api.fastino.ai"
_MODEL_ALIASES = {"glide": "fastino/GLiDE"}


class FastinoProvider(TypesafeProvider):
    """Provider for Fastino's hosted GLiDE models (e.g. ``fastino:glide``)."""

    required_env = "FASTINO_API_KEY"
    retry_statuses = frozenset({425, 429, 503})
    request_timeout = 300.0

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        return {"X-API-Key": credentials["FASTINO_API_KEY"]}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Map ``glide`` to the hosted ``fastino/GLiDE`` model, pass other ids through."""
        return _MODEL_ALIASES.get(model_id, model_id), _BASE_URL

    @classmethod
    def _score_criteria(cls, labels: dict[str, str]) -> list[Any]:
        """GLiDE wants Score criteria as an ordered array of level descriptions."""
        return list(labels.values())
