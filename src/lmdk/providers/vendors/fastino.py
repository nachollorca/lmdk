"""Vendor for Fastino's GLiDE models (hosted ``POST https://api.fastino.ai/v1/systemone``).

Subclasses the ``systemone`` shape with the hosted default location and the
``FASTINO_API_KEY`` credential; an ``@<address>`` suffix overrides the default
location -- e.g. a self-hosted GLiDE server -- and then only the optional
``LOCAL_API_KEY`` is sent, in Fastino's ``X-API-Key`` header format, never the
vendor key::

    decide("fastino:glide@192.168.10.51:8100", state=..., questions=...)

``glide`` is mapped to the hosted ``fastino/GLiDE`` model; any other model id is
sent as-is (e.g. the id of a GLiDE model you deployed).

Differences from TypeSafe per https://docs.fastino.ai/inference/systemone:

- auth header is ``X-API-Key``
- Score ``criteria`` is an ordered array of level descriptions, not label/description objects
- retry ``425`` (model warming), ``429`` and ``503`` with bounded backoff
- use a read timeout of at least 300 seconds
- accepts ``store: false`` to keep the request out of storage (TypeSafe rejects it)
"""

from typing import Any

from lmdk.providers.shapes.systemone import SystemoneProvider

FASTINO_BASE_URL = "https://api.fastino.ai"
_MODEL_ALIASES = {"glide": "fastino/GLiDE"}


class FastinoProvider(SystemoneProvider):
    """Provider for Fastino's hosted GLiDE models (e.g. ``fastino:glide``)."""

    required_env = "FASTINO_API_KEY"
    base_url = FASTINO_BASE_URL
    retry_statuses = frozenset({425, 429, 503})
    request_timeout = 300.0
    # The state is often personal data (a CV, a mailbox); don't leave it on their side.
    store_requests = False

    @classmethod
    def _key_headers(cls, api_key: str) -> dict:
        """Fastino's auth header format is ``X-API-Key``."""
        return {"X-API-Key": api_key}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Map ``glide`` to the hosted ``fastino/GLiDE`` model, pass other ids through."""
        model, location = super()._parse_model_id(model_id)
        return _MODEL_ALIASES.get(model, model), location

    @classmethod
    def _score_criteria(cls, labels: dict[str, str]) -> list[Any]:
        """GLiDE wants Score criteria as an ordered array of level descriptions."""
        return list(labels.values())
