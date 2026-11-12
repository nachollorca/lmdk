"""Vendor for models hosted in the Mistral API.

Subclasses the ``chat`` shape with the hosted default location and the
``MISTRAL_API_KEY`` credential; an ``@<address>`` suffix overrides the default
location (then only the optional ``LOCAL_API_KEY`` is sent as a ``Bearer``
token, never the vendor key).

Mistral speaks the OpenAI ``/chat/completions`` protocol, so everything but
authentication and the reasoning control comes from the ``chat`` shape.
"""

from lmdk.datatypes import CompletionRequest
from lmdk.providers.shapes.chat import ChatProvider

MISTRAL_BASE_URL = "https://api.mistral.ai"


class MistralProvider(ChatProvider):
    """Provider for models hosted on the Mistral API."""

    required_env = "MISTRAL_API_KEY"
    base_url = MISTRAL_BASE_URL

    @classmethod
    def _build_payload(cls, request: CompletionRequest, model: str, stream: bool = False) -> dict:
        """Add Mistral's ``reasoning_effort`` to the shared chat-completions payload.

        Mistral adjustable reasoning models only accept ``reasoning_effort``
        "none" or "high" (low/medium return 400), so any non-"none" lmdk
        ``thinking_effort`` maps to "high" and cross-provider effort levels still
        enable reasoning. A caller-provided ``reasoning_effort`` wins.

        Greedy sampling (lmdk defaults ``temperature`` to 0) is rejected unless
        ``top_p`` is explicitly 1, because Mistral's server-side default top_p
        is lower; send it so the default call goes through.
        """
        payload = super()._build_payload(request, model, stream=stream)
        if request.thinking_effort != "none":
            payload.setdefault("reasoning_effort", "high")
        if payload.get("temperature") == 0:
            payload.setdefault("top_p", 1)
        return payload

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return ``reasoning_effort`` from the outbound Mistral payload."""
        effort = cls._build_payload(request, request.model_id).get("reasoning_effort")
        return effort if effort is not None else "none"
