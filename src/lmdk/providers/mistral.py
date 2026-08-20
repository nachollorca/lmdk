"""Implements the provider to use models hosted in Mistral API."""

from lmdk.datatypes import CompletionRequest
from lmdk.providers._chat_completions import ChatCompletionsProvider

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Sampling controls Mistral rejects when ``reasoning_effort`` is enabled.
_REASONING_INCOMPATIBLE_KWARGS = ("temperature", "top_p")


class MistralProvider(ChatCompletionsProvider):
    """Provider for models hosted on the Mistral API.

    Mistral speaks the OpenAI ``/chat/completions`` protocol, so everything but
    authentication and the reasoning control comes from
    :class:`~lmdk.providers._chat_completions.ChatCompletionsProvider`.
    """

    required_env = "MISTRAL_API_KEY"
    base_url = MISTRAL_BASE_URL

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Mistral Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {credentials['MISTRAL_API_KEY']}"}

    @classmethod
    def _build_payload(cls, request: CompletionRequest, model: str, stream: bool = False) -> dict:
        """Add Mistral's ``reasoning_effort`` to the shared chat-completions payload.

        Mistral adjustable reasoning models only accept ``reasoning_effort``
        "none" or "high" (low/medium return 400), so any non-"none" lmdk
        ``thinking_effort`` maps to "high" and cross-provider effort levels still
        enable reasoning. A caller-provided ``reasoning_effort`` wins. Reasoning
        models also reject sampling controls like temperature/top_p (lmdk
        defaults temperature to 0), so those are dropped.
        """
        payload = super()._build_payload(request, model, stream=stream)
        if request.thinking_effort != "none":
            for key in _REASONING_INCOMPATIBLE_KWARGS:
                payload.pop(key, None)
            payload.setdefault("reasoning_effort", "high")
        return payload

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return ``reasoning_effort`` from the outbound Mistral payload."""
        effort = cls._build_payload(request, request.model_id).get("reasoning_effort")
        return effort if effort is not None else "none"
