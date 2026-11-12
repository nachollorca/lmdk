"""Vendor for models hosted in the Anthropic API.

Subclasses the ``messages`` shape with the hosted default location and the
``ANTHROPIC_API_KEY`` credential; an ``@<address>`` suffix overrides the
default location (then only the optional ``LOCAL_API_KEY`` is sent, in the
``x-api-key`` header format, never the vendor key)::

    complete("anthropic:claude-sonnet-4", "Hello!")

Vendor quirk: newer Anthropic models reject sampling params, and lmdk's global
``temperature=0`` default would otherwise trigger HTTP 400, so sampling
controls are dropped for them.
"""

from lmdk.datatypes import CompletionRequest
from lmdk.providers.shapes.messages import (
    ANTHROPIC_VERSION,
    DEFAULT_MAX_TOKENS,
    MessagesProvider,
    _prepare_schema,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Sampling controls newer Anthropic models reject.
_SAMPLING_RESTRICTED_KWARGS = {"temperature", "top_p", "top_k"}

__all__ = [
    "ANTHROPIC_API_URL",
    "ANTHROPIC_VERSION",
    "DEFAULT_MAX_TOKENS",
    "AnthropicProvider",
    "_prepare_schema",
]


class AnthropicProvider(MessagesProvider):
    """Provider for models hosted on the Anthropic API."""

    required_env = "ANTHROPIC_API_KEY"
    base_url = ANTHROPIC_BASE_URL

    @classmethod
    def _normalize_generation_kwargs(cls, request: CompletionRequest) -> dict:
        kwargs = super()._normalize_generation_kwargs(request)
        # Newer Anthropic models reject sampling params; lmdk's global temperature=0
        # default would otherwise trigger HTTP 400.
        for key in _SAMPLING_RESTRICTED_KWARGS:
            kwargs.pop(key, None)
        return kwargs
