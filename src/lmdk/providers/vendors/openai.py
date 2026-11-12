"""Vendor for models hosted in the OpenAI API.

Subclasses the ``responses`` shape with the hosted default location and the
``OPENAI_API_KEY`` credential; an ``@<address>`` suffix overrides the default
location (then only the optional ``LOCAL_API_KEY`` is sent, never the vendor
key)::

    complete("openai:gpt-5.5", "Hello!")

Vendor quirks per OpenAI's API: newer reasoning/GPT-5 models reject custom
sampling controls (``temperature`` / ``top_p``) and very small
``max_output_tokens`` values, so those are dropped/raised for them.
"""

from lmdk.datatypes import CompletionRequest
from lmdk.providers.shapes.responses import ResponsesProvider

OPENAI_BASE_URL = "https://api.openai.com"
OPENAI_API_URL = "https://api.openai.com/v1/responses"

_TEMPERATURE_RESTRICTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_TEMPERATURE_RESTRICTED_KWARGS = {"temperature", "top_p"}
_MIN_MAX_OUTPUT_TOKENS = 16


class OpenaiProvider(ResponsesProvider):
    """Provider for models hosted on the OpenAI API."""

    required_env = "OPENAI_API_KEY"
    base_url = OPENAI_BASE_URL

    @classmethod
    def _normalize_generation_kwargs(cls, request: CompletionRequest) -> dict:
        """Apply OpenAI-specific restrictions on top of the shape's normalization.

        ``lmdk.completion.complete`` defaults to ``temperature=0`` for every
        provider, but newer OpenAI reasoning/GPT-5 models reject custom
        sampling controls and only allow their defaults. For those models we
        drop unsupported sampling keys so a normal call succeeds. Very small
        ``max_output_tokens`` values are raised to OpenAI's minimum.
        """
        kwargs = super()._normalize_generation_kwargs(request)

        if "max_output_tokens" in kwargs and kwargs["max_output_tokens"] < _MIN_MAX_OUTPUT_TOKENS:
            kwargs["max_output_tokens"] = _MIN_MAX_OUTPUT_TOKENS

        if request.model_id.startswith(_TEMPERATURE_RESTRICTED_MODEL_PREFIXES):
            for key in _TEMPERATURE_RESTRICTED_KWARGS:
                kwargs.pop(key, None)

        return kwargs


# Backwards-/ergonomics-friendly alias for users who prefer the official casing.
OpenAIProvider = OpenaiProvider
