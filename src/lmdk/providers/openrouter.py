"""Provider for OpenRouter's unified, OpenAI-compatible chat-completions API.

OpenRouter is a gateway that exposes hundreds of models from many providers
behind a single OpenAI ``/chat/completions`` wire protocol at a fixed hosted
base URL, gated by an ``OPENROUTER_API_KEY`` bearer token. Model ids carry the
organization prefix and are passed verbatim::

    complete("openrouter:anthropic/claude-sonnet-4.6", "Hello!")

Browse the catalog at https://openrouter.ai/docs/guides/overview/models.

Optionally, set ``OPENROUTER_SITE_URL`` and/or ``OPENROUTER_APP_TITLE`` to have
your app appear on the OpenRouter leaderboards; both are forwarded as the
ranking headers documented by OpenRouter and are otherwise omitted.

``thinking_effort`` is a no-op here: reasoning behaviour is model- and
provider-specific on OpenRouter; pass any reasoning controls through
``generation_kwargs`` when a model supports them.
"""

import os

from lmdk.providers._chat_completions import ChatCompletionsProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_API_KEY_ENV = "OPENROUTER_API_KEY"
_SITE_URL_ENV = "OPENROUTER_SITE_URL"
_APP_TITLE_ENV = "OPENROUTER_APP_TITLE"


class OpenrouterProvider(ChatCompletionsProvider):
    """Provider for OpenRouter's hosted OpenAI-compatible gateway.

    Reuses :class:`~lmdk.providers._chat_completions.ChatCompletionsProvider`'s
    payload building and response parsing, overriding only the fixed endpoint
    and the ``OPENROUTER_API_KEY`` bearer authentication. Optional ranking
    headers are added when the corresponding environment variables are set.
    """

    required_env = _API_KEY_ENV

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return OpenRouter Bearer-token headers plus optional ranking headers."""
        headers = {"Authorization": f"Bearer {credentials[_API_KEY_ENV]}"}
        if site_url := os.getenv(_SITE_URL_ENV):
            headers["HTTP-Referer"] = site_url
        if app_title := os.getenv(_APP_TITLE_ENV):
            headers["X-Title"] = app_title
        return headers

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Use the model id verbatim; the endpoint is fixed for OpenRouter."""
        return model_id, OPENROUTER_BASE_URL

    @classmethod
    def _build_url(cls, location: str) -> str:
        """Return the fixed OpenRouter chat-completions URL."""
        return f"{location.rstrip('/')}/chat/completions"
