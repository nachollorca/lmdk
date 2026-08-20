"""Provider for Lyceum's serverless, OpenAI-compatible chat-completions API.

Lyceum exposes the same OpenAI ``/chat/completions`` wire protocol as the
``local`` provider, but at a fixed hosted base URL and behind a
``LYCEUM_API_KEY`` bearer token::

    complete("lyceum:moonshotai/kimi-k3", "Hello!")
"""

from lmdk.providers._chat_completions import ChatCompletionsProvider

LYCEUM_BASE_URL = "https://api.lyceum.technology/api/v2/external/serverless"

_API_KEY_ENV = "LYCEUM_API_KEY"


class LyceumProvider(ChatCompletionsProvider):
    """Provider for Lyceum's serverless OpenAI-compatible endpoint.

    Reuses :class:`~lmdk.providers._chat_completions.ChatCompletionsProvider`'s
    payload building and response parsing, setting only the fixed ``base_url``
    and the ``LYCEUM_API_KEY`` bearer authentication.

    ``thinking_effort`` is a no-op: Lyceum toggles reasoning through per-model
    ``chat_template_kwargs`` (e.g. ``{"thinking": False}``) rather than a
    top-level control, so pass those via ``generation_kwargs`` when needed.
    """

    required_env = _API_KEY_ENV
    base_url = LYCEUM_BASE_URL

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Lyceum Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {credentials[_API_KEY_ENV]}"}
