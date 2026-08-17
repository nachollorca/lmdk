"""Provider for Lyceum's serverless, OpenAI-compatible chat-completions API.

Lyceum exposes the same OpenAI ``/chat/completions`` wire protocol as the
``local`` provider, but at a fixed hosted base URL and behind a
``LYCEUM_API_KEY`` bearer token::

    complete("lyceum:moonshotai/kimi-k3", "Hello!")
"""

from lmdk.providers.local import LocalProvider

LYCEUM_BASE_URL = "https://api.lyceum.technology/api/v2/external/serverless"

_API_KEY_ENV = "LYCEUM_API_KEY"


class LyceumProvider(LocalProvider):
    """Provider for Lyceum's serverless OpenAI-compatible endpoint.

    Reuses :class:`~lmdk.providers.local.LocalProvider`'s payload building and
    response parsing, overriding only the endpoint and authentication: the base
    URL is fixed and the ``LYCEUM_API_KEY`` environment variable is required.

    ``thinking_effort`` is a no-op: Lyceum toggles reasoning through per-model
    ``chat_template_kwargs`` (e.g. ``{"thinking": false}``) rather than a
    top-level control, so pass those via ``generation_kwargs`` when needed.
    """

    required_env = _API_KEY_ENV

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Lyceum Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {credentials[_API_KEY_ENV]}"}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Use the model id verbatim; the endpoint is fixed for Lyceum."""
        return model_id, LYCEUM_BASE_URL

    @classmethod
    def _build_url(cls, location: str) -> str:
        """Return the fixed Lyceum chat-completions URL."""
        return f"{location.rstrip('/')}/chat/completions"
