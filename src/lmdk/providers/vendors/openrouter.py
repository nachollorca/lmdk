"""Vendor for OpenRouter's unified, OpenAI-compatible chat-completions API.

Subclasses the ``chat`` shape with the hosted default location and the
``OPENROUTER_API_KEY`` credential; an ``@<address>`` suffix overrides the
default location (then only the optional ``LOCAL_API_KEY`` is sent as a
``Bearer`` token, never the vendor key). Model ids carry the organization
prefix and are passed verbatim::

    complete("openrouter:anthropic/claude-sonnet-4.6", "Hello!")

Browse the catalog at https://openrouter.ai/docs/guides/overview/models.

Optionally, set ``OPENROUTER_SITE_URL`` and/or ``OPENROUTER_APP_TITLE`` to have
your app appear on the OpenRouter leaderboards; both are forwarded as the
ranking headers documented by OpenRouter and are otherwise omitted. They are
only sent to OpenRouter's own hosted endpoint.

``thinking_effort`` is a no-op here: reasoning behaviour is model- and
provider-specific on OpenRouter; pass any reasoning controls through
``generation_kwargs`` when a model supports them.
"""

import os

from lmdk.providers.shapes.chat import ChatProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api"

_API_KEY_ENV = "OPENROUTER_API_KEY"
_SITE_URL_ENV = "OPENROUTER_SITE_URL"
_APP_TITLE_ENV = "OPENROUTER_APP_TITLE"


class OpenrouterProvider(ChatProvider):
    """Provider for OpenRouter's hosted OpenAI-compatible gateway."""

    required_env = _API_KEY_ENV
    base_url = OPENROUTER_BASE_URL

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return the auth headers, plus optional ranking headers on the hosted endpoint."""
        headers = super()._build_auth_headers(credentials)
        if credentials.get(_API_KEY_ENV):
            if site_url := os.getenv(_SITE_URL_ENV):
                headers["HTTP-Referer"] = site_url
            if app_title := os.getenv(_APP_TITLE_ENV):
                headers["X-Title"] = app_title
        return headers
