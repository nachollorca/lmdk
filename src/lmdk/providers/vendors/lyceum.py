"""Vendor for Lyceum's serverless, OpenAI-compatible chat-completions API.

Subclasses the ``chat`` shape with the hosted default location and the
``LYCEUM_API_KEY`` credential; an ``@<address>`` suffix overrides the default
location (then only the optional ``LOCAL_API_KEY`` is sent as a ``Bearer``
token, never the vendor key)::

    complete("lyceum:moonshotai/kimi-k3", "Hello!")

Quirk: Lyceum's hosted base path already carries its version prefix
(``/api/v2/external/serverless``), so its endpoint path is ``/chat/completions``
instead of the shape's ``/v1/chat/completions``.

``thinking_effort`` is a no-op: Lyceum toggles reasoning through per-model
``chat_template_kwargs`` (e.g. ``{"thinking": False}``) rather than a
top-level control, so pass those via ``generation_kwargs`` when needed.
"""

from lmdk.providers.shapes.chat import ChatProvider

LYCEUM_BASE_URL = "https://api.lyceum.technology/api/v2/external/serverless"

_API_KEY_ENV = "LYCEUM_API_KEY"


class LyceumProvider(ChatProvider):
    """Provider for Lyceum's serverless OpenAI-compatible endpoint."""

    required_env = _API_KEY_ENV
    base_url = LYCEUM_BASE_URL
    path = "/chat/completions"
