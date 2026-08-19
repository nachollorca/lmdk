"""Generic provider for local, OpenAI-compatible chat-completions endpoints.

Most local model servers (llama.cpp, vLLM, Ollama, LM Studio, …) expose the
same OpenAI ``/v1/chat/completions`` wire protocol; only the endpoint differs.
This single provider targets any of them by reading the endpoint from the
``@location`` suffix of the model identifier:

    ``local:<model>@<host>[:<port>]``

For example::

    complete("local:Qwen3.6-27B-BF16@192.168.10.51:4000", "Hello")

The location is mandatory -- there is no default base URL. When the location
has no scheme, ``http://`` is assumed. Servers that gate access with a token
may set the optional ``LOCAL_API_KEY`` environment variable, which is sent as a
``Bearer`` credential.

Thinking / reasoning is entirely backend-dependent: whether ``thinking_effort``
has any effect, and whether responses expose ``reasoning_content``, thinking
chunks, or token breakdowns, depends on the server, model, and how it was
deployed (e.g. llama.cpp without thinking enabled is a no-op for Section 12).
This provider only parses fields when the server returns them.
"""

import os

from lmdk.errors import ProviderError
from lmdk.providers._chat_completions import ChatCompletionsProvider

# Optional bearer token for servers that require authentication (e.g. vLLM --api-key).
_API_KEY_ENV = "LOCAL_API_KEY"


class LocalProvider(ChatCompletionsProvider):
    """Provider for any local OpenAI-compatible chat-completions server.

    The endpoint is supplied per call via the ``@location`` suffix of the model
    identifier (``local:<model>@<host>[:<port>]``); no environment variables are
    required. An optional ``LOCAL_API_KEY`` is forwarded as a Bearer token when
    set.
    """

    # No environment variables are required; the endpoint comes from @location.
    required_env = ()

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return a Bearer header when ``LOCAL_API_KEY`` is set, else no auth."""
        api_key = os.getenv(_API_KEY_ENV)
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Split ``model_id`` into ``(model, location)``.

        The endpoint is mandatory and provided as an ``@location`` suffix, e.g.
        ``"Qwen3.6-27B-BF16@192.168.10.51:4000"``. Raises ``ProviderError`` when
        the suffix is missing.
        """
        if "@" not in model_id:
            raise ProviderError(
                status_code=0,
                message=(
                    f"{cls.__name__}: model must include an endpoint as "
                    f"'<model>@<host>[:<port>]' (got '{model_id}')."
                ),
                provider=cls.__name__,
            )
        model, location = model_id.rsplit("@", 1)
        return model, location

    @classmethod
    def _build_url(cls, location: str) -> str:
        """Build the chat-completions URL from a ``host[:port]`` location.

        Assumes ``http://`` when the location has no scheme.
        """
        base = location.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        return f"{base}/v1/chat/completions"
