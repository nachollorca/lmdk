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
"""

import os
from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.errors import ProviderError
from lmdk.provider import Provider, RawResponse

# Optional bearer token for servers that require authentication (e.g. vLLM --api-key).
_API_KEY_ENV = "LOCAL_API_KEY"


class LocalProvider(Provider):
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

    @classmethod
    def _build_prompt_payload(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest."""
        api_messages: list[dict] = []
        if request.system_instruction:
            api_messages.append({"role": "system", "content": request.system_instruction})
        api_messages.extend(m.to_dict() for m in request.prompt)
        return api_messages

    @classmethod
    def _build_payload(cls, request: CompletionRequest, model: str, stream: bool = False) -> dict:
        """Build the full request payload for the OpenAI-compatible API."""
        payload: dict = {
            "model": model,
            "messages": cls._build_prompt_payload(request),
            "stream": stream,
            **(request.generation_kwargs or {}),
        }

        if request.output_schema and not stream:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": request.output_schema.model_json_schema(),
                },
            }
        return payload

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        model, location = cls._parse_model_id(request.model_id)
        response = cls._make_request(
            cls._build_url(location),
            json=cls._build_payload(request, model, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        return RawResponse(
            content=body["choices"][0]["message"]["content"],
            input_tokens=body.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=body.get("usage", {}).get("completion_tokens", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        model, location = cls._parse_model_id(request.model_id)
        response = cls._make_request(
            cls._build_url(location),
            json=cls._build_payload(request, model, stream=True),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            choices = chunk.get("choices", [])
            if choices:
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token
