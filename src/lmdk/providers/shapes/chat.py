"""The OpenAI ``/v1/chat/completions`` wire format, addressed via ``@<address>``.

Many servers -- hosted APIs (Mistral) and gateways (Lyceum, OpenRouter, …) as
well as local model servers (llama.cpp, vLLM, Ollama, LM Studio, …) -- expose
the exact same OpenAI ``/v1/chat/completions`` wire protocol: ``messages`` in,
``choices[]`` out. This shape implements that protocol end to end (payload
building, response parsing, streaming). Vendors with a fixed endpoint subclass
it and set ``base_url``; the bare shape targets any server via the model
identifier's ``@`` suffix::

    complete("chat:Qwen3.6-27B-BF16@192.168.10.51:4000", "Hello")

The address is mandatory for the bare shape and ``http://`` is assumed when it
has no scheme. Servers that gate access with a token may set the optional
``LOCAL_API_KEY`` environment variable, which is sent as a ``Bearer``
credential.

Thinking / reasoning is entirely backend-dependent: whether ``thinking_effort``
has any effect, and whether responses expose ``reasoning_content``, thinking
chunks, or token breakdowns, depends on the server, model, and how it was
deployed. This shape only parses fields when they are present.
"""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.errors import ProviderError, RequestTooLargeError
from lmdk.provider import RawResponse
from lmdk.providers._schema import prepare_schema
from lmdk.providers.shapes._base import ShapeProvider


class ChatProvider(ShapeProvider):
    """Provider for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Built on the OpenAI framework: request payloads and response parsing follow
    OpenAI's Chat Completions API (``messages`` in, ``choices[]`` out), so any
    server or gateway implementing that wire protocol works unchanged.

    Subclasses set ``base_url`` (everything up to but excluding
    ``/v1/chat/completions``) to make the ``@<address>`` suffix optional.
    """

    path = "/v1/chat/completions"

    @classmethod
    def _classify_error(
        cls, status_code: int, data: dict | None, message: str | None
    ) -> tuple[type[ProviderError], int]:
        # OpenAI-compatible APIs return 400 instead of 413 for context overflow.
        # This mapping is arbitrary, but I want callers to distinguish
        # context overflow from genuine bad request syntax
        if status_code == 400 and isinstance(data, dict):
            err = data.get("error")
            err_dict = err if isinstance(err, dict) else {}
            code = err_dict.get("code")
            err_type = err_dict.get("type")
            msg = (message or "").lower()
            if (
                code == "context_length_exceeded"
                or err_type == "exceed_context_size_error"
                or "context length" in msg
                or "context size" in msg
            ):
                return RequestTooLargeError, 413
        return super()._classify_error(status_code, data, message)

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
        """Build the full request payload for the OpenAI-compatible API.

        ``thinking_effort`` is not mapped to any wire field here; backends vary
        widely and most ignore it unless the caller passes server-specific
        kwargs via ``generation_kwargs``. Subclasses that do support a reasoning
        control (e.g. Mistral) override this and patch the payload.

        Structured output uses the strict JSON-Schema subset (see
        ``_schema.prepare_schema``) that hosted OpenAI-compatible APIs require;
        servers that do not implement ``strict`` simply ignore the flag.
        """
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
                    "schema": prepare_schema(request.output_schema.model_json_schema()),
                    "strict": True,
                },
            }
        return payload

    @staticmethod
    def _extract_text(content: str | list | None) -> str:
        """Extract answer text from message content (plain string or chunk list)."""
        if isinstance(content, list):
            return "".join(
                chunk.get("text", "") for chunk in content if chunk.get("type") == "text"
            )
        return content or ""

    @staticmethod
    def _thinking_text_from_chunk(chunk: dict) -> str:
        """Extract thinking text from a single ``type: thinking`` chunk."""
        nested = chunk.get("thinking")
        if isinstance(nested, list):
            return "".join(item.get("text", "") for item in nested if isinstance(item, dict))
        if isinstance(nested, str):
            return nested
        return chunk.get("text", "")

    @staticmethod
    def _extract_thinking_from_chunks(content: list) -> str | None:
        """Join thinking text from a list-shaped message ``content``."""
        parts = [
            ChatProvider._thinking_text_from_chunk(chunk)
            for chunk in content
            if chunk.get("type") == "thinking"
        ]
        joined = "".join(parts)
        return joined if joined else None

    @staticmethod
    def _extract_thinking(message: dict) -> str | None:
        """Extract thinking/reasoning text when the server exposes it.

        Backends that do not return reasoning fields always yield ``None``
        regardless of ``thinking_effort``.
        """
        for key in ("reasoning_content", "reasoning"):
            reasoning = message.get(key)
            if isinstance(reasoning, str) and reasoning:
                return reasoning

        content = message.get("content")
        if isinstance(content, list):
            return ChatProvider._extract_thinking_from_chunks(content)

        return None

    @staticmethod
    def _extract_thinking_tokens(usage: dict) -> int:
        """Extract reasoning/thinking token count from usage metadata when present.

        Looked up in ``completion_tokens_details`` first (OpenAI's shape), then
        at the top level of ``usage`` (used by some servers, e.g. Mistral).
        """
        details = usage.get("completion_tokens_details") or {}
        for source in (details, usage):
            for key in ("reasoning_tokens", "thinking_tokens"):
                value = source.get(key)
                if isinstance(value, int):
                    return value
        return 0

    @classmethod
    def _send_completion_request(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> RawResponse:
        model, location = cls._parse_model_id(request.model_id)
        response = cls._make_request(
            cls._build_url(location),
            json=cls._build_payload(request, model, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        usage = body.get("usage", {})
        return RawResponse(
            content=cls._extract_text(message.get("content")),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            thinking=cls._extract_thinking(message),
            thinking_tokens=cls._extract_thinking_tokens(usage),
            finish_reason=choice.get("finish_reason"),
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
                token = cls._extract_text(choices[0].get("delta", {}).get("content", ""))
                if token:
                    yield token
