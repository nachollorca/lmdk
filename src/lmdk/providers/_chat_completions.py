"""Shared base for providers speaking the OpenAI ``/chat/completions`` protocol.

Many providers -- hosted APIs (Mistral) and gateways (Lyceum, OpenRouter, …) as
well as local model servers (llama.cpp, vLLM, Ollama, LM Studio, …) -- expose
the exact same OpenAI ``/chat/completions`` wire protocol: ``messages`` in,
``choices[]`` out. This base class implements that protocol end to end (payload
building, response parsing, streaming). A subclass with a fixed endpoint only
needs ``base_url`` and ``_build_auth_headers``; providers whose endpoint varies
per call override ``_parse_model_id`` / ``_build_url``.

Thinking / reasoning is entirely backend-dependent: whether ``thinking_effort``
has any effect, and whether responses expose ``reasoning_content``, thinking
chunks, or token breakdowns, depends on the server, model, and how it was
deployed. This base class only parses fields when they are present.
"""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse
from lmdk.providers._schema import prepare_schema


class ChatCompletionsProvider(Provider):
    """Base provider for any OpenAI-compatible ``/chat/completions`` endpoint.

    Built on the OpenAI framework: request payloads and response parsing follow
    OpenAI's Chat Completions API (``messages`` in, ``choices[]`` out), so any
    server or gateway implementing that wire protocol works unchanged.

    Subclasses set ``base_url`` (everything up to but excluding
    ``/chat/completions``) and implement ``_build_auth_headers``. Providers whose
    endpoint is not fixed (e.g. ``local``) override ``_parse_model_id`` and
    ``_build_url`` instead.
    """

    # Fixed endpoint, up to but excluding ``/chat/completions``.
    base_url: str = ""

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Split ``model_id`` into ``(model, location)``.

        The default treats the whole id as the model and the fixed ``base_url``
        as the location, which ``_build_url`` turns into an endpoint.
        """
        return model_id, cls.base_url

    @classmethod
    def _build_url(cls, location: str) -> str:
        """Build the chat-completions URL from a ``location``."""
        return f"{location.rstrip('/')}/chat/completions"

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
            ChatCompletionsProvider._thinking_text_from_chunk(chunk)
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
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            return reasoning

        content = message.get("content")
        if isinstance(content, list):
            return ChatCompletionsProvider._extract_thinking_from_chunks(content)

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
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        model, location = cls._parse_model_id(request.model_id)
        response = cls._make_request(
            cls._build_url(location),
            json=cls._build_payload(request, model, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        message = body["choices"][0]["message"]
        usage = body.get("usage", {})
        return RawResponse(
            content=cls._extract_text(message.get("content")),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            thinking=cls._extract_thinking(message),
            thinking_tokens=cls._extract_thinking_tokens(usage),
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
