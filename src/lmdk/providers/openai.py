"""Implements the provider to use models hosted in the OpenAI API."""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse
from lmdk.providers._schema import prepare_schema

OPENAI_API_URL = "https://api.openai.com/v1/responses"

_TEMPERATURE_RESTRICTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_TEMPERATURE_RESTRICTED_KWARGS = {"temperature", "top_p"}
_MIN_MAX_OUTPUT_TOKENS = 16


class OpenaiProvider(Provider):
    """Provider for models hosted on the OpenAI API."""

    required_env = "OPENAI_API_KEY"

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return OpenAI Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {credentials['OPENAI_API_KEY']}"}

    @classmethod
    def _build_input(cls, request: CompletionRequest) -> list[dict]:
        """Build the Responses API input list from a CompletionRequest."""
        return [m.to_dict() for m in request.prompt]

    @classmethod
    def _normalize_generation_kwargs(cls, request: CompletionRequest) -> dict:
        """Translate common kwargs to the OpenAI Responses API shape.

        ``lmdk.core.complete`` defaults to ``temperature=0`` for every
        provider, but newer OpenAI reasoning/GPT-5 models reject custom
        sampling controls and only allow their defaults. For those models we
        drop unsupported sampling keys so a normal call succeeds.
        """
        kwargs = dict(request.generation_kwargs or {})

        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            kwargs.setdefault("max_output_tokens", max_tokens)
        if "max_completion_tokens" in kwargs:
            max_completion_tokens = kwargs.pop("max_completion_tokens")
            kwargs.setdefault("max_output_tokens", max_completion_tokens)
        if "max_output_tokens" in kwargs and kwargs["max_output_tokens"] < _MIN_MAX_OUTPUT_TOKENS:
            kwargs["max_output_tokens"] = _MIN_MAX_OUTPUT_TOKENS
        if "stop_sequences" in kwargs and "stop" not in kwargs:
            kwargs["stop"] = kwargs.pop("stop_sequences")

        if request.model_id.startswith(_TEMPERATURE_RESTRICTED_MODEL_PREFIXES):
            for key in _TEMPERATURE_RESTRICTED_KWARGS:
                kwargs.pop(key, None)

        return kwargs

    @classmethod
    def _build_payload(cls, request: CompletionRequest, stream: bool = False) -> dict:
        """Build the full request payload for the OpenAI Responses API."""
        payload: dict = {
            "model": request.model_id,
            "input": cls._build_input(request),
            "stream": stream,
            **cls._normalize_generation_kwargs(request),
        }

        if request.system_instruction:
            payload["instructions"] = request.system_instruction

        if request.thinking_effort != "none":
            payload.setdefault("reasoning", {"effort": request.thinking_effort})

        if request.output_schema and not stream:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema.__name__,
                    "schema": prepare_schema(request.output_schema.model_json_schema()),
                    "strict": True,
                },
            }

        return payload

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return ``reasoning.effort`` from the outbound OpenAI payload."""
        effort = cls._build_payload(request, stream=False).get("reasoning", {}).get("effort")
        return effort if effort is not None else "none"

    @classmethod
    def _extract_text(cls, body: dict) -> str:
        """Extract generated text from an OpenAI Responses API body."""
        if isinstance(body.get("output_text"), str):
            return body["output_text"]

        parts: list[str] = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "".join(parts)

    @classmethod
    def _extract_thinking(cls, body: dict) -> str | None:
        """Extract reasoning text from ``reasoning`` items in the response output."""
        parts: list[str] = []
        for item in body.get("output", []):
            if item.get("type") != "reasoning":
                continue
            for entry in item.get("summary", []):
                if entry.get("type") == "summary_text":
                    parts.append(entry.get("text", ""))
            for entry in item.get("content", []):
                if entry.get("type") == "reasoning_text":
                    parts.append(entry.get("text", ""))
        joined = "".join(parts)
        return joined if joined else None

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        response = cls._make_request(
            OPENAI_API_URL,
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        usage = body.get("usage", {})
        output_details = usage.get("output_tokens_details", {})
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            thinking=cls._extract_thinking(body),
            thinking_tokens=output_details.get("reasoning_tokens", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        response = cls._make_request(
            OPENAI_API_URL,
            json=cls._build_payload(request, stream=True),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            if chunk.get("type") == "response.output_text.delta":
                token = chunk.get("delta", "")
                if token:
                    yield token


# Backwards-/ergonomics-friendly alias for users who prefer the official casing.
OpenAIProvider = OpenaiProvider
