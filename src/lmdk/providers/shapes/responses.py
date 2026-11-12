"""The OpenAI Responses API wire format (``/v1/responses``), addressed via ``@<address>``.

This shape speaks OpenAI's Responses API (``input`` in, ``output[]`` out) and
is addressed explicitly through the model identifier's ``@`` suffix::

    complete("responses:gpt-5.5@192.168.10.51:8000", "Hello")

The address is mandatory for the bare shape and ``http://`` is assumed when it
has no scheme. Servers that gate access with a token may set the optional
``LOCAL_API_KEY`` environment variable, which is sent as a ``Bearer``
credential. The ``openai`` vendor subclasses this shape with the hosted default
location and its own API key.
"""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import RawResponse
from lmdk.providers._schema import prepare_schema
from lmdk.providers.shapes._base import ShapeProvider


class ResponsesProvider(ShapeProvider):
    """Provider for any endpoint speaking the OpenAI Responses API (``/v1/responses``)."""

    path = "/v1/responses"

    @classmethod
    def _build_input(cls, request: CompletionRequest) -> list[dict]:
        """Build the Responses API input list from a CompletionRequest."""
        return [m.to_dict() for m in request.prompt]

    @classmethod
    def _normalize_generation_kwargs(cls, request: CompletionRequest) -> dict:
        """Translate common kwargs to the OpenAI Responses API shape.

        ``max_tokens`` / ``max_completion_tokens`` become ``max_output_tokens``
        and ``stop_sequences`` becomes ``stop``. Vendors override this to add
        model-specific restrictions.
        """
        kwargs = dict(request.generation_kwargs or {})

        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            kwargs.setdefault("max_output_tokens", max_tokens)
        if "max_completion_tokens" in kwargs:
            max_completion_tokens = kwargs.pop("max_completion_tokens")
            kwargs.setdefault("max_output_tokens", max_completion_tokens)
        if "stop_sequences" in kwargs and "stop" not in kwargs:
            kwargs["stop"] = kwargs.pop("stop_sequences")

        return kwargs

    @classmethod
    def _build_payload(
        cls, request: CompletionRequest, model: str | None = None, stream: bool = False
    ) -> dict:
        """Build the full request payload for the OpenAI Responses API."""
        if model is None:
            model = cls._parse_model_id(request.model_id)[0]
        payload: dict = {
            "model": model,
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
        """Return ``reasoning.effort`` from the outbound Responses API payload."""
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
        usage = body.get("usage", {})
        output_details = usage.get("output_tokens_details", {})
        incomplete_reason = (body.get("incomplete_details") or {}).get("reason")
        finish_reason = incomplete_reason or body.get("status")
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            thinking=cls._extract_thinking(body),
            thinking_tokens=output_details.get("reasoning_tokens", 0),
            finish_reason=finish_reason,
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
            if chunk.get("type") == "response.output_text.delta":
                token = chunk.get("delta", "")
                if token:
                    yield token
