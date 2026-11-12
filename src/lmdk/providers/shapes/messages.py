"""The Anthropic Messages API wire format (``/v1/messages``), addressed via ``@<address>``.

This shape speaks Anthropic's Messages API and is addressed explicitly through
the model identifier's ``@`` suffix::

    complete("messages:claude-sonnet-4@192.168.10.51:8000", "Hello")

The address is mandatory for the bare shape and ``http://`` is assumed when it
has no scheme. Servers that gate access with a token may set the optional
``LOCAL_API_KEY`` environment variable, which is sent in the ``x-api-key``
header (Anthropic's auth header format). The ``anthropic`` vendor subclasses
this shape with the hosted default location and its own API key; ``bedrock``
reuses it with an AWS-specific envelope.
"""

from collections.abc import Iterator
from typing import Any

from lmdk.datatypes import CompletionRequest
from lmdk.provider import RawResponse
from lmdk.providers.shapes._base import ShapeProvider

ANTHROPIC_VERSION = "2023-06-01"

# Anthropic requires ``max_tokens``, so a default is unavoidable. Current Claude
# models cap output at 64k; legacy ones (3.x) cap at 4k-8k and reject anything
# larger, so pass ``max_tokens`` explicitly when calling those.
DEFAULT_MAX_TOKENS = 32000

# Sampling controls Anthropic rejects when ``thinking`` is enabled.
_THINKING_INCOMPATIBLE_KWARGS = ("temperature", "top_p", "top_k")


def _prepare_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Pydantic JSON schema for the Anthropic API.

    Anthropic requires ``additionalProperties: false`` on every object node.
    This function recursively walks the schema and adds the constraint
    wherever ``"type": "object"`` appears, including inside ``$defs``.
    """
    schema = dict(schema)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            _prepare_schema_in_place(prop)

    if "$defs" in schema:
        schema["$defs"] = {
            name: _prepare_schema(definition) for name, definition in schema["$defs"].items()
        }

    return schema


def _prepare_schema_in_place(node: dict[str, Any]) -> None:
    """Recursively add ``additionalProperties: false`` to nested object nodes."""
    if node.get("type") == "object":
        node.setdefault("additionalProperties", False)
        for prop in node.get("properties", {}).values():
            _prepare_schema_in_place(prop)
    elif node.get("type") == "array" and "items" in node:
        _prepare_schema_in_place(node["items"])


class MessagesProvider(ShapeProvider):
    """Provider for any endpoint speaking Anthropic's Messages API (``/v1/messages``)."""

    path = "/v1/messages"

    @classmethod
    def _key_headers(cls, api_key: str) -> dict:
        """Anthropic's auth header format is ``x-api-key``, not ``Bearer``."""
        return {"x-api-key": api_key}

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Key header plus the ``anthropic-version`` header the wire format requires."""
        headers = dict(super()._build_auth_headers(credentials))
        headers["anthropic-version"] = ANTHROPIC_VERSION
        return headers

    @classmethod
    def _build_messages(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest.

        Anthropic uses a top-level ``system`` parameter instead of a system
        message role, so only user/assistant messages are included here.
        """
        return [m.to_dict() for m in request.prompt]

    @classmethod
    def _normalize_generation_kwargs(cls, request: CompletionRequest) -> dict:
        """Pass generation kwargs through; vendors may drop unsupported controls."""
        return dict(request.generation_kwargs or {})

    @classmethod
    def _build_payload(
        cls, request: CompletionRequest, model: str | None = None, stream: bool = False
    ) -> dict:
        """Build the full request payload for the Anthropic Messages API.

        ``max_tokens`` is required by the API. If not provided in
        ``generation_kwargs``, ``DEFAULT_MAX_TOKENS`` is used.
        """
        if model is None:
            model = cls._parse_model_id(request.model_id)[0]
        generation_kwargs = cls._normalize_generation_kwargs(request)
        max_tokens = generation_kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)

        thinking_block: dict | None = None
        effort: str | None = None
        if request.thinking_effort != "none" and "thinking" not in generation_kwargs:
            # Anthropic rejects custom temperature/top_p/top_k when thinking
            # is enabled. Drop them so the request goes through cleanly.
            for key in _THINKING_INCOMPATIBLE_KWARGS:
                generation_kwargs.pop(key, None)
            # Adaptive thinking lets Claude decide when and how much to think;
            # ``effort`` (low/medium/high) is soft guidance and ``max_tokens``
            # remains the hard cap on thinking + response tokens.
            thinking_block = {"type": "adaptive"}
            effort = request.thinking_effort

        payload: dict = {
            "model": model,
            "messages": cls._build_messages(request),
            "max_tokens": max_tokens,
            **generation_kwargs,
        }

        if thinking_block is not None:
            payload["thinking"] = thinking_block

        if request.system_instruction:
            payload["system"] = request.system_instruction

        output_config: dict = {}
        if effort is not None:
            output_config["effort"] = effort
        if request.output_schema and not stream:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _prepare_schema(request.output_schema.model_json_schema()),
            }
        if output_config:
            payload["output_config"] = output_config

        if stream:
            payload["stream"] = True

        return payload

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return ``output_config.effort`` from the outbound Anthropic payload."""
        effort = cls._build_payload(request, stream=False).get("output_config", {}).get("effort")
        return effort if effort is not None else "none"

    @classmethod
    def _extract_text(cls, body: dict) -> str:
        """Extract text content from the Anthropic response body.

        The response ``content`` field is an array of typed blocks.
        Only ``text`` blocks are extracted and joined.
        """
        parts = []
        for block in body.get("content", []):
            if block.get("type") == "text":
                parts.append(block["text"])
        return "".join(parts)

    @classmethod
    def _extract_thinking(cls, body: dict) -> str | None:
        """Extract thinking content from ``thinking`` blocks in the response."""
        parts = []
        for block in body.get("content", []):
            if block.get("type") == "thinking":
                parts.append(block["thinking"])
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
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            thinking=cls._extract_thinking(body),
            thinking_tokens=usage.get("output_tokens_details", {}).get("thinking_tokens", 0),
            finish_reason=body.get("stop_reason"),
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
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield text
