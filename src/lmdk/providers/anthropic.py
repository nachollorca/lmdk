"""Implements the provider to use models hosted in Anthropic API."""

from collections.abc import Iterator
from typing import Any

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


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


class AnthropicProvider(Provider):
    """Provider for models hosted on the Anthropic API."""

    required_env = "ANTHROPIC_API_KEY"

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Anthropic API-key authentication headers."""
        return {
            "x-api-key": credentials["ANTHROPIC_API_KEY"],
            "anthropic-version": ANTHROPIC_VERSION,
        }

    @classmethod
    def _build_messages(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest.

        Anthropic uses a top-level ``system`` parameter instead of a system
        message role, so only user/assistant messages are included here.
        """
        return [m.to_dict() for m in request.prompt]

    @classmethod
    def _build_payload(cls, request: CompletionRequest, stream: bool = False) -> dict:
        """Build the full request payload for the Anthropic API.

        ``max_tokens`` is required by the Anthropic API. If not provided in
        ``generation_kwargs``, a default of 4096 is used.
        """
        generation_kwargs = dict(request.generation_kwargs or {})
        max_tokens = generation_kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)

        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_messages(request),
            "max_tokens": max_tokens,
            **generation_kwargs,
        }

        if request.system_instruction:
            payload["system"] = request.system_instruction

        if request.output_schema and not stream:
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _prepare_schema(request.output_schema.model_json_schema()),
                },
            }

        if stream:
            payload["stream"] = True

        return payload

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
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        response = cls._make_request(
            ANTHROPIC_API_URL,
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=body["usage"]["input_tokens"],
            output_tokens=body["usage"]["output_tokens"],
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        response = cls._make_request(
            ANTHROPIC_API_URL,
            json=cls._build_payload(request, stream=True),
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
