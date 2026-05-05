"""Implements the provider to use models hosted in the OpenAI API."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse

OPENAI_API_URL = "https://api.openai.com/v1/responses"

_SCHEMA_COMBINATORS = ("anyOf", "oneOf", "allOf")
_TEMPERATURE_RESTRICTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_TEMPERATURE_RESTRICTED_KWARGS = {"temperature", "top_p"}
_MIN_MAX_OUTPUT_TOKENS = 16


def _prepare_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Pydantic JSON schema for OpenAI structured outputs.

    OpenAI's strict JSON-schema mode requires every object node to declare
    ``additionalProperties: false`` and all object properties to be listed in
    ``required``. Pydantic also emits ``default`` entries for fields with
    defaults, which are not part of OpenAI's supported strict-schema subset.

    The input schema is deep-copied and never mutated.
    """
    prepared = deepcopy(schema)
    _prepare_schema_in_place(prepared)
    return prepared


def _prepare_schema_in_place(node: Any) -> None:
    """Recursively normalize a JSON Schema node in place."""
    if not isinstance(node, dict):
        return

    node.pop("default", None)

    _prepare_object_schema(node)
    _prepare_recursive_schemas(node)


def _prepare_object_schema(node: dict[str, Any]) -> None:
    """Handle object-specific JSON Schema normalization."""
    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
        node["additionalProperties"] = False
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            for prop in properties.values():
                _prepare_schema_in_place(prop)


def _prepare_recursive_schemas(node: dict[str, Any]) -> None:
    """Handle recursive JSON Schema structures."""
    if "$defs" in node:
        for definition in node["$defs"].values():
            _prepare_schema_in_place(definition)

    if "items" in node:
        _prepare_schema_in_place(node["items"])

    additional_properties = node.get("additionalProperties")
    if isinstance(additional_properties, dict):
        _prepare_schema_in_place(additional_properties)

    for key in _SCHEMA_COMBINATORS:
        for option in node.get(key, []) or []:
            _prepare_schema_in_place(option)


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

        if request.output_schema and not stream:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema.__name__,
                    "schema": _prepare_schema(request.output_schema.model_json_schema()),
                    "strict": True,
                },
            }

        return payload

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
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        response = cls._make_request(
            OPENAI_API_URL,
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        usage = body.get("usage", {})
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
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
