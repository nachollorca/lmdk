"""Tests for lmdk.providers.openai — OpenaiProvider."""

import json
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, UserMessage
from lmdk.provider import RawResponse
from lmdk.providers.openai import OPENAI_API_URL, OpenAIProvider, OpenaiProvider, _prepare_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "gpt-5.5",
        "prompt": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_chat_response(content: str = "hello", input_tokens: int = 10, output_tokens: int = 5):
    """Build a mock requests.Response that mimics an OpenAI Responses API response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics an OpenAI Responses API SSE stream."""
    lines = []
    for token in tokens:
        chunk = {"type": "response.output_text.delta", "delta": token}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append('data: {"type": "response.completed", "response": {}}')

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


# ---------------------------------------------------------------------------
# Pydantic models for structured output tests
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""


class Recipe(BaseModel):
    ingredients: list[Ingredient]


# ---------------------------------------------------------------------------
# _build_auth_headers
# ---------------------------------------------------------------------------


class TestBuildAuthHeaders:
    def test_returns_bearer_token(self):
        headers = OpenaiProvider._build_auth_headers({"OPENAI_API_KEY": "sk-test"})
        assert headers["Authorization"] == "Bearer sk-test"

    def test_official_casing_alias(self):
        assert OpenAIProvider is OpenaiProvider


# ---------------------------------------------------------------------------
# _build_input
# ---------------------------------------------------------------------------


class TestBuildInput:
    def test_builds_message_input(self):
        payload = OpenaiProvider._build_input(_make_request())
        assert len(payload) == 1
        assert payload[0] == {"role": "user", "content": "hi"}

    def test_system_instruction_not_in_input(self):
        request = _make_request(system_instruction="Be a pirate.")
        payload = OpenaiProvider._build_input(request)
        assert len(payload) == 1
        assert all(m["role"] != "system" for m in payload)


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_basic_payload(self):
        payload = OpenaiProvider._build_payload(_make_request())
        assert payload["model"] == "gpt-5.5"
        assert payload["input"] == [{"role": "user", "content": "hi"}]
        assert payload["stream"] is False
        assert "text" not in payload

    def test_system_instruction_in_payload(self):
        payload = OpenaiProvider._build_payload(_make_request(system_instruction="Be a pirate."))
        assert payload["instructions"] == "Be a pirate."

    def test_stream_flag(self):
        payload = OpenaiProvider._build_payload(_make_request(), stream=True)
        assert payload["stream"] is True

    def test_generation_kwargs_forwarded_and_mapped(self):
        request = _make_request(
            model_id="gpt-4.1-mini",
            generation_kwargs={"temperature": 0.9, "max_tokens": 10, "stop_sequences": ["END"]},
        )
        payload = OpenaiProvider._build_payload(request)
        assert payload["temperature"] == 0.9
        assert payload["max_output_tokens"] == 16
        assert "max_tokens" not in payload
        assert payload["stop"] == ["END"]
        assert "stop_sequences" not in payload

    def test_drops_unsupported_sampling_kwargs_for_gpt5(self):
        request = _make_request(generation_kwargs={"temperature": 0, "top_p": 0.9})
        payload = OpenaiProvider._build_payload(request)
        assert "temperature" not in payload
        assert "top_p" not in payload

    def test_maps_max_completion_tokens(self):
        request = _make_request(generation_kwargs={"max_completion_tokens": 20})
        payload = OpenaiProvider._build_payload(request)
        assert payload["max_output_tokens"] == 20
        assert "max_completion_tokens" not in payload

    def test_clamps_max_output_tokens_to_responses_api_minimum(self):
        request = _make_request(generation_kwargs={"max_output_tokens": 10})
        payload = OpenaiProvider._build_payload(request)
        assert payload["max_output_tokens"] == 16

    def test_keeps_explicit_max_output_tokens(self):
        request = _make_request(generation_kwargs={"max_tokens": 10, "max_output_tokens": 20})
        payload = OpenaiProvider._build_payload(request)
        assert "max_tokens" not in payload
        assert payload["max_output_tokens"] == 20

    def test_structured_output_payload(self):
        request = _make_request(output_schema=Person)
        payload = OpenaiProvider._build_payload(request)
        text_format = payload["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "Person"
        assert text_format["strict"] is True
        assert "schema" in text_format

    def test_no_structured_output_while_streaming(self):
        request = _make_request(output_schema=Person)
        payload = OpenaiProvider._build_payload(request, stream=True)
        assert "text" not in payload

    def test_no_reasoning_block_when_thinking_effort_none(self):
        payload = OpenaiProvider._build_payload(_make_request())
        assert "reasoning" not in payload

    def test_thinking_effort_maps_to_reasoning_effort(self):
        request = _make_request(thinking_effort="medium")
        payload = OpenaiProvider._build_payload(request)
        assert payload["reasoning"] == {"effort": "medium"}

    def test_thinking_effort_with_structured_output(self):
        request = _make_request(thinking_effort="high", output_schema=Person)
        payload = OpenaiProvider._build_payload(request)
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["text"]["format"]["type"] == "json_schema"

    def test_explicit_reasoning_in_generation_kwargs_overrides_thinking_effort(self):
        request = _make_request(
            thinking_effort="high",
            generation_kwargs={"reasoning": {"effort": "low"}},
        )
        payload = OpenaiProvider._build_payload(request)
        assert payload["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# _prepare_schema
# ---------------------------------------------------------------------------


class TestPrepareSchema:
    def test_adds_openai_strict_object_constraints(self):
        result = _prepare_schema(Person.model_json_schema())
        assert result["additionalProperties"] is False
        assert result["required"] == ["name", "age"]

    def test_handles_nested_defs_and_defaults(self):
        result = _prepare_schema(Recipe.model_json_schema())
        ingredient = result["$defs"]["Ingredient"]
        assert result["additionalProperties"] is False
        assert ingredient["additionalProperties"] is False
        assert ingredient["required"] == ["name", "quantity", "unit"]
        assert "default" not in ingredient["properties"]["unit"]

    def test_does_not_mutate_original_schema(self):
        schema = Recipe.model_json_schema()
        _prepare_schema(schema)
        assert "additionalProperties" not in schema
        assert "default" in schema["$defs"]["Ingredient"]["properties"]["unit"]

    def test_handles_arrays_and_combinators(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "anyOf": [{"type": "object", "properties": {"name": {"type": "string"}}}]
                    },
                }
            },
        }
        result = _prepare_schema(schema)
        nested = result["properties"]["items"]["items"]["anyOf"][0]
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["name"]


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_uses_output_text_when_present(self):
        assert OpenaiProvider._extract_text({"output_text": "Hello"}) == "Hello"

    def test_extracts_output_content_parts(self):
        body = {
            "output": [
                {"content": [{"type": "output_text", "text": "Hello "}]},
                {"content": [{"type": "output_text", "text": "world"}]},
            ]
        }
        assert OpenaiProvider._extract_text(body) == "Hello world"

    def test_missing_output_returns_empty_string(self):
        assert OpenaiProvider._extract_text({}) == ""


# ---------------------------------------------------------------------------
# _send_request — basic text completion
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = OpenaiProvider._send_request(
                _make_request(), credentials={"OPENAI_API_KEY": "sk-test"}
            )

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert mock_post.call_args[0][0] == OPENAI_API_URL
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_missing_usage_defaults_to_zero(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": []}
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            result = OpenaiProvider._send_request(
                _make_request(), credentials={"OPENAI_API_KEY": "sk-test"}
            )
        assert result.content == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_structured_output_payload(self):
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Person)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = OpenaiProvider._send_request(
                request, credentials={"OPENAI_API_KEY": "sk-test"}
            )

        assert result.content == content
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["text"]["format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                OpenaiProvider._stream_response(
                    _make_request(), credentials={"OPENAI_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(
                OpenaiProvider._stream_response(
                    _make_request(), credentials={"OPENAI_API_KEY": "sk-test"}
                )
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True

    def test_skips_empty_or_non_text_events(self):
        lines = [
            f"data: {json.dumps({'type': 'response.created', 'response': {}})}",
            f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': ''})}",
            f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': 'ok'})}",
            f"data: {json.dumps({'type': 'response.completed', 'response': {}})}",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                OpenaiProvider._stream_response(
                    _make_request(), credentials={"OPENAI_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["ok"]
