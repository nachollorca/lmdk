"""Tests for lmdk.providers.anthropic — AnthropicProvider."""

import json
from unittest.mock import MagicMock, patch

from lmdk.datatypes import CompletionRequest, UserMessage
from lmdk.provider import RawResponse
from lmdk.providers.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    DEFAULT_MAX_TOKENS,
    AnthropicProvider,
    _prepare_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "claude-sonnet-4-20250514",
        "prompt": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_chat_response(content: str = "hello", input_tokens: int = 10, output_tokens: int = 5):
    """Build a mock requests.Response that mimics an Anthropic message response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics an Anthropic SSE stream."""
    lines = [
        'data: {"type": "message_start", "message": {"id": "msg_test", "type": "message", "role": "assistant", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}}}',
        'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
    ]
    for token in tokens:
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": token},
        }
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append('data: {"type": "content_block_stop", "index": 0}')
    lines.append(
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}'
    )
    lines.append('data: {"type": "message_stop"}')

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


# ---------------------------------------------------------------------------
# Pydantic models for structured output tests
# ---------------------------------------------------------------------------

from pydantic import BaseModel


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
    def test_returns_api_key_and_version(self):
        headers = AnthropicProvider._build_auth_headers({"ANTHROPIC_API_KEY": "sk-test"})
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == ANTHROPIC_VERSION

    def test_no_bearer_prefix(self):
        """Anthropic uses x-api-key, not Bearer auth."""
        headers = AnthropicProvider._build_auth_headers({"ANTHROPIC_API_KEY": "sk-test"})
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_without_system_instruction(self):
        request = _make_request()
        messages = AnthropicProvider._build_messages(request)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "hi"}

    def test_system_instruction_not_in_messages(self):
        """System instruction is handled at the payload level, not in messages."""
        request = _make_request(system_instruction="Be a pirate.")
        messages = AnthropicProvider._build_messages(request)
        assert len(messages) == 1
        assert all(m["role"] != "system" for m in messages)


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_basic_payload(self):
        request = _make_request()
        payload = AnthropicProvider._build_payload(request)
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
        assert "stream" not in payload
        assert "system" not in payload

    def test_system_instruction_in_payload(self):
        request = _make_request(system_instruction="Be a pirate.")
        payload = AnthropicProvider._build_payload(request)
        assert payload["system"] == "Be a pirate."

    def test_stream_flag_in_payload(self):
        request = _make_request()
        payload = AnthropicProvider._build_payload(request, stream=True)
        assert payload["stream"] is True

    def test_no_stream_key_when_not_streaming(self):
        request = _make_request()
        payload = AnthropicProvider._build_payload(request, stream=False)
        assert "stream" not in payload

    def test_max_tokens_from_generation_kwargs(self):
        request = _make_request(generation_kwargs={"max_tokens": 100})
        payload = AnthropicProvider._build_payload(request)
        assert payload["max_tokens"] == 100

    def test_default_max_tokens(self):
        request = _make_request(generation_kwargs={})
        payload = AnthropicProvider._build_payload(request)
        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_generation_kwargs_forwarded(self):
        request = _make_request(generation_kwargs={"temperature": 0.9, "top_p": 0.95})
        payload = AnthropicProvider._build_payload(request)
        assert payload["temperature"] == 0.9
        assert payload["top_p"] == 0.95

    def test_output_config_for_structured_output(self):
        request = _make_request(output_schema=Person)
        payload = AnthropicProvider._build_payload(request)
        oc = payload["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert "schema" in oc["format"]
        assert "output_config" not in AnthropicProvider._build_payload(
            _make_request(output_schema=Person), stream=True
        )

    def test_no_output_config_without_schema(self):
        request = _make_request()
        payload = AnthropicProvider._build_payload(request)
        assert "output_config" not in payload


# ---------------------------------------------------------------------------
# _prepare_schema
# ---------------------------------------------------------------------------


class TestPrepareSchema:
    def test_adds_additional_properties_false_to_top_level(self):
        schema = Person.model_json_schema()
        assert "additionalProperties" not in schema
        result = _prepare_schema(schema)
        assert result["additionalProperties"] is False

    def test_adds_additional_properties_false_to_defs(self):
        schema = Recipe.model_json_schema()
        result = _prepare_schema(schema)
        assert result["additionalProperties"] is False
        assert result["$defs"]["Ingredient"]["additionalProperties"] is False

    def test_does_not_mutate_original_schema(self):
        schema = Person.model_json_schema()
        _prepare_schema(schema)
        assert "additionalProperties" not in schema

    def test_preserves_existing_additional_properties(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        result = _prepare_schema(schema)
        assert result["additionalProperties"] is True

    def test_handles_nested_inline_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        }
        result = _prepare_schema(schema)
        assert result["additionalProperties"] is False
        assert result["properties"]["address"]["additionalProperties"] is False

    def test_handles_array_of_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            },
        }
        result = _prepare_schema(schema)
        assert result["properties"]["items"]["items"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_single_text_block(self):
        body = {"content": [{"type": "text", "text": "Hello world"}]}
        assert AnthropicProvider._extract_text(body) == "Hello world"

    def test_multiple_text_blocks(self):
        body = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        }
        assert AnthropicProvider._extract_text(body) == "Hello world"

    def test_empty_content(self):
        body = {"content": []}
        assert AnthropicProvider._extract_text(body) == ""

    def test_missing_content_key(self):
        body = {}
        assert AnthropicProvider._extract_text(body) == ""


# ---------------------------------------------------------------------------
# _send_request — basic text completion
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = AnthropicProvider._send_request(
                _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
            )

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        # Verify the POST call
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("headers", {}).get("x-api-key") == "sk-test"
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_posts_to_correct_url(self):
        mock_resp = _mock_chat_response()
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            AnthropicProvider._send_request(
                _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
            )

        assert mock_post.call_args[0][0] == ANTHROPIC_API_URL

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_chat_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 100})
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            AnthropicProvider._send_request(request, credentials={"ANTHROPIC_API_KEY": "sk-test"})

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 100

    def test_system_instruction_in_payload(self):
        mock_resp = _mock_chat_response()
        request = _make_request(system_instruction="Talk like a pirate")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            AnthropicProvider._send_request(request, credentials={"ANTHROPIC_API_KEY": "sk-test"})

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["system"] == "Talk like a pirate"

    def test_structured_output_payload(self):
        """Verify that output_config is included for structured output."""
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Person)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = AnthropicProvider._send_request(
                request, credentials={"ANTHROPIC_API_KEY": "sk-test"}
            )

        assert result.content == content

        # Verify output_config was included in payload
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        oc = payload["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert "schema" in oc["format"]

    def test_structured_output_nested_payload(self):
        """Verify that nested schemas produce the correct output_config."""
        content = json.dumps(
            {
                "ingredients": [
                    {"name": "tomato", "quantity": 5, "unit": "pieces"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"},
                ]
            }
        )
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Recipe)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = AnthropicProvider._send_request(
                request, credentials={"ANTHROPIC_API_KEY": "sk-test"}
            )

        assert result.content == content

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        oc = payload["output_config"]
        assert oc["format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                AnthropicProvider._stream_response(
                    _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(
                AnthropicProvider._stream_response(
                    _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
                )
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True

    def test_skips_non_delta_events(self):
        """Only content_block_delta events with text_delta are yielded."""
        lines = [
            'data: {"type": "message_start", "message": {"id": "msg_test"}}',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hi'}})}",
            'data: {"type": "content_block_stop", "index": 0}',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}',
            'data: {"type": "message_stop"}',
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                AnthropicProvider._stream_response(
                    _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["hi"]

    def test_skips_empty_text_deltas(self):
        """Chunks with empty text in the delta are silently skipped."""
        lines = [
            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ''}})}",
            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'ok'}})}",
            'data: {"type": "message_stop"}',
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                AnthropicProvider._stream_response(
                    _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["ok"]

    def test_skips_empty_and_event_lines(self):
        """Empty lines and 'event:' lines are ignored by _iter_sse_chunks."""
        lines = [
            "",
            "event: content_block_delta",
            f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hi'}})}",
            "",
            "event: message_stop",
            'data: {"type": "message_stop"}',
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                AnthropicProvider._stream_response(
                    _make_request(), credentials={"ANTHROPIC_API_KEY": "sk-test"}
                )
            )

        assert tokens == ["hi"]
