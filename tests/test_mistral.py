"""Tests for lmdk.providers.mistral — MistralProvider."""

import json
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, UserMessage
from lmdk.provider import RawResponse
from lmdk.providers.mistral import MistralProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "mistral-small-2603",
        "messages": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_chat_response(
    content: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 5
):
    """Build a mock requests.Response that mimics a Mistral chat completion."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics a Mistral SSE stream."""
    lines = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")

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
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_without_system_instruction(self):
        request = _make_request()
        messages = MistralProvider._build_messages(request)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "hi"}

    def test_with_system_instruction(self):
        request = _make_request(system_instruction="Be a pirate.")
        messages = MistralProvider._build_messages(request)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be a pirate."}
        assert messages[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# _send_request — basic text completion
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = MistralProvider._send_request(_make_request(), api_key="test-key")

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        # Verify the POST call
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert payload["model"] == "mistral-small-2603"
        assert "response_format" not in payload

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_chat_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 10})
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            MistralProvider._send_request(request, api_key="test-key")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 10

    def test_structured_output_payload(self):
        """Verify that response_format is included for structured output."""
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Person)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = MistralProvider._send_request(request, api_key="test-key")

        assert result.content == content

        # Verify response_format was included in payload
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Person"
        assert "schema" in rf["json_schema"]

    def test_structured_output_nested_payload(self):
        """Verify that nested schemas produce the correct response_format."""
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
            result = MistralProvider._send_request(request, api_key="test-key")

        assert result.content == content

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Recipe"


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(MistralProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(MistralProvider._stream_response(_make_request(), api_key="test-key"))

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True

    def test_skips_empty_lines_and_non_data_lines(self):
        """Lines that are empty or don't start with 'data: ' are ignored."""
        lines = [
            "",
            ": keep-alive",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(MistralProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["hi"]

    def test_skips_empty_content_deltas(self):
        """Chunks with empty or missing content are silently skipped."""
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ''}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(MistralProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["ok"]
