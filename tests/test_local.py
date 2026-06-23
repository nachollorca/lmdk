"""Tests for lmdk.providers.local — LocalProvider."""

import json
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_completion_request
from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, Message, ThinkingEffort
from lmdk.errors import ProviderError
from lmdk.provider import RawResponse
from lmdk.providers.local import LocalProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENDPOINT = "192.168.10.51:4000"


def _make_request(
    *,
    model_id: str = f"any-model@{ENDPOINT}",
    prompt: Sequence[Message] | None = None,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    generation_kwargs: dict | None = None,
    thinking_effort: ThinkingEffort = "none",
) -> CompletionRequest:
    return make_completion_request(
        model_id=model_id,
        prompt=prompt,
        system_instruction=system_instruction,
        output_schema=output_schema,
        generation_kwargs=generation_kwargs,
        thinking_effort=thinking_effort,
    )


def _mock_chat_response(
    content: str | list = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    *,
    message: dict | None = None,
    usage: dict | None = None,
):
    """Build a mock requests.Response that mimics an OpenAI-compatible completion."""
    resp = MagicMock()
    resp.status_code = 200
    msg = message if message is not None else {"content": content}
    use = (
        usage
        if usage is not None
        else {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    )
    resp.json.return_value = {
        "choices": [{"message": msg}],
        "usage": use,
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics an OpenAI-compatible SSE stream."""
    lines = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


class Person(BaseModel):
    name: str
    age: int


# ---------------------------------------------------------------------------
# Model id parsing
# ---------------------------------------------------------------------------


class TestParseModelId:
    def test_splits_model_and_location(self):
        assert LocalProvider._parse_model_id(f"Qwen3.6-27B-BF16@{ENDPOINT}") == (
            "Qwen3.6-27B-BF16",
            ENDPOINT,
        )

    def test_missing_endpoint_raises(self):
        with pytest.raises(ProviderError, match="must include an endpoint"):
            LocalProvider._parse_model_id("no-endpoint")


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


class TestBuildUrl:
    def test_assumes_http_when_no_scheme(self):
        assert LocalProvider._build_url(ENDPOINT) == f"http://{ENDPOINT}/v1/chat/completions"

    def test_preserves_explicit_scheme_and_strips_trailing_slash(self):
        assert (
            LocalProvider._build_url("https://host:8080/")
            == "https://host:8080/v1/chat/completions"
        )


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    def test_no_key_means_no_auth(self, monkeypatch):
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        assert LocalProvider._build_auth_headers({}) == {}

    def test_key_produces_bearer(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        assert LocalProvider._build_auth_headers({}) == {"Authorization": "Bearer secret"}


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = LocalProvider._send_request(_make_request(), credentials={})

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.thinking is None
        assert result.thinking_tokens == 0

        # The location must be stripped from the model name sent to the server.
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["model"] == "any-model"
        assert payload["stream"] is False
        assert "response_format" not in payload

        # URL is built from the @location suffix.
        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args[0][0]
        assert url == f"http://{ENDPOINT}/v1/chat/completions"

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_chat_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 10})
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            LocalProvider._send_request(request, credentials={})

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 10

    def test_structured_output_payload(self):
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Person)
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = LocalProvider._send_request(request, credentials={})

        assert result.content == content
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Person"
        assert "schema" in rf["json_schema"]


# ---------------------------------------------------------------------------
# Thinking / reasoning extraction
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_plain_string(self):
        assert LocalProvider._extract_text("hello") == "hello"

    def test_none_returns_empty(self):
        assert LocalProvider._extract_text(None) == ""

    def test_list_keeps_only_text_chunks(self):
        content = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "reasoning..."}]},
            {"type": "text", "text": "the "},
            {"type": "text", "text": "answer"},
        ]
        assert LocalProvider._extract_text(content) == "the answer"


class TestExtractThinking:
    def test_reasoning_content_field(self):
        message = {"content": "answer", "reasoning_content": "let me think..."}
        assert LocalProvider._extract_thinking(message) == "let me think..."

    def test_list_thinking_chunks(self):
        message = {
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "step 1. "}]},
                {"type": "text", "text": "done"},
            ]
        }
        assert LocalProvider._extract_thinking(message) == "step 1. "

    def test_absent_returns_none(self):
        assert LocalProvider._extract_thinking({"content": "plain answer"}) is None


class TestExtractThinkingTokens:
    def test_reasoning_tokens_in_completion_details(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 15},
        }
        assert LocalProvider._extract_thinking_tokens(usage) == 15

    def test_thinking_tokens_alias(self):
        usage = {"completion_tokens_details": {"thinking_tokens": 8}}
        assert LocalProvider._extract_thinking_tokens(usage) == 8

    def test_absent_returns_zero(self):
        assert LocalProvider._extract_thinking_tokens({}) == 0


class TestSendRequestThinking:
    def test_o1_style_reasoning_content(self):
        mock_resp = _mock_chat_response(
            message={"content": "42", "reasoning_content": "counting..."},
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        )
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            result = LocalProvider._send_request(_make_request(), credentials={})

        assert result.content == "42"
        assert result.thinking == "counting..."
        assert result.thinking_tokens == 7

    def test_mistral_style_list_content(self):
        content = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]},
            {"type": "text", "text": "ok"},
        ]
        mock_resp = _mock_chat_response(message={"content": content})
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            result = LocalProvider._send_request(_make_request(), credentials={})

        assert result.content == "ok"
        assert result.thinking == "hmm"
        assert result.thinking_tokens == 0


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(LocalProvider._stream_response(_make_request(), credentials={}))

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(LocalProvider._stream_response(_make_request(), credentials={}))

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True
