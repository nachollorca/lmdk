"""Tests for lmdk.providers.llamacpp — LlamacppProvider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from lmdk.datatypes import CompletionRequest, UserMessage
from lmdk.provider import RawResponse
from lmdk.providers.llamacpp import LlamacppProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "any-model",
        "prompt": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_chat_response(
    content: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 5
):
    """Build a mock requests.Response that mimics a Llama.cpp chat completion."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics a Llama.cpp SSE stream."""
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
# URL Building
# ---------------------------------------------------------------------------


class TestGetBaseUrl:
    def test_build_url_with_http(self):
        credentials = {"LLAMACPP_URL": "http://localhost", "LLAMACPP_PORT": "8080"}
        url = LlamacppProvider._get_base_url(credentials)
        assert url == "http://localhost:8080/v1/chat/completions"

    def test_build_url_without_protocol(self):
        credentials = {"LLAMACPP_URL": "127.0.0.1", "LLAMACPP_PORT": "8080"}
        url = LlamacppProvider._get_base_url(credentials)
        assert url == "http://127.0.0.1:8080/v1/chat/completions"


# ---------------------------------------------------------------------------
# _build_prompt_payload
# ---------------------------------------------------------------------------


class TestBuildPromptPayload:
    def test_without_system_instruction(self):
        request = _make_request()
        payload = LlamacppProvider._build_prompt_payload(request)
        assert len(payload) == 1
        assert payload[0] == {"role": "user", "content": "hi"}

    def test_with_system_instruction(self):
        request = _make_request(system_instruction="Be a pirate.")
        payload = LlamacppProvider._build_prompt_payload(request)
        assert len(payload) == 2
        assert payload[0] == {"role": "system", "content": "Be a pirate."}
        assert payload[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# _send_request — basic text completion
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        credentials = {"LLAMACPP_URL": "localhost", "LLAMACPP_PORT": "8080"}
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = LlamacppProvider._send_request(_make_request(), credentials=credentials)

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        # Verify the POST call
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert payload["model"] == "any-model"
        assert payload["stream"] is False

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_chat_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 10})
        credentials = {"LLAMACPP_URL": "localhost", "LLAMACPP_PORT": "8080"}
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            LlamacppProvider._send_request(request, credentials=credentials)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 10

    def test_structured_output_not_implemented(self):
        """Verify that output_schema raises NotImplementedError."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        request = _make_request(output_schema=Person)
        credentials = {"LLAMACPP_URL": "localhost", "LLAMACPP_PORT": "8080"}
        with pytest.raises(NotImplementedError, match="Structured output is not implemented"):
            LlamacppProvider._send_request(request, credentials=credentials)


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        credentials = {"LLAMACPP_URL": "localhost", "LLAMACPP_PORT": "8080"}
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                LlamacppProvider._stream_response(_make_request(), credentials=credentials)
            )

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        credentials = {"LLAMACPP_URL": "localhost", "LLAMACPP_PORT": "8080"}
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(LlamacppProvider._stream_response(_make_request(), credentials=credentials))

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True
