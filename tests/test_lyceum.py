"""Tests for lmdk.providers.lyceum — LyceumProvider.

The chat-completions protocol itself is covered by ``test_local.py``; these
tests only pin Lyceum's endpoint and authentication.
"""

import json
from unittest.mock import MagicMock, patch

from conftest import make_completion_request
from pydantic import BaseModel

from lmdk.provider import RawResponse
from lmdk.providers.lyceum import LYCEUM_BASE_URL, LyceumProvider

MODEL_ID = "moonshotai/kimi-k3"
CREDENTIALS = {"LYCEUM_API_KEY": "secret"}


class Person(BaseModel):
    name: str
    age: int


def _make_request(**kwargs):
    kwargs.setdefault("model_id", MODEL_ID)
    return make_completion_request(**kwargs)


def _mock_chat_response(content="Hello there!"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return resp


class TestEndpoint:
    def test_model_id_used_verbatim_with_fixed_base(self):
        assert LyceumProvider._parse_model_id(MODEL_ID) == (MODEL_ID, LYCEUM_BASE_URL)

    def test_build_url_is_fixed_chat_completions(self):
        assert LyceumProvider._build_url(LYCEUM_BASE_URL) == f"{LYCEUM_BASE_URL}/chat/completions"


class TestAuthHeaders:
    def test_bearer_from_credentials(self):
        assert LyceumProvider._build_auth_headers(CREDENTIALS) == {"Authorization": "Bearer secret"}


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response()
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = LyceumProvider._send_request(_make_request(), credentials=CREDENTIALS)

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == MODEL_ID
        assert payload["stream"] is False
        assert mock_post.call_args.args[0] == f"{LYCEUM_BASE_URL}/chat/completions"

    def test_structured_output_is_strict(self):
        content = '{"name": "Alice", "age": 30}'
        with patch("lmdk.provider.requests.post", return_value=_mock_chat_response(content)) as p:
            result = LyceumProvider._send_request(
                _make_request(output_schema=Person), credentials=CREDENTIALS
            )

        assert result.content == content
        json_schema = p.call_args.kwargs["json"]["response_format"]["json_schema"]
        assert json_schema["name"] == "Person"
        assert json_schema["strict"] is True
        assert json_schema["schema"]["additionalProperties"] is False


class TestStreamResponse:
    def test_yields_tokens(self):
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}"
            for tok in ["Hello", " there"]
        ] + ["data: [DONE]"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(LyceumProvider._stream_response(_make_request(), CREDENTIALS))

        assert tokens == ["Hello", " there"]
