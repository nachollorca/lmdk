"""Tests for lmdk.providers.openrouter — OpenrouterProvider."""

import json
from unittest.mock import MagicMock, patch

from conftest import make_completion_request

from lmdk.provider import RawResponse
from lmdk.providers.openrouter import OPENROUTER_BASE_URL, OpenrouterProvider

MODEL_ID = "anthropic/claude-sonnet-4.6"


def _make_request(**kwargs):
    kwargs.setdefault("model_id", MODEL_ID)
    return make_completion_request(**kwargs)


def _mock_chat_response(content="Hello there!", prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


# ---------------------------------------------------------------------------
# Model id parsing / URL building
# ---------------------------------------------------------------------------


class TestEndpoint:
    def test_model_id_used_verbatim_with_fixed_base(self):
        assert OpenrouterProvider._parse_model_id(MODEL_ID) == (MODEL_ID, OPENROUTER_BASE_URL)

    def test_build_url_is_fixed_chat_completions(self):
        assert (
            OpenrouterProvider._build_url(OPENROUTER_BASE_URL)
            == "https://openrouter.ai/api/v1/chat/completions"
        )


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    def test_bearer_from_credentials(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
        headers = OpenrouterProvider._build_auth_headers({"OPENROUTER_API_KEY": "secret"})
        assert headers == {"Authorization": "Bearer secret"}

    def test_optional_ranking_headers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.com")
        monkeypatch.setenv("OPENROUTER_APP_TITLE", "My App")
        headers = OpenrouterProvider._build_auth_headers({"OPENROUTER_API_KEY": "secret"})
        assert headers == {
            "Authorization": "Bearer secret",
            "HTTP-Referer": "https://example.com",
            "X-Title": "My App",
        }


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
        mock_resp = _mock_chat_response()
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = OpenrouterProvider._send_request(
                _make_request(), credentials={"OPENROUTER_API_KEY": "secret"}
            )

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == MODEL_ID
        assert payload["stream"] is False

        url = mock_post.call_args.args[0]
        assert url == "https://openrouter.ai/api/v1/chat/completions"

    def test_stream_response(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}"
            for tok in ["Hello", " there"]
        ] + ["data: [DONE]"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                OpenrouterProvider._stream_response(
                    _make_request(), credentials={"OPENROUTER_API_KEY": "secret"}
                )
            )

        assert tokens == ["Hello", " there"]
