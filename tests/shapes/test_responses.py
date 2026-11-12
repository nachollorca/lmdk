"""Tests for lmdk.providers.shapes.responses — ResponsesProvider (the bare ``responses:`` shape)."""

from unittest.mock import MagicMock, patch

import pytest
from conftest import make_completion_request

from lmdk.errors import ProviderError
from lmdk.provider import load_provider
from lmdk.providers.shapes.responses import ResponsesProvider

ENDPOINT = "192.168.10.51:8000"


def _mock_response(content: str = "hello"):
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
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "status": "completed",
    }
    return resp


class TestRegistry:
    def test_load_provider_finds_shape(self):
        assert load_provider("responses") is ResponsesProvider


class TestParseModelId:
    def test_splits_model_and_address(self):
        assert ResponsesProvider._parse_model_id(f"gpt-5.5@{ENDPOINT}") == ("gpt-5.5", ENDPOINT)

    def test_missing_address_raises(self):
        with pytest.raises(ProviderError, match="must include an address"):
            ResponsesProvider._parse_model_id("gpt-5.5")


class TestBuildUrl:
    def test_assumes_http_when_no_scheme(self):
        assert ResponsesProvider._build_url(ENDPOINT) == f"http://{ENDPOINT}/v1/responses"

    def test_preserves_explicit_scheme(self):
        assert (
            ResponsesProvider._build_url("https://host:8443/") == "https://host:8443/v1/responses"
        )


class TestAuthHeaders:
    def test_no_key_means_no_auth(self, monkeypatch):
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        assert ResponsesProvider._build_auth_headers({}) == {}

    def test_local_key_produces_bearer(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        assert ResponsesProvider._build_auth_headers({}) == {"Authorization": "Bearer secret"}


class TestSendRequest:
    def test_posts_to_address_plus_shape_path(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        request = make_completion_request(model_id=f"gpt-5.5@{ENDPOINT}")
        with patch("lmdk.provider.requests.post", return_value=_mock_response()) as mock_post:
            result = ResponsesProvider._send_completion_request(request, credentials={})

        assert mock_post.call_args.args[0] == f"http://{ENDPOINT}/v1/responses"
        assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}
        # the address is stripped from the model sent to the server
        assert mock_post.call_args.kwargs["json"]["model"] == "gpt-5.5"
        assert result.content == "hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
