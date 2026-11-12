"""Tests for lmdk.providers.shapes.messages — MessagesProvider (the bare ``messages:`` shape)."""

from unittest.mock import MagicMock, patch

import pytest
from conftest import make_completion_request

from lmdk.errors import ProviderError
from lmdk.provider import load_provider
from lmdk.providers.shapes.messages import ANTHROPIC_VERSION, MessagesProvider

ENDPOINT = "192.168.10.51:8000"


def _mock_response(content: str = "hello"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }
    return resp


class TestRegistry:
    def test_load_provider_finds_shape(self):
        assert load_provider("messages") is MessagesProvider


class TestParseModelId:
    def test_splits_model_and_address(self):
        assert MessagesProvider._parse_model_id(f"claude-sonnet-4@{ENDPOINT}") == (
            "claude-sonnet-4",
            ENDPOINT,
        )

    def test_missing_address_raises(self):
        with pytest.raises(ProviderError, match="must include an address"):
            MessagesProvider._parse_model_id("claude-sonnet-4")


class TestBuildUrl:
    def test_assumes_http_when_no_scheme(self):
        assert MessagesProvider._build_url(ENDPOINT) == f"http://{ENDPOINT}/v1/messages"

    def test_preserves_explicit_scheme(self):
        assert MessagesProvider._build_url("https://host:8443/") == "https://host:8443/v1/messages"


class TestAuthHeaders:
    def test_no_key_means_only_version_header(self, monkeypatch):
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        assert MessagesProvider._build_auth_headers({}) == {"anthropic-version": ANTHROPIC_VERSION}

    def test_local_key_uses_x_api_key_format(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        assert MessagesProvider._build_auth_headers({}) == {
            "x-api-key": "secret",
            "anthropic-version": ANTHROPIC_VERSION,
        }


class TestSendRequest:
    def test_posts_to_address_plus_shape_path(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        request = make_completion_request(model_id=f"claude-sonnet-4@{ENDPOINT}")
        with patch("lmdk.provider.requests.post", return_value=_mock_response()) as mock_post:
            result = MessagesProvider._send_completion_request(request, credentials={})

        assert mock_post.call_args.args[0] == f"http://{ENDPOINT}/v1/messages"
        assert mock_post.call_args.kwargs["headers"] == {
            "x-api-key": "secret",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        # the address is stripped from the model sent to the server
        assert mock_post.call_args.kwargs["json"]["model"] == "claude-sonnet-4"
        assert result.content == "hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.finish_reason == "end_turn"
