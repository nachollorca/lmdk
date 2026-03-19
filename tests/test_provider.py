"""Tests for lmtk.provider — Provider ABC and load_provider."""

from unittest.mock import MagicMock, patch

import pytest

from lmtk.datatypes import CompletionRequest, CompletionResponse, UserMessage
from lmtk.errors import AuthenticationError, InternalServerError, ProviderError, RateLimitError
from lmtk.provider import Provider, RawResponse, load_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "test-model",
        "messages": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_http_response(status_code: int, reason: str = "Error", text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = reason
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Provider.get_response — credential resolution & dispatch
# ---------------------------------------------------------------------------


class TestProviderGetResponse:
    def test_raises_auth_error_when_key_missing(self, fake_provider, monkeypatch):
        monkeypatch.delenv("FAKE_API_KEY", raising=False)
        with pytest.raises(AuthenticationError, match="FAKE_API_KEY"):
            fake_provider.get_response(request=_make_request(), stream=False)

    def test_delegates_to_get_response(self, fake_provider):
        result = fake_provider.get_response(request=_make_request(), stream=False)
        assert isinstance(result, CompletionResponse)
        assert result.content == "fake response"

    def test_delegates_to_stream(self, fake_provider):
        result = fake_provider.get_response(request=_make_request(), stream=True)
        assert list(result) == ["chunk1", "chunk2"]

    def test_custom_response_fn(self, fake_provider):
        custom = RawResponse(content="custom", input_tokens=0, output_tokens=0)
        fake_provider.response_fn = lambda req, key: custom

        result = fake_provider.get_response(request=_make_request(), stream=False)
        assert result.content == "custom"

    def test_custom_stream_fn(self, fake_provider):
        fake_provider.stream_fn = lambda req, key: iter(["a", "b", "c"])

        result = fake_provider.get_response(request=_make_request(), stream=True)
        assert list(result) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Provider._make_request — HTTP POST wrapper with error mapping
# ---------------------------------------------------------------------------


class TestMakeRequest:
    def test_200_returns_response(self, fake_provider):
        mock_resp = _mock_http_response(200)
        with patch("lmtk.provider.requests.post", return_value=mock_resp):
            result = fake_provider._make_request("https://example.com", json={"a": 1})
        assert result is mock_resp

    def test_401_raises_auth_error(self, fake_provider):
        mock_resp = _mock_http_response(401, reason="Unauthorized", text="bad key")
        with patch("lmtk.provider.requests.post", return_value=mock_resp):
            with pytest.raises(AuthenticationError) as exc_info:
                fake_provider._make_request("https://example.com", json={})
        assert exc_info.value.status_code == 401
        assert exc_info.value.body == "bad key"

    def test_429_raises_rate_limit(self, fake_provider):
        mock_resp = _mock_http_response(429, reason="Too Many Requests")
        with patch("lmtk.provider.requests.post", return_value=mock_resp):
            with pytest.raises(RateLimitError) as exc_info:
                fake_provider._make_request("https://example.com", json={})
        assert exc_info.value.status_code == 429

    def test_500_raises_internal_server_error(self, fake_provider):
        mock_resp = _mock_http_response(500, reason="Internal Server Error")
        with patch("lmtk.provider.requests.post", return_value=mock_resp):
            with pytest.raises(InternalServerError) as exc_info:
                fake_provider._make_request("https://example.com", json={})
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# load_provider — dynamic import
# ---------------------------------------------------------------------------


class TestLoadProvider:
    def test_loads_mistral(self):
        cls = load_provider("mistral")
        assert cls.__name__ == "MistralProvider"
        assert issubclass(cls, Provider)

    def test_unknown_provider_raises_import_error(self):
        with pytest.raises(ModuleNotFoundError):
            load_provider("nonexistent_provider_xyz")

    def test_stub_provider_raises_attribute_error(self):
        """llamacpp.py exists but has no VertexProvider class yet."""
        with pytest.raises(AttributeError):
            load_provider("llamacpp")
