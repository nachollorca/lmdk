"""Shared fixtures for lmtk tests."""

from collections.abc import Callable, Iterator

import pytest

from lmtk.datatypes import (
    AssistantMessage,
    CompletionRequest,
    CompletionResponse,
    Message,
    UserMessage,
)
from lmtk.provider import Provider, RawResponse

# ---------------------------------------------------------------------------
# FakeProvider — a concrete Provider for testing the abstract layer
# ---------------------------------------------------------------------------

_DEFAULT_RAW = RawResponse(
    content="fake response",
    input_tokens=10,
    output_tokens=5,
)


class FakeProvider(Provider):
    """Minimal concrete ``Provider`` whose behaviour is controlled per-test.

    Class-level callables are swapped by tests to simulate successes,
    failures, or streaming output without hitting any real API.
    """

    api_key_name: str = "FAKE_API_KEY"

    # Callables that tests can override
    response_fn: Callable[[CompletionRequest, str], RawResponse] | None = None
    stream_fn: Callable[[CompletionRequest, str], Iterator[str]] | None = None

    @classmethod
    def _build_auth_headers(cls, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def _send_request(cls, request: CompletionRequest, api_key: str) -> RawResponse:
        if cls.response_fn is not None:
            return cls.response_fn(request, api_key)
        return _DEFAULT_RAW

    @classmethod
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        if cls.stream_fn is not None:
            return cls.stream_fn(request, api_key)
        return iter(["chunk1", "chunk2"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_provider(monkeypatch):
    """Yield a clean ``FakeProvider`` and reset class state afterwards."""
    monkeypatch.setenv("FAKE_API_KEY", "test-key-123")
    # Reset callables before each test
    FakeProvider.response_fn = None
    FakeProvider.stream_fn = None
    yield FakeProvider
    # Cleanup happens via monkeypatch teardown


@pytest.fixture()
def patch_load_provider(monkeypatch, fake_provider):
    """Monkeypatch ``load_provider`` in ``lmtk.core`` to return ``FakeProvider``."""

    def _load(name: str):
        return fake_provider

    monkeypatch.setattr("lmtk.core.load_provider", _load)
    return fake_provider


@pytest.fixture()
def sample_messages() -> list[Message]:
    """A small conversation for reuse across tests."""
    return [
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi there!"),
        UserMessage(content="How are you?"),
    ]
