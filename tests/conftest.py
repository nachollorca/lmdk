"""Shared fixtures for lmdk tests."""

from collections.abc import Callable, Iterator, Sequence

import pytest
from pydantic import BaseModel

from lmdk.datatypes import (
    AssistantMessage,
    CompletionRequest,
    Message,
    ThinkingEffort,
    UserMessage,
)
from lmdk.provider import Provider, RawResponse


def make_completion_request(
    *,
    model_id: str,
    prompt: Sequence[Message] | None = None,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    generation_kwargs: dict | None = None,
    thinking_effort: ThinkingEffort = "none",
) -> CompletionRequest:
    """Build a :class:`CompletionRequest` with typed defaults for tests."""
    return CompletionRequest(
        model_id=model_id,
        prompt=prompt if prompt is not None else [UserMessage(content="hi")],
        system_instruction=system_instruction,
        output_schema=output_schema,
        generation_kwargs=generation_kwargs if generation_kwargs is not None else {},
        thinking_effort=thinking_effort,
    )


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

    required_env: tuple[str, ...] = ("FAKE_API_KEY",)

    # Callables that tests can override
    response_fn: Callable[[CompletionRequest, dict[str, str]], RawResponse] | None = None
    stream_fn: Callable[[CompletionRequest, dict[str, str]], Iterator[str]] | None = None

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        return {"Authorization": f"Bearer {credentials['FAKE_API_KEY']}"}

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        if cls.response_fn is not None:
            return cls.response_fn(request, credentials)
        return _DEFAULT_RAW

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        if cls.stream_fn is not None:
            return cls.stream_fn(request, credentials)
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
    """Monkeypatch ``load_provider`` in ``lmdk.core`` to return ``FakeProvider``."""

    def _load(name: str):
        return fake_provider

    monkeypatch.setattr("lmdk.core.load_provider", _load)
    return fake_provider


@pytest.fixture()
def sample_messages() -> list[Message]:
    """A small conversation for reuse across tests."""
    return [
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi there!"),
        UserMessage(content="How are you?"),
    ]
