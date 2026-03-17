"""Tests for lmtk.datatypes — data contracts used across the app."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from lmtk.datatypes import (
    AssistantMessage,
    CompletionRequest,
    CompletionResponse,
    Message,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_dict(self):
        msg = Message(content="hello", role="user")
        assert msg.to_dict() == {"content": "hello", "role": "user"}

    def test_attributes(self):
        msg = Message(content="hi", role="system")
        assert msg.content == "hi"
        assert msg.role == "system"


# ---------------------------------------------------------------------------
# UserMessage / AssistantMessage
# ---------------------------------------------------------------------------


class TestUserMessage:
    def test_default_role(self):
        msg = UserMessage(content="question")
        assert msg.role == "user"

    def test_to_dict(self):
        msg = UserMessage(content="question")
        assert msg.to_dict() == {"content": "question", "role": "user"}

    def test_is_message(self):
        assert isinstance(UserMessage(content="x"), Message)


class TestAssistantMessage:
    def test_default_role(self):
        msg = AssistantMessage(content="answer")
        assert msg.role == "assistant"

    def test_to_dict(self):
        msg = AssistantMessage(content="answer")
        assert msg.to_dict() == {"content": "answer", "role": "assistant"}

    def test_is_message(self):
        assert isinstance(AssistantMessage(content="x"), Message)


# ---------------------------------------------------------------------------
# CompletionRequest
# ---------------------------------------------------------------------------


class TestCompletionRequest:
    def test_creation(self, sample_messages):
        req = CompletionRequest(
            model_id="some-model",
            messages=sample_messages,
            system_instruction="Be helpful.",
            output_schema=None,
            generation_kwargs={"temperature": 0.5},
        )
        assert req.model_id == "some-model"
        assert len(req.messages) == 3
        assert req.system_instruction == "Be helpful."
        assert req.generation_kwargs == {"temperature": 0.5}

    def test_frozen(self, sample_messages):
        req = CompletionRequest(
            model_id="m",
            messages=sample_messages,
            system_instruction=None,
            output_schema=None,
            generation_kwargs={},
        )
        with pytest.raises(FrozenInstanceError):
            req.model_id = "other"


# ---------------------------------------------------------------------------
# CompletionResponse
# ---------------------------------------------------------------------------


class TestCompletionResponse:
    def test_message_property(self):
        resp = CompletionResponse(
            content="hello",
            input_tokens=10,
            output_tokens=5,
            latency=0.1,
        )
        msg = resp.message
        assert isinstance(msg, AssistantMessage)
        assert msg.content == "hello"
        assert msg.role == "assistant"

    def test_parsed_defaults_to_none(self):
        resp = CompletionResponse(content="x", input_tokens=0, output_tokens=0, latency=0.0)
        assert resp.parsed is None

    def test_parsed_with_schema(self):
        class Mood(BaseModel):
            label: str

        mood = Mood(label="happy")
        resp = CompletionResponse(
            content="happy", input_tokens=1, output_tokens=1, latency=0.0, parsed=mood
        )
        assert resp.parsed == mood
        assert resp.parsed.label == "happy"
