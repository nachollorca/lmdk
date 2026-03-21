"""Tests for lmdk.datatypes — data contracts used across the app."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from lmdk.datatypes import (
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


# ---------------------------------------------------------------------------
# Pydantic helpers used across output tests
# ---------------------------------------------------------------------------


class SingleField(BaseModel):
    summary: str


class MultiField(BaseModel):
    title: str
    score: float


class ListField(BaseModel):
    items: list[str]


def _resp(content="x", parsed=None, **kw):
    defaults = {"input_tokens": 10, "output_tokens": 5, "latency": 0.1}
    defaults.update(kw)
    return CompletionResponse(content=content, parsed=parsed, **defaults)


# ---------------------------------------------------------------------------
# CompletionResponse.output — single response
# ---------------------------------------------------------------------------


class TestOutputSingleResponse:
    def test_no_parsed_returns_content(self):
        assert _resp(content="hello").output == "hello"

    def test_single_field_model_unwraps(self):
        resp = _resp(parsed=SingleField(summary="TL;DR"))
        assert resp.output == "TL;DR"

    def test_multi_field_model_returns_model(self):
        model = MultiField(title="A", score=0.9)
        resp = _resp(parsed=model)
        assert resp.output is model

    def test_single_field_list_unwraps_to_list(self):
        resp = _resp(parsed=ListField(items=["a", "b"]))
        assert resp.output == ["a", "b"]


# ---------------------------------------------------------------------------
# CompletionResponse.from_list — aggregation
# ---------------------------------------------------------------------------


class TestFromList:
    def test_aggregates_tokens(self):
        responses = [
            _resp(input_tokens=10, output_tokens=5, latency=0.1),
            _resp(input_tokens=20, output_tokens=15, latency=0.2),
        ]
        agg = CompletionResponse.from_list(responses)
        assert agg.input_tokens == 30
        assert agg.output_tokens == 20

    def test_latency_is_max(self):
        responses = [
            _resp(latency=0.1),
            _resp(latency=0.5),
            _resp(latency=0.3),
        ]
        agg = CompletionResponse.from_list(responses)
        assert agg.latency == 0.5

    def test_collects_parsed_objects(self):
        r1 = _resp(parsed=SingleField(summary="a"))
        r2 = _resp(parsed=SingleField(summary="b"))
        agg = CompletionResponse.from_list([r1, r2])

        assert isinstance(agg.parsed, list)
        assert len(agg.parsed) == 2
        assert agg.parsed[0].summary == "a"
        assert agg.parsed[1].summary == "b"

    def test_skips_none_parsed(self):
        r1 = _resp(parsed=SingleField(summary="a"))
        r2 = _resp(parsed=None)
        agg = CompletionResponse.from_list([r1, r2])

        assert isinstance(agg.parsed, list)
        assert len(agg.parsed) == 1

    def test_empty_list(self):
        agg = CompletionResponse.from_list([])
        assert agg.input_tokens == 0
        assert agg.output_tokens == 0
        assert agg.parsed == []

    def test_content_is_empty(self):
        agg = CompletionResponse.from_list([_resp(content="hello")])
        assert agg.content == ""


# ---------------------------------------------------------------------------
# CompletionResponse.output — aggregated responses
# ---------------------------------------------------------------------------


class TestOutputAggregated:
    def test_single_field_models_unwrap(self):
        responses = [
            _resp(parsed=SingleField(summary="a")),
            _resp(parsed=SingleField(summary="b")),
        ]
        agg = CompletionResponse.from_list(responses)
        assert agg.output == ["a", "b"]

    def test_multi_field_models_stay_as_models(self):
        m1 = MultiField(title="A", score=0.9)
        m2 = MultiField(title="B", score=0.8)
        responses = [_resp(parsed=m1), _resp(parsed=m2)]
        agg = CompletionResponse.from_list(responses)

        assert agg.output == [m1, m2]

    def test_list_fields_flatten(self):
        responses = [
            _resp(parsed=ListField(items=["a", "b"])),
            _resp(parsed=ListField(items=["c"])),
        ]
        agg = CompletionResponse.from_list(responses)
        assert agg.output == ["a", "b", "c"]

    def test_empty_aggregation_returns_empty_list(self):
        agg = CompletionResponse.from_list([])
        assert agg.output == []

    def test_mixed_list_and_scalar_does_not_flatten(self):
        """When not all outputs are lists, no flattening occurs."""

        class MixedField(BaseModel):
            value: str

        responses = [
            _resp(parsed=ListField(items=["a", "b"])),
            _resp(parsed=MixedField(value="c")),
        ]
        agg = CompletionResponse.from_list(responses)
        # First unwraps to ["a", "b"], second unwraps to "c" — not all lists
        assert agg.output == [["a", "b"], "c"]
