"""Tests for lmdk.datatypes — data contracts used across the app."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from lmdk.datatypes import (
    AssistantMessage,
    CompletionBatch,
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
            prompt=sample_messages,
            system_instruction="Be helpful.",
            output_schema=None,
            generation_kwargs={"temperature": 0.5},
        )
        assert req.model_id == "some-model"
        assert len(req.prompt) == 3
        assert req.system_instruction == "Be helpful."
        assert req.generation_kwargs == {"temperature": 0.5}

    def test_frozen(self, sample_messages):
        req = CompletionRequest(
            model_id="m",
            prompt=sample_messages,
            system_instruction=None,
            output_schema=None,
            generation_kwargs={},
        )
        with pytest.raises(FrozenInstanceError):
            req.model_id = "other"  # ty: ignore[invalid-assignment]


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
        parsed = resp.parsed
        assert parsed is not None
        assert parsed.label == "happy"


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
# CompletionBatch — aggregation
# ---------------------------------------------------------------------------


class TestCompletionBatch:
    def test_aggregates_tokens(self):
        batch = CompletionBatch(
            results=[
                _resp(input_tokens=10, output_tokens=5, latency=0.1),
                _resp(input_tokens=20, output_tokens=15, latency=0.2),
            ]
        )
        assert batch.input_tokens == 30
        assert batch.output_tokens == 20

    def test_latency_is_max(self):
        batch = CompletionBatch(
            results=[_resp(latency=0.1), _resp(latency=0.5), _resp(latency=0.3)]
        )
        assert batch.latency == 0.5

    def test_collects_parsed_objects(self):
        r1 = _resp(parsed=SingleField(summary="a"))
        r2 = _resp(parsed=SingleField(summary="b"))
        batch = CompletionBatch(results=[r1, r2])

        assert len(batch.parsed) == 2
        assert isinstance(batch.parsed[0], SingleField)
        assert batch.parsed[0].summary == "a"
        assert isinstance(batch.parsed[1], SingleField)
        assert batch.parsed[1].summary == "b"

    def test_skips_none_parsed(self):
        r1 = _resp(parsed=SingleField(summary="a"))
        r2 = _resp(parsed=None)
        batch = CompletionBatch(results=[r1, r2])
        assert len(batch.parsed) == 1

    def test_empty_batch(self):
        batch = CompletionBatch(results=[])
        assert batch.input_tokens == 0
        assert batch.output_tokens == 0
        assert batch.latency == 0.0
        assert batch.parsed == []

    def test_frozen(self):
        batch = CompletionBatch(results=[])
        with pytest.raises(FrozenInstanceError):
            batch.results = [_resp()]  # ty: ignore[invalid-assignment]

    def test_exceptions_are_separated_from_responses(self):
        r1 = _resp(input_tokens=5, output_tokens=3, latency=0.1)
        err = RuntimeError("boom")
        r2 = _resp(input_tokens=7, output_tokens=4, latency=0.2)
        batch = CompletionBatch(results=[r1, err, r2])

        # responses / errors split correctly, preserving order
        assert batch.responses == [r1, r2]
        assert batch.errors == [err]

        # aggregates ignore the exception
        assert batch.input_tokens == 12
        assert batch.output_tokens == 7
        assert batch.latency == 0.2

    def test_iterable_and_indexable(self):
        r1 = _resp()
        err = RuntimeError("boom")
        batch = CompletionBatch(results=[r1, err])

        assert len(batch) == 2
        assert batch[0] is r1
        assert batch[1] is err
        assert list(batch) == [r1, err]


# ---------------------------------------------------------------------------
# CompletionBatch.output
# ---------------------------------------------------------------------------


class TestCompletionBatchOutput:
    def test_single_field_models_unwrap(self):
        batch = CompletionBatch(
            results=[
                _resp(parsed=SingleField(summary="a")),
                _resp(parsed=SingleField(summary="b")),
            ]
        )
        assert batch.output == ["a", "b"]

    def test_multi_field_models_stay_as_models(self):
        m1 = MultiField(title="A", score=0.9)
        m2 = MultiField(title="B", score=0.8)
        batch = CompletionBatch(results=[_resp(parsed=m1), _resp(parsed=m2)])
        assert batch.output == [m1, m2]

    def test_list_fields_flatten(self):
        batch = CompletionBatch(
            results=[
                _resp(parsed=ListField(items=["a", "b"])),
                _resp(parsed=ListField(items=["c"])),
            ]
        )
        assert batch.output == ["a", "b", "c"]

    def test_empty_batch_returns_empty_list(self):
        assert CompletionBatch(results=[]).output == []

    def test_mixed_list_and_scalar_does_not_flatten(self):
        """When not all outputs are lists, no flattening occurs."""

        class MixedField(BaseModel):
            value: str

        batch = CompletionBatch(
            results=[
                _resp(parsed=ListField(items=["a", "b"])),
                _resp(parsed=MixedField(value="c")),
            ]
        )
        # First unwraps to ["a", "b"], second unwraps to "c" — not all lists
        assert batch.output == [["a", "b"], "c"]
