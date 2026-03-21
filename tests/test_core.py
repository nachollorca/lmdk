"""Tests for lmdk.core — complete and complete_batch."""

import pytest

# We import the public functions; load_provider is patched via the
# ``patch_load_provider`` fixture from conftest.
from lmdk.core import complete, complete_batch
from lmdk.datatypes import CompletionResponse, UserMessage
from lmdk.errors import AllModelsFailedError, ProviderError
from lmdk.provider import RawResponse

# ---------------------------------------------------------------------------
# complete — input normalization
# ---------------------------------------------------------------------------


class TestCompleteInputs:
    def test_string_message_converted_to_user_message(self, patch_load_provider):
        """A plain string should be wrapped in a UserMessage."""
        captured = {}

        def spy(request, api_key):
            captured["messages"] = request.messages
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = spy
        complete(model="fake:model", messages="hello")

        assert len(captured["messages"]) == 1
        assert isinstance(captured["messages"][0], UserMessage)
        assert captured["messages"][0].content == "hello"

    def test_default_generation_kwargs(self, patch_load_provider):
        """When no generation_kwargs is passed, default to temperature=0."""
        captured = {}

        def spy(request, api_key):
            captured["kwargs"] = request.generation_kwargs
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = spy
        complete(model="fake:model", messages="hi")

        assert captured["kwargs"] == {"temperature": 0}

    def test_custom_generation_kwargs(self, patch_load_provider):
        captured = {}

        def spy(request, api_key):
            captured["kwargs"] = request.generation_kwargs
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = spy
        complete(model="fake:model", messages="hi", generation_kwargs={"temperature": 0.7})

        assert captured["kwargs"] == {"temperature": 0.7}


# ---------------------------------------------------------------------------
# complete — validation
# ---------------------------------------------------------------------------


class TestCompleteValidation:
    def test_stream_and_output_schema_raises(self, patch_load_provider):
        from pydantic import BaseModel

        class Dummy(BaseModel):
            x: int

        with pytest.raises(ValueError, match="Only `stream` or `output_schema`"):
            complete(
                model="fake:model",
                messages="hi",
                output_schema=Dummy,
                stream=True,
            )


# ---------------------------------------------------------------------------
# complete — single model
# ---------------------------------------------------------------------------


class TestCompleteSingleModel:
    def test_returns_response(self, patch_load_provider):
        result = complete(model="fake:model", messages="hi")
        assert isinstance(result, CompletionResponse)
        assert result.content == "fake response"

    def test_stream_returns_iterator(self, patch_load_provider):
        result = complete(model="fake:model", messages="hi", stream=True)
        assert list(result) == ["chunk1", "chunk2"]

    def test_provider_exception_propagates(self, patch_load_provider):
        def boom(request, api_key):
            raise ProviderError(status_code=500, message="server error", provider="fake")

        patch_load_provider.response_fn = boom

        with pytest.raises(ProviderError, match="server error"):
            complete(model="fake:model", messages="hi")


# ---------------------------------------------------------------------------
# complete — fallback across multiple models
# ---------------------------------------------------------------------------


class TestCompleteFallback:
    def test_falls_back_to_second_model(self, patch_load_provider):
        call_count = {"n": 0}

        def fail_then_succeed(request, api_key):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first model down")
            return RawResponse(content="from fallback", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = fail_then_succeed

        result = complete(model=["fake:model1", "fake:model2"], messages="hi")
        assert result.content == "from fallback"
        assert call_count["n"] == 2

    def test_all_models_fail_raises(self, patch_load_provider):
        def always_fail(request, api_key):
            raise RuntimeError("down")

        patch_load_provider.response_fn = always_fail

        with pytest.raises(AllModelsFailedError) as exc_info:
            complete(model=["fake:a", "fake:b"], messages="hi")

        assert len(exc_info.value.errors) == 2

    def test_single_model_failure_raises_original(self, patch_load_provider):
        """With one model, the original exception is re-raised (not AllModelsFailedError)."""

        def fail(request, api_key):
            raise ProviderError(status_code=503, message="unavailable", provider="fake")

        patch_load_provider.response_fn = fail

        with pytest.raises(ProviderError, match="unavailable"):
            complete(model="fake:model", messages="hi")


# ---------------------------------------------------------------------------
# complete_batch
# ---------------------------------------------------------------------------


class TestCompleteBatch:
    def test_returns_results_in_order(self, patch_load_provider):
        def echo(request, api_key):
            text = request.messages[0].content
            return RawResponse(content=text, input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = echo

        results = complete_batch(
            model="fake:model",
            messages_list=["first", "second", "third"],
        )

        assert len(results) == 3
        assert results[0].content == "first"
        assert results[1].content == "second"
        assert results[2].content == "third"

    def test_captures_per_item_exceptions(self, patch_load_provider):
        def fail_on_second(request, api_key):
            if request.messages[0].content == "fail":
                raise RuntimeError("bad input")
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = fail_on_second

        results = complete_batch(
            model="fake:model",
            messages_list=["good", "fail", "good"],
        )

        assert isinstance(results[0], CompletionResponse)
        assert isinstance(results[1], RuntimeError)
        assert isinstance(results[2], CompletionResponse)
