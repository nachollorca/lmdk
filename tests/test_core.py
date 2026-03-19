"""Tests for lmtk.core — get_response and get_response_batch."""

import pytest

# We import the public functions; load_provider is patched via the
# ``patch_load_provider`` fixture from conftest.
from lmtk.core import get_response, get_response_batch
from lmtk.datatypes import CompletionResponse, UserMessage
from lmtk.errors import AllModelsFailedError, ProviderError
from lmtk.provider import RawResponse

# ---------------------------------------------------------------------------
# get_response — input normalization
# ---------------------------------------------------------------------------


class TestGetResponseInputs:
    def test_string_message_converted_to_user_message(self, patch_load_provider):
        """A plain string should be wrapped in a UserMessage."""
        captured = {}

        def spy(request, api_key):
            captured["messages"] = request.messages
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = spy
        get_response(model="fake:model", messages="hello")

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
        get_response(model="fake:model", messages="hi")

        assert captured["kwargs"] == {"temperature": 0}

    def test_custom_generation_kwargs(self, patch_load_provider):
        captured = {}

        def spy(request, api_key):
            captured["kwargs"] = request.generation_kwargs
            return RawResponse(content="ok", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = spy
        get_response(model="fake:model", messages="hi", generation_kwargs={"temperature": 0.7})

        assert captured["kwargs"] == {"temperature": 0.7}


# ---------------------------------------------------------------------------
# get_response — validation
# ---------------------------------------------------------------------------


class TestGetResponseValidation:
    def test_stream_and_output_schema_raises(self, patch_load_provider):
        from pydantic import BaseModel

        class Dummy(BaseModel):
            x: int

        with pytest.raises(ValueError, match="Only `stream` or `output_schema`"):
            get_response(
                model="fake:model",
                messages="hi",
                output_schema=Dummy,
                stream=True,
            )


# ---------------------------------------------------------------------------
# get_response — single model
# ---------------------------------------------------------------------------


class TestGetResponseSingleModel:
    def test_returns_response(self, patch_load_provider):
        result = get_response(model="fake:model", messages="hi")
        assert isinstance(result, CompletionResponse)
        assert result.content == "fake response"

    def test_stream_returns_iterator(self, patch_load_provider):
        result = get_response(model="fake:model", messages="hi", stream=True)
        assert list(result) == ["chunk1", "chunk2"]

    def test_provider_exception_propagates(self, patch_load_provider):
        def boom(request, api_key):
            raise ProviderError(status_code=500, message="server error", provider="fake")

        patch_load_provider.response_fn = boom

        with pytest.raises(ProviderError, match="server error"):
            get_response(model="fake:model", messages="hi")


# ---------------------------------------------------------------------------
# get_response — fallback across multiple models
# ---------------------------------------------------------------------------


class TestGetResponseFallback:
    def test_falls_back_to_second_model(self, patch_load_provider):
        call_count = {"n": 0}

        def fail_then_succeed(request, api_key):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first model down")
            return RawResponse(content="from fallback", input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = fail_then_succeed

        result = get_response(model=["fake:model1", "fake:model2"], messages="hi")
        assert result.content == "from fallback"
        assert call_count["n"] == 2

    def test_all_models_fail_raises(self, patch_load_provider):
        def always_fail(request, api_key):
            raise RuntimeError("down")

        patch_load_provider.response_fn = always_fail

        with pytest.raises(AllModelsFailedError) as exc_info:
            get_response(model=["fake:a", "fake:b"], messages="hi")

        assert len(exc_info.value.errors) == 2

    def test_single_model_failure_raises_original(self, patch_load_provider):
        """With one model, the original exception is re-raised (not AllModelsFailedError)."""

        def fail(request, api_key):
            raise ProviderError(status_code=503, message="unavailable", provider="fake")

        patch_load_provider.response_fn = fail

        with pytest.raises(ProviderError, match="unavailable"):
            get_response(model="fake:model", messages="hi")


# ---------------------------------------------------------------------------
# get_response_batch
# ---------------------------------------------------------------------------


class TestGetResponseBatch:
    def test_returns_results_in_order(self, patch_load_provider):
        def echo(request, api_key):
            text = request.messages[0].content
            return RawResponse(content=text, input_tokens=0, output_tokens=0)

        patch_load_provider.response_fn = echo

        results = get_response_batch(
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

        results = get_response_batch(
            model="fake:model",
            messages_list=["good", "fail", "good"],
        )

        assert isinstance(results[0], CompletionResponse)
        assert isinstance(results[1], RuntimeError)
        assert isinstance(results[2], CompletionResponse)
