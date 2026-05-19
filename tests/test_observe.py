"""Tests for ``lmdk.observe`` and the listener seam in ``lmdk._listeners``.

These tests use the ``patch_load_provider`` fixture from ``conftest.py`` so
no real provider is hit.
"""

import pytest

from lmdk import observe
from lmdk.core import complete, complete_batch
from lmdk.datatypes import CompletionRequest, CompletionResponse
from lmdk.errors import ProviderError
from lmdk.observe import CompletionObserver, CompletionRecord, _current_observer
from lmdk.provider import RawResponse


class TestObserverBasics:
    def test_observer_starts_empty(self):
        with observe() as obs:
            assert obs.records == []

    def test_no_active_observer_outside_block(self):
        assert _current_observer() is None
        with observe():
            pass
        assert _current_observer() is None

    def test_records_single_completion(self, patch_load_provider):
        with observe() as obs:
            response = complete(model="fake:model", prompt="hi")

        assert len(obs.records) == 1
        record = obs.records[0]
        assert isinstance(record, CompletionRecord)
        assert isinstance(record.request, CompletionRequest)
        assert isinstance(record.response, CompletionResponse)
        # The recorded response is the same object returned to the caller.
        assert record.response is response
        # The recorded request matches what was sent to the provider.
        assert record.request.model_id == "model"
        assert record.request.prompt[0].content == "hi"

    def test_records_multiple_completions_in_order(self, patch_load_provider):
        with observe() as obs:
            complete(model="fake:model", prompt="first")
            complete(model="fake:model", prompt="second")
            complete(model="fake:model", prompt="third")

        assert [r.request.prompt[0].content for r in obs.records] == [
            "first",
            "second",
            "third",
        ]


class TestObserverScope:
    def test_no_recording_outside_block(self, patch_load_provider):
        observer = CompletionObserver()
        complete(model="fake:model", prompt="hi")
        assert observer.records == []

    def test_observer_inactive_after_block_exits(self, patch_load_provider):
        with observe() as obs:
            complete(model="fake:model", prompt="inside")
        # Subsequent calls do not land in obs.
        complete(model="fake:model", prompt="outside")
        assert len(obs.records) == 1
        assert obs.records[0].request.prompt[0].content == "inside"

    def test_nested_observers_isolate(self, patch_load_provider):
        with observe() as outer:
            complete(model="fake:model", prompt="outer-before")
            with observe() as inner:
                complete(model="fake:model", prompt="inner-1")
                complete(model="fake:model", prompt="inner-2")
            complete(model="fake:model", prompt="outer-after")

        assert [r.request.prompt[0].content for r in inner.records] == [
            "inner-1",
            "inner-2",
        ]
        assert [r.request.prompt[0].content for r in outer.records] == [
            "outer-before",
            "outer-after",
        ]

    def test_streaming_calls_are_not_recorded(self, patch_load_provider):
        with observe() as obs:
            list(complete(model="fake:model", prompt="hi", stream=True))
        assert obs.records == []

    def test_failed_call_is_not_recorded(self, patch_load_provider):
        def boom(request, credentials):
            raise ProviderError(status_code=500, message="server error", provider="fake")

        patch_load_provider.response_fn = boom

        with observe() as obs, pytest.raises(ProviderError):
            complete(model="fake:model", prompt="hi")

        assert obs.records == []

    def test_only_successful_fallback_is_recorded(self, patch_load_provider):
        call_count = {"n": 0}

        def fail_then_succeed(request, credentials):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ProviderError(status_code=500, message="down", provider="fake")
            return RawResponse(content="from fallback", input_tokens=1, output_tokens=2)

        patch_load_provider.response_fn = fail_then_succeed

        with observe() as obs:
            response = complete(model=["fake:bad", "fake:good"], prompt="hi")

        assert len(obs.records) == 1
        assert obs.records[0].response is response
        assert obs.records[0].request.model_id == "good"


class TestObserverWithBatch:
    def test_batch_calls_do_not_propagate_to_observer(self, patch_load_provider):
        """Documented behavior: contextvars don't follow threads spawned by
        ``complete_batch``, so calls inside the pool are not recorded.
        If this ever changes (e.g. via ``contextvars.copy_context``), this
        test should be flipped to assert propagation.
        """
        with observe() as obs:
            results = complete_batch(
                model="fake:model",
                prompt_list=["a", "b", "c"],
                max_workers=2,
            )

        assert len(results) == 3
        assert obs.records == []
