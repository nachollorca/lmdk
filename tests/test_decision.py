"""Tests for lmdk.decision — decide."""

from typing import ClassVar

import pytest

from lmdk import Question, decide, decide_batch
from lmdk.datatypes import DecisionResponse
from lmdk.errors import AllModelsFailedError
from lmdk.provider import Provider

QUESTIONS = {"spam": Question("Is this spam?", {"yes": "unsolicited", "no": "legit"})}


class DecideProvider(Provider):
    requests: ClassVar[list] = []
    failing: ClassVar[set[str]] = set()

    @classmethod
    def _build_auth_headers(cls, credentials):
        return {}

    @classmethod
    def _send_decision_request(cls, request, credentials):
        cls.requests.append(request)
        if request.model_id in cls.failing:
            raise RuntimeError(f"{request.model_id} down")
        return DecisionResponse(
            probabilities={"spam": {"yes": 0.9, "no": 0.1}}, input_tokens=3, output_tokens=0
        )


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    DecideProvider.requests = []
    DecideProvider.failing = set()
    monkeypatch.setattr("lmdk.provider.load_provider", lambda name: DecideProvider)
    return DecideProvider


def test_builds_request_and_returns_response(provider):
    result = decide("fake:m", state={"body": "Buy now"}, questions=QUESTIONS, calling_service="svc")
    assert result.probabilities == {"spam": {"yes": 0.9, "no": 0.1}}
    (request,) = provider.requests
    assert request.model_id == "m"
    assert request.state == {"body": "Buy now"}
    assert request.questions == QUESTIONS
    assert request.calling_service == "svc"


def test_probabilities_rounded_to_5_decimals(provider, monkeypatch):
    raw = DecisionResponse(
        probabilities={"spam": {"yes": 0.123456789, "no": 0.876543211}},
        input_tokens=3,
        output_tokens=0,
    )
    monkeypatch.setattr(
        DecideProvider, "_send_decision_request", classmethod(lambda cls, request, credentials: raw)
    )
    result = decide("fake:m", state="x", questions=QUESTIONS)
    assert result.probabilities == {"spam": {"yes": 0.12346, "no": 0.87654}}


def test_falls_back_to_second_model(provider):
    provider.failing = {"a"}
    result = decide(["fake:a", "fake:b"], state="Buy now", questions=QUESTIONS)
    assert result.input_tokens == 3
    assert [r.model_id for r in provider.requests] == ["a", "b"]


def test_single_model_failure_raises_original(provider):
    provider.failing = {"a"}
    with pytest.raises(RuntimeError, match="a down"):
        decide("fake:a", state="Buy now", questions=QUESTIONS)


def test_all_models_fail_raises(provider):
    provider.failing = {"a", "b"}
    with pytest.raises(AllModelsFailedError) as exc_info:
        decide(["fake:a", "fake:b"], state="Buy now", questions=QUESTIONS)
    assert set(exc_info.value.errors) == {"fake:a", "fake:b"}


# ---------------------------------------------------------------------------
# decide_batch
# ---------------------------------------------------------------------------


def test_batch_results_in_state_order(provider):
    batch = decide_batch(
        "fake:m", state_list=[{"i": 1}, {"i": 2}, {"i": 3}], questions=QUESTIONS, max_workers=3
    )
    assert len(batch) == 3
    assert sorted(r.state["i"] for r in provider.requests) == [1, 2, 3]
    assert [r.probabilities for r in batch.responses] == [{"spam": {"yes": 0.9, "no": 0.1}}] * 3
    assert batch.errors == []


def test_batch_captures_per_state_exceptions(provider, monkeypatch):
    def fail_on_two(cls, request, credentials):
        if request.state == {"i": 2}:
            raise RuntimeError("bad state")
        return DecisionResponse(
            probabilities={"spam": {"yes": 0.9, "no": 0.1}}, input_tokens=3, output_tokens=1
        )

    monkeypatch.setattr(DecideProvider, "_send_decision_request", classmethod(fail_on_two))
    batch = decide_batch("fake:m", state_list=[{"i": 1}, {"i": 2}], questions=QUESTIONS)

    assert isinstance(batch[0], DecisionResponse)
    assert isinstance(batch[1], RuntimeError)
    assert len(batch.responses) == 1
    assert len(batch.errors) == 1


def test_batch_aggregates_tokens_over_successes_only(provider, monkeypatch):
    def half_fail(cls, request, credentials):
        if request.state == {"i": 2}:
            raise RuntimeError("bad state")
        return DecisionResponse(
            probabilities={"spam": {"yes": 0.9, "no": 0.1}}, input_tokens=10, output_tokens=2
        )

    monkeypatch.setattr(DecideProvider, "_send_decision_request", classmethod(half_fail))
    batch = decide_batch("fake:m", state_list=[{"i": 1}, {"i": 2}, {"i": 3}], questions=QUESTIONS)

    assert batch.input_tokens == 20
    assert batch.output_tokens == 4
