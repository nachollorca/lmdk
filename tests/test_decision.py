"""Tests for lmdk.decision — decide."""

from typing import ClassVar

import pytest

from lmdk import Question, decide
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
