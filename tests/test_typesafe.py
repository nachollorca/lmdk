"""Tests for lmdk.providers.typesafe and lmdk.providers.laya."""

from unittest.mock import MagicMock, patch

import pytest

from lmdk.datatypes import DecisionRequest, Question
from lmdk.errors import ProviderError
from lmdk.providers.laya import LayaProvider
from lmdk.providers.typesafe import TypesafeProvider

QUESTIONS = {
    "spam": Question("Is this spam?", {"Yes": "unsolicited", "No": "legit"}),
    "urgency": Question("How urgent?", {"low": "can wait", "high": "now"}, is_ordered=True),
    "dept": Question("Which team?", {"billing": "money", "tech": "bugs"}),
}

ANSWERS = {
    "spam": {"type": "noul", "noul": 0.75},
    "urgency": {"type": "score", "score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}},
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}},
}


def _decide(provider, model_id: str):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"answers": ANSWERS, "usage": {"input_tokens": 7, "output_tokens": 0}}
    request = DecisionRequest(model_id=model_id, state={"body": "hi"}, questions=QUESTIONS)
    with patch("lmdk.provider.requests.post", return_value=resp) as post:
        result = provider.decide(request)
    return result, post.call_args


def test_typesafe_request_and_response(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    result, call = _decide(TypesafeProvider, "jev-latest")

    assert call.args[0] == "https://api.typesafe.ai/v1/systemone"
    assert call.kwargs["headers"] == {"Authorization": "Bearer k"}
    body = call.kwargs["json"]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"body": "hi"}
    assert body["questions"] == {
        "spam": {
            "type": "noul",
            "instructions": "Is this spam?",
            "criteria": {"true": "unsolicited", "false": "legit"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": [
                {"label": "low", "description": "can wait"},
                {"label": "high", "description": "now"},
            ],
        },
        "dept": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "money", "tech": "bugs"},
        },
    }

    assert result.probabilities == {
        "spam": {"Yes": 0.75, "No": 0.25},
        "urgency": {"low": 0.2, "high": 0.8},
        "dept": {"billing": 0.9, "tech": 0.1},
    }
    assert result.input_tokens == 7
    assert result.output_tokens == 0


def test_laya_uses_location_and_optional_key(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    _, call = _decide(LayaProvider, "english@localhost:8000")
    assert call.args[0] == "http://localhost:8000/v1/systemone"
    assert call.kwargs["headers"] == {}
    assert call.kwargs["json"]["model"] == "english"

    monkeypatch.setenv("LAYA_API_KEY", "s")
    _, call = _decide(LayaProvider, "english@https://laya.example.com/")
    assert call.args[0] == "https://laya.example.com/v1/systemone"
    assert call.kwargs["headers"] == {"Authorization": "Bearer s"}


def test_laya_requires_location():
    with pytest.raises(ProviderError, match="endpoint"):
        _decide(LayaProvider, "english")
