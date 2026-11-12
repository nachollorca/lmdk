"""Tests for lmdk.providers.vendors.typesafe — TypesafeProvider."""

from unittest.mock import MagicMock, patch

from lmdk.datatypes import DecisionRequest, Question
from lmdk.providers.vendors.typesafe import TypesafeProvider

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


def test_vendor_with_address_override_uses_local_key(monkeypatch):
    """``typesafe:jev@host`` hits the address and sends LOCAL_API_KEY, never the vendor key."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "vendor-secret")
    monkeypatch.setenv("LOCAL_API_KEY", "local-secret")
    result, call = _decide(TypesafeProvider, "jev-latest@192.168.10.51:8100")

    assert call.args[0] == "http://192.168.10.51:8100/v1/systemone"
    assert call.kwargs["headers"] == {"Authorization": "Bearer local-secret"}
    assert call.kwargs["json"]["model"] == "jev-latest"
    assert result.probabilities["dept"] == {"billing": 0.9, "tech": 0.1}


def test_vendor_with_address_override_needs_no_env_var(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    _, call = _decide(TypesafeProvider, "jev-latest@https://jev.example.com/")

    assert call.args[0] == "https://jev.example.com/v1/systemone"
    assert call.kwargs["headers"] == {}
