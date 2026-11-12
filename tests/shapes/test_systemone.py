"""Tests for lmdk.providers.shapes.systemone — SystemoneProvider (the bare ``systemone:`` shape)."""

from unittest.mock import MagicMock, patch

import pytest

from lmdk.datatypes import DecisionRequest, Question
from lmdk.errors import ProviderError
from lmdk.provider import load_provider, resolve_model
from lmdk.providers.shapes.systemone import SystemoneProvider

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

ENDPOINT = "192.168.10.51:8100"


def _decide(model_id: str):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"answers": ANSWERS, "usage": {"input_tokens": 7, "output_tokens": 0}}
    request = DecisionRequest(model_id=model_id, state={"body": "hi"}, questions=QUESTIONS)
    with patch("lmdk.provider.requests.post", return_value=resp) as post:
        result = SystemoneProvider.decide(request)
    return result, post.call_args


class TestRegistry:
    def test_load_provider_finds_shape(self):
        assert load_provider("systemone") is SystemoneProvider

    def test_resolve_model(self):
        name, model_id, cls = resolve_model(f"systemone:english@{ENDPOINT}")
        assert name == "systemone"
        assert model_id == f"english@{ENDPOINT}"
        assert cls is SystemoneProvider


class TestParseModelId:
    def test_splits_model_and_address(self):
        assert SystemoneProvider._parse_model_id(f"english@{ENDPOINT}") == ("english", ENDPOINT)

    def test_missing_address_raises(self):
        with pytest.raises(ProviderError, match="must include an address"):
            SystemoneProvider._parse_model_id("english")


class TestBuildUrl:
    def test_assumes_http_when_no_scheme(self):
        assert SystemoneProvider._build_url(ENDPOINT) == f"http://{ENDPOINT}/v1/systemone"

    def test_preserves_explicit_scheme_and_strips_trailing_slash(self):
        assert (
            SystemoneProvider._build_url("https://host:8080/") == "https://host:8080/v1/systemone"
        )


class TestAuthHeaders:
    def test_no_key_means_no_auth(self, monkeypatch):
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        assert SystemoneProvider._build_auth_headers({}) == {}

    def test_local_key_produces_bearer(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "secret")
        assert SystemoneProvider._build_auth_headers({}) == {"Authorization": "Bearer secret"}


class TestDecide:
    def test_posts_to_address_plus_shape_path(self, monkeypatch):
        monkeypatch.delenv("LOCAL_API_KEY", raising=False)
        result, call = _decide(f"english@{ENDPOINT}")

        assert call.args[0] == f"http://{ENDPOINT}/v1/systemone"
        assert call.kwargs["headers"] == {}
        body = call.kwargs["json"]
        # the address is stripped from the model sent to the server
        assert body["model"] == "english"
        assert body["state"] == {"body": "hi"}
        assert body["questions"]["spam"] == {
            "type": "noul",
            "instructions": "Is this spam?",
            "criteria": {"true": "unsolicited", "false": "legit"},
        }
        assert body["questions"]["urgency"]["type"] == "score"
        assert body["questions"]["dept"]["type"] == "choice"

        # noul / score / choice answers all map back to {label: probability}
        assert result.probabilities == {
            "spam": {"Yes": 0.75, "No": 0.25},
            "urgency": {"low": 0.2, "high": 0.8},
            "dept": {"billing": 0.9, "tech": 0.1},
        }
        assert result.input_tokens == 7
        assert result.output_tokens == 0

    def test_local_api_key_sent_as_bearer(self, monkeypatch):
        monkeypatch.setenv("LOCAL_API_KEY", "s")
        _, call = _decide("english@https://systemone.example.com/")
        assert call.args[0] == "https://systemone.example.com/v1/systemone"
        assert call.kwargs["headers"] == {"Authorization": "Bearer s"}
