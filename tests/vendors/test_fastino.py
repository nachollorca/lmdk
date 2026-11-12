"""Tests for lmdk.providers.vendors.fastino."""

from unittest.mock import MagicMock, patch

import pytest

from lmdk.datatypes import DecisionRequest, Question
from lmdk.errors import ProviderError, ServiceUnavailableError
from lmdk.providers.vendors.fastino import FastinoProvider

QUESTIONS = {
    "spam": Question("Is this spam?", {"Yes": "unsolicited", "No": "legit"}),
    "urgency": Question("How urgent?", {"low": "can wait", "high": "now"}, is_ordered=True),
    "dept": Question("Which team?", {"billing": "money", "tech": "bugs"}),
}

ANSWERS = {
    "spam": {"type": "noul", "noul": 0.75, "confidence": 0.5},
    "urgency": {
        "type": "score",
        "score": 1,
        "expected_level": 0.8,
        "probabilities": {"0": 0.2, "1": 0.8},
        "legend": {"0": "can wait", "1": "now"},
    },
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}},
}


def _decide(model_id: str):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "model": "glide",
        "answers": ANSWERS,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    request = DecisionRequest(model_id=model_id, state={"body": "hi"}, questions=QUESTIONS)
    with patch("lmdk.provider.requests.post", return_value=resp) as post:
        result = FastinoProvider.decide(request)
    return result, post.call_args


def test_fastino_request_and_response(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    result, call = _decide("glide")

    assert call.args[0] == "https://api.fastino.ai/v1/systemone"
    assert call.kwargs["headers"] == {"X-API-Key": "fast_sk_test"}
    assert call.kwargs["timeout"] == 300.0
    body = call.kwargs["json"]
    assert body["model"] == "fastino/GLiDE"
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
            "criteria": ["can wait", "now"],
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
    assert result.output_tokens == 3


def test_deployed_model_id_passes_through(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    _, call = _decide("my-finetuned-glide")
    assert call.kwargs["json"]["model"] == "my-finetuned-glide"


def _mock_http_response(status_code: int, reason: str = "Error") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = reason
    resp.text = ""
    resp.headers = {}
    return resp


def test_425_model_warming_retries_and_succeeds(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    warming = _mock_http_response(425, "model_warming")
    ok = _mock_http_response(200)
    ok.json.return_value = {"answers": ANSWERS, "usage": {"input_tokens": 1, "output_tokens": 1}}
    request = DecisionRequest(model_id="glide", state="hi", questions=QUESTIONS)

    with (
        patch("lmdk.provider.requests.post", side_effect=[warming, ok]) as post,
        patch("lmdk.provider.time.sleep"),
    ):
        result = FastinoProvider.decide(request)

    assert post.call_count == 2
    assert result.input_tokens == 1


def test_503_retries_then_raises_service_unavailable(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    request = DecisionRequest(model_id="glide", state="hi", questions=QUESTIONS)

    with (
        patch("lmdk.provider.requests.post", return_value=_mock_http_response(503)) as post,
        patch("lmdk.provider.time.sleep"),
        pytest.raises(ServiceUnavailableError) as exc_info,
    ):
        FastinoProvider.decide(request)

    assert exc_info.value.status_code == 503
    assert post.call_count == FastinoProvider.max_retries + 1


def test_425_not_retried_by_typesafe_defaults():
    # the base retry set stays {429, 529}; only Fastino adds 425/503
    assert FastinoProvider.retry_statuses == frozenset({425, 429, 503})
    from lmdk.providers.vendors.typesafe import TypesafeProvider

    assert TypesafeProvider.retry_statuses == frozenset({429, 529})


def test_422_raises_provider_error_without_retry(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "fast_sk_test")
    request = DecisionRequest(model_id="glide", state="hi", questions=QUESTIONS)

    with (
        patch(
            "lmdk.provider.requests.post", return_value=_mock_http_response(422, "Unprocessable")
        ) as post,
        patch("lmdk.provider.time.sleep") as sleep,
        pytest.raises(ProviderError) as exc_info,
    ):
        FastinoProvider.decide(request)

    assert exc_info.value.status_code == 422
    assert post.call_count == 1
    assert sleep.call_count == 0


# ---------------------------------------------------------------------------
# @<address> override (self-hosted GLiDE server)
# ---------------------------------------------------------------------------


def test_decide_fastino_glide_at_address(monkeypatch):
    """``decide("fastino:glide@192.168.10.51:8100", ...)`` hits the address, needs no
    FASTINO_API_KEY, and sends LOCAL_API_KEY in Fastino's ``X-API-Key`` format."""
    from lmdk import decide

    monkeypatch.delenv("FASTINO_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_API_KEY", "local-secret")

    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "model": "fastino/GLiDE",
        "answers": ANSWERS,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    with patch("lmdk.provider.requests.post", return_value=resp) as post:
        result = decide(
            model="fastino:glide@192.168.10.51:8100",
            state={"body": "hi"},
            questions=QUESTIONS,
        )

    assert post.call_args.args[0] == "http://192.168.10.51:8100/v1/systemone"
    # LOCAL_API_KEY in the vendor's header format; the vendor key is never sent
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "local-secret"}
    body = post.call_args.kwargs["json"]
    assert body["model"] == "fastino/GLiDE"  # glide alias still applies
    assert body["questions"]["urgency"]["criteria"] == ["can wait", "now"]  # GLiDE quirk
    # noul, choice and score answers all come back
    assert result.probabilities == {
        "spam": {"Yes": 0.75, "No": 0.25},
        "urgency": {"low": 0.2, "high": 0.8},
        "dept": {"billing": 0.9, "tech": 0.1},
    }


def test_vendor_key_never_sent_to_address(monkeypatch):
    monkeypatch.setenv("FASTINO_API_KEY", "vendor-secret")
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    request = DecisionRequest(model_id="glide@10.0.0.5:9000", state="hi", questions=QUESTIONS)
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"answers": ANSWERS, "usage": {"input_tokens": 1, "output_tokens": 1}}
    with patch("lmdk.provider.requests.post", return_value=resp) as post:
        FastinoProvider.decide(request)

    assert post.call_args.args[0] == "http://10.0.0.5:9000/v1/systemone"
    assert post.call_args.kwargs["headers"] == {}
    assert "vendor-secret" not in str(post.call_args.kwargs["headers"])
