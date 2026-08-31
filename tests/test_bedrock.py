"""Tests for lmdk.providers.bedrock — BedrockProvider."""

import base64
import json
import struct
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_completion_request
from pydantic import BaseModel

from lmdk.errors import TruncatedResponseError
from lmdk.providers.bedrock import (
    _SCHEMA_TOOL,
    BEDROCK_ANTHROPIC_VERSION,
    DEFAULT_REGION,
    BedrockProvider,
)

MODEL = "eu.anthropic.claude-opus-5"
CREDENTIALS = {"AWS_BEARER_TOKEN_BEDROCK": "token-123"}


class Person(BaseModel):
    name: str
    age: int


def _eventstream(chunks: list[dict]) -> bytes:
    """Frame Anthropic stream events as an AWS ``vnd.amazon.eventstream`` body."""
    body = b""
    for chunk in chunks:
        payload = json.dumps(
            {"bytes": base64.b64encode(json.dumps(chunk).encode()).decode()}
        ).encode()
        headers = b":x"  # arbitrary header bytes; the decoder only skips them
        total = 12 + len(headers) + len(payload) + 4
        body += struct.pack(">III", total, len(headers), 0) + headers + payload + b"\0\0\0\0"
    return body


class TestBuildAuthHeaders:
    def test_uses_bearer_token(self):
        assert BedrockProvider._build_auth_headers(CREDENTIALS) == {
            "Authorization": "Bearer token-123"
        }


class TestBuildUrl:
    def test_region_inferred_from_geo_prefix(self):
        url = BedrockProvider._build_url(MODEL, stream=False)
        assert url == f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{MODEL}/invoke"

    def test_explicit_region_suffix_wins(self):
        url = BedrockProvider._build_url(f"{MODEL}@eu-central-1", stream=False)
        assert url.startswith("https://bedrock-runtime.eu-central-1.amazonaws.com/")
        assert url.endswith(f"/model/{MODEL}/invoke")

    def test_unknown_prefix_falls_back_to_default_region(self):
        url = BedrockProvider._build_url("anthropic.claude-opus-5", stream=False)
        assert f"bedrock-runtime.{DEFAULT_REGION}." in url

    def test_stream_action(self):
        url = BedrockProvider._build_url(MODEL, stream=True)
        assert url.endswith("/invoke-with-response-stream")


class TestBuildPayload:
    def test_model_and_stream_move_out_of_body(self):
        payload = BedrockProvider._build_payload(make_completion_request(model_id=MODEL))
        assert "model" not in payload
        assert "stream" not in payload
        assert payload["anthropic_version"] == BEDROCK_ANTHROPIC_VERSION

    def test_no_empty_output_config(self):
        payload = BedrockProvider._build_payload(make_completion_request(model_id=MODEL))
        assert "output_config" not in payload

    def test_thinking_effort_kept_in_output_config(self):
        payload = BedrockProvider._build_payload(
            make_completion_request(model_id=MODEL, thinking_effort="medium")
        )
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "medium"}

    def test_output_schema_becomes_forced_tool_call(self):
        payload = BedrockProvider._build_payload(
            make_completion_request(model_id=MODEL, output_schema=Person)
        )
        assert "output_config" not in payload
        assert payload["tool_choice"] == {"type": "tool", "name": _SCHEMA_TOOL}
        assert payload["tools"][0]["input_schema"]["properties"].keys() == {"name", "age"}


class TestSendRequest:
    def _mock_response(self, content_blocks: list[dict]):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "content": content_blocks,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "output_tokens_details": {"thinking_tokens": 3},
            },
        }
        return resp

    def test_plain_text(self):
        resp = self._mock_response(
            [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "hi"}]
        )
        with patch("lmdk.provider.requests.post", return_value=resp):
            raw = BedrockProvider._send_request(
                make_completion_request(model_id=MODEL), CREDENTIALS
            )
        assert (raw.content, raw.thinking) == ("hi", "hmm")
        assert (raw.input_tokens, raw.output_tokens, raw.thinking_tokens) == (11, 7, 3)

    def test_tool_use_is_serialized_as_json(self):
        resp = self._mock_response(
            [{"type": "tool_use", "name": _SCHEMA_TOOL, "input": {"name": "Ada", "age": 36}}]
        )
        with patch("lmdk.provider.requests.post", return_value=resp):
            raw = BedrockProvider._send_request(
                make_completion_request(model_id=MODEL, output_schema=Person), CREDENTIALS
            )
        assert Person.model_validate_json(raw.content) == Person(name="Ada", age=36)


class TestStreamResponse:
    def test_yields_only_text_deltas(self):
        body = _eventstream(
            [
                {"type": "message_start"},
                {"type": "content_block_delta", "delta": {"type": "thinking_delta", "text": "x"}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "llo"}},
                {"type": "message_stop"},
            ]
        )
        resp = MagicMock()
        resp.status_code = 200
        # Split mid-frame to exercise the buffering path.
        resp.iter_content.return_value = iter([body[:7], body[7:40], body[40:]])

        with patch("lmdk.provider.requests.post", return_value=resp):
            tokens = list(
                BedrockProvider._stream_response(
                    make_completion_request(model_id=MODEL), CREDENTIALS
                )
            )
        assert "".join(tokens) == "hello"


class TestTruncation:
    def test_max_tokens_stop_reason_raises_instead_of_empty_tool_input(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "stop_reason": "max_tokens",
            "content": [{"type": "tool_use", "name": _SCHEMA_TOOL, "input": {}}],
            "usage": {"input_tokens": 11, "output_tokens": 32000},
        }
        with (
            patch("lmdk.provider.requests.post", return_value=resp),
            pytest.raises(TruncatedResponseError, match="max_tokens"),
        ):
            BedrockProvider._send_request(
                make_completion_request(model_id=MODEL, output_schema=Person), CREDENTIALS
            )
