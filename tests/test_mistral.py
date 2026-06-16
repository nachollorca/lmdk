"""Tests for lmdk.providers.mistral — MistralProvider."""

import json
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, UserMessage
from lmdk.provider import RawResponse
from lmdk.providers.mistral import MistralProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "mistral-small-2603",
        "prompt": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_chat_response(
    content: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 5
):
    """Build a mock requests.Response that mimics a Mistral chat completion."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics a Mistral SSE stream."""
    lines = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


# ---------------------------------------------------------------------------
# Pydantic models for structured output tests
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""


class Recipe(BaseModel):
    ingredients: list[Ingredient]


# ---------------------------------------------------------------------------
# _build_prompt_payload
# ---------------------------------------------------------------------------


class TestBuildPromptPayload:
    def test_without_system_instruction(self):
        request = _make_request()
        payload = MistralProvider._build_prompt_payload(request)
        assert len(payload) == 1
        assert payload[0] == {"role": "user", "content": "hi"}

    def test_with_system_instruction(self):
        request = _make_request(system_instruction="Be a pirate.")
        payload = MistralProvider._build_prompt_payload(request)
        assert len(payload) == 2
        assert payload[0] == {"role": "system", "content": "Be a pirate."}
        assert payload[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# _build_payload — thinking_effort
# ---------------------------------------------------------------------------


class TestBuildPayloadThinking:
    def test_no_reasoning_effort_when_none(self):
        payload = MistralProvider._build_payload(_make_request())
        assert "reasoning_effort" not in payload

    def test_low_collapses_to_high(self):
        payload = MistralProvider._build_payload(_make_request(thinking_effort="low"))
        assert payload["reasoning_effort"] == "high"

    def test_medium_collapses_to_high(self):
        payload = MistralProvider._build_payload(_make_request(thinking_effort="medium"))
        assert payload["reasoning_effort"] == "high"

    def test_high_maps_to_high(self):
        payload = MistralProvider._build_payload(_make_request(thinking_effort="high"))
        assert payload["reasoning_effort"] == "high"

    def test_explicit_reasoning_effort_overrides_thinking_effort(self):
        # The override is passed through verbatim; lmdk does not validate it.
        # Use a sentinel value to assert passthrough without implying it is a
        # value the Mistral API actually accepts.
        request = _make_request(
            thinking_effort="high",
            generation_kwargs={"reasoning_effort": "sentinel-override"},
        )
        payload = MistralProvider._build_payload(request)
        assert payload["reasoning_effort"] == "sentinel-override"

    def test_drops_sampling_kwargs_when_reasoning(self):
        request = _make_request(
            thinking_effort="high",
            generation_kwargs={"temperature": 0, "top_p": 0.9},
        )
        payload = MistralProvider._build_payload(request)
        assert "temperature" not in payload
        assert "top_p" not in payload

    def test_keeps_sampling_kwargs_when_not_reasoning(self):
        request = _make_request(generation_kwargs={"temperature": 0})
        payload = MistralProvider._build_payload(request)
        assert payload["temperature"] == 0


# ---------------------------------------------------------------------------
# _extract_text — reasoning chunk handling
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_plain_string(self):
        assert MistralProvider._extract_text("hello") == "hello"

    def test_none_returns_empty(self):
        assert MistralProvider._extract_text(None) == ""

    def test_list_keeps_only_text_chunks(self):
        content = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "reasoning..."}]},
            {"type": "text", "text": "the "},
            {"type": "text", "text": "answer"},
        ]
        assert MistralProvider._extract_text(content) == "the answer"


# ---------------------------------------------------------------------------
# _send_request — basic text completion
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_chat_response(content="Hello there!")
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = MistralProvider._send_request(
                _make_request(), credentials={"MISTRAL_API_KEY": "test-key"}
            )

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        # Verify the POST call
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert payload["model"] == "mistral-small-2603"
        assert "response_format" not in payload

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_chat_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 10})
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            MistralProvider._send_request(request, credentials={"MISTRAL_API_KEY": "test-key"})

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 10

    def test_structured_output_payload(self):
        """Verify that response_format is included for structured output."""
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Person)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = MistralProvider._send_request(
                request, credentials={"MISTRAL_API_KEY": "test-key"}
            )

        assert result.content == content

        # Verify response_format was included in payload
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Person"
        assert "schema" in rf["json_schema"]

    def test_structured_output_nested_payload(self):
        """Verify that nested schemas produce the correct response_format."""
        content = json.dumps(
            {
                "ingredients": [
                    {"name": "tomato", "quantity": 5, "unit": "pieces"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"},
                ]
            }
        )
        mock_resp = _mock_chat_response(content=content)
        request = _make_request(output_schema=Recipe)

        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            result = MistralProvider._send_request(
                request, credentials={"MISTRAL_API_KEY": "test-key"}
            )

        assert result.content == content

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Recipe"


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                MistralProvider._stream_response(
                    _make_request(), credentials={"MISTRAL_API_KEY": "test-key"}
                )
            )

        assert tokens == ["Hello", " ", "world"]

    def test_stream_flag_in_payload(self):
        mock_resp = _mock_stream_response(["ok"])
        with patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post:
            list(
                MistralProvider._stream_response(
                    _make_request(), credentials={"MISTRAL_API_KEY": "test-key"}
                )
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["stream"] is True

    def test_skips_empty_lines_and_non_data_lines(self):
        """Lines that are empty or don't start with 'data: ' are ignored."""
        lines = [
            "",
            ": keep-alive",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                MistralProvider._stream_response(
                    _make_request(), credentials={"MISTRAL_API_KEY": "test-key"}
                )
            )

        assert tokens == ["hi"]

    def test_skips_empty_content_deltas(self):
        """Chunks with empty or missing content are silently skipped."""
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ''}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("lmdk.provider.requests.post", return_value=mock_resp):
            tokens = list(
                MistralProvider._stream_response(
                    _make_request(), credentials={"MISTRAL_API_KEY": "test-key"}
                )
            )

        assert tokens == ["ok"]
