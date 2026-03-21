"""Tests for lmdk.providers.vertex — VertexProvider."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from lmdk.datatypes import AssistantMessage, CompletionRequest, UserMessage
from lmdk.errors import AuthenticationError
from lmdk.provider import RawResponse
from lmdk.providers.vertex import DEFAULT_LOCATION, VertexProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ID = "my-gcp-project"


def _make_request(**overrides) -> CompletionRequest:
    defaults = {
        "model_id": "gemini-2.5-flash",
        "messages": [UserMessage(content="hi")],
        "system_instruction": None,
        "output_schema": None,
        "generation_kwargs": {},
    }
    defaults.update(overrides)
    return CompletionRequest(**defaults)


def _mock_vertex_response(
    content: str = "hello",
    prompt_tokens: int = 10,
    candidates_tokens: int = 5,
    parts: list[dict] | None = None,
):
    """Build a mock requests.Response that mimics a Vertex generateContent response."""
    if parts is None:
        parts = [{"text": content}]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [{"content": {"parts": parts}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
        },
    }
    return resp


def _mock_stream_response(tokens: list[str]):
    """Build a mock requests.Response that mimics a Vertex SSE stream."""
    lines = []
    for token in tokens:
        chunk = {"candidates": [{"content": {"parts": [{"text": token}]}}]}
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
# _build_auth_headers
# ---------------------------------------------------------------------------


class TestBuildAuthHeaders:
    def test_returns_api_key_header(self):
        headers = VertexProvider._build_auth_headers("my-key")
        assert headers == {"x-goog-api-key": "my-key"}


# ---------------------------------------------------------------------------
# _parse_model_id
# ---------------------------------------------------------------------------


class TestParseModelId:
    def test_without_location(self):
        model, location = VertexProvider._parse_model_id("gemini-2.5-flash")
        assert model == "gemini-2.5-flash"
        assert location == DEFAULT_LOCATION

    def test_with_location(self):
        model, location = VertexProvider._parse_model_id("gemini-2.5-flash@europe-west4")
        assert model == "gemini-2.5-flash"
        assert location == "europe-west4"

    def test_with_different_location(self):
        model, location = VertexProvider._parse_model_id("gemini-2.0-pro@asia-east1")
        assert model == "gemini-2.0-pro"
        assert location == "asia-east1"


# ---------------------------------------------------------------------------
# _resolve_project_id
# ---------------------------------------------------------------------------


class TestResolveProjectId:
    def test_reads_from_env(self):
        with patch.dict("os.environ", {"GCP_PROJECT_ID": "test-project"}):
            assert VertexProvider._resolve_project_id() == "test-project"

    def test_raises_when_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="GCP_PROJECT_ID"):
                VertexProvider._resolve_project_id()


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------


class TestBuildUrl:
    def test_non_streaming_url(self):
        url = VertexProvider._build_url("gemini-2.5-flash", "us-central1", PROJECT_ID, stream=False)
        expected = (
            "https://us-central1-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/us-central1/"
            "publishers/google/models/gemini-2.5-flash:generateContent"
        )
        assert url == expected

    def test_streaming_url(self):
        url = VertexProvider._build_url("gemini-2.5-flash", "us-central1", PROJECT_ID, stream=True)
        expected = (
            "https://us-central1-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/us-central1/"
            "publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        )
        assert url == expected

    def test_custom_location_in_url(self):
        url = VertexProvider._build_url(
            "gemini-2.5-flash", "europe-west4", PROJECT_ID, stream=False
        )
        assert "europe-west4-aiplatform.googleapis.com" in url
        assert "locations/europe-west4" in url


# ---------------------------------------------------------------------------
# _build_contents
# ---------------------------------------------------------------------------


class TestBuildContents:
    def test_user_message(self):
        request = _make_request()
        contents = VertexProvider._build_contents(request)
        assert len(contents) == 1
        assert contents[0] == {"role": "user", "parts": [{"text": "hi"}]}

    def test_assistant_role_mapped_to_model(self):
        request = _make_request(
            messages=[
                UserMessage(content="hi"),
                AssistantMessage(content="hello"),
                UserMessage(content="how are you?"),
            ]
        )
        contents = VertexProvider._build_contents(request)
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"

    def test_message_content_preserved(self):
        request = _make_request(messages=[UserMessage(content="Tell me a joke.")])
        contents = VertexProvider._build_contents(request)
        assert contents[0]["parts"] == [{"text": "Tell me a joke."}]


# ---------------------------------------------------------------------------
# _build_generation_config
# ---------------------------------------------------------------------------


class TestBuildGenerationConfig:
    def test_empty_kwargs(self):
        request = _make_request()
        config = VertexProvider._build_generation_config(request)
        assert config == {}

    def test_maps_openai_style_keys(self):
        request = _make_request(generation_kwargs={"max_tokens": 100, "top_p": 0.9, "top_k": 40})
        config = VertexProvider._build_generation_config(request)
        assert config["maxOutputTokens"] == 100
        assert config["topP"] == 0.9
        assert config["topK"] == 40

    def test_vertex_native_keys_pass_through(self):
        request = _make_request(generation_kwargs={"temperature": 0.7, "candidateCount": 1})
        config = VertexProvider._build_generation_config(request)
        assert config["temperature"] == 0.7
        assert config["candidateCount"] == 1

    def test_structured_output_adds_response_schema(self):
        request = _make_request(output_schema=Person)
        config = VertexProvider._build_generation_config(request)
        assert config["responseMimeType"] == "application/json"
        assert "responseSchema" in config
        schema = config["responseSchema"]
        assert schema["type"] == "OBJECT"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_unmapped_key_passes_through(self):
        request = _make_request(generation_kwargs={"customParam": "value"})
        config = VertexProvider._build_generation_config(request)
        assert config["customParam"] == "value"


# ---------------------------------------------------------------------------
# _pydantic_schema_to_vertex / _convert_schema_node
# ---------------------------------------------------------------------------


class TestPydanticSchemaToVertex:
    def test_simple_model_types_uppercased(self):
        schema = Person.model_json_schema()
        result = VertexProvider._pydantic_schema_to_vertex(schema)
        assert result["type"] == "OBJECT"
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["properties"]["age"]["type"] == "INTEGER"

    def test_required_fields_preserved(self):
        schema = Person.model_json_schema()
        result = VertexProvider._pydantic_schema_to_vertex(schema)
        assert "name" in result["required"]
        assert "age" in result["required"]

    def test_nested_model_with_refs_resolved(self):
        schema = Recipe.model_json_schema()
        result = VertexProvider._pydantic_schema_to_vertex(schema)
        assert result["type"] == "OBJECT"
        items = result["properties"]["ingredients"]["items"]
        assert items["type"] == "OBJECT"
        assert "name" in items["properties"]
        assert "quantity" in items["properties"]
        assert "unit" in items["properties"]

    def test_array_type_uppercased(self):
        schema = Recipe.model_json_schema()
        result = VertexProvider._pydantic_schema_to_vertex(schema)
        assert result["properties"]["ingredients"]["type"] == "ARRAY"

    def test_default_value_preserved(self):
        schema = Ingredient.model_json_schema()
        result = VertexProvider._pydantic_schema_to_vertex(schema)
        assert result["properties"]["unit"]["default"] == ""


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_basic_payload_structure(self):
        request = _make_request()
        payload = VertexProvider._build_payload(request)
        assert "contents" in payload
        assert "generationConfig" in payload
        assert "systemInstruction" not in payload

    def test_system_instruction_included(self):
        request = _make_request(system_instruction="Be a pirate.")
        payload = VertexProvider._build_payload(request)
        assert payload["systemInstruction"] == {"parts": [{"text": "Be a pirate."}]}

    def test_thinking_disabled_by_default(self):
        request = _make_request()
        payload = VertexProvider._build_payload(request)
        thinking = payload["generationConfig"]["thinkingConfig"]
        assert thinking == {"thinkingBudget": 0}

    def test_thinking_config_can_be_overridden(self):
        request = _make_request(generation_kwargs={"thinkingConfig": {"thinkingBudget": 1024}})
        payload = VertexProvider._build_payload(request)
        thinking = payload["generationConfig"]["thinkingConfig"]
        assert thinking == {"thinkingBudget": 1024}


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_extracts_plain_text(self):
        body = {"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]}
        assert VertexProvider._extract_text(body) == "Hello!"

    def test_concatenates_multiple_text_parts(self):
        body = {"candidates": [{"content": {"parts": [{"text": "Hello"}, {"text": " world"}]}}]}
        assert VertexProvider._extract_text(body) == "Hello world"

    def test_filters_out_thought_parts(self):
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": "Answer"},
                        ]
                    }
                }
            ]
        }
        assert VertexProvider._extract_text(body) == "Answer"

    def test_returns_empty_string_when_no_content(self):
        body = {"candidates": [{}]}
        assert VertexProvider._extract_text(body) == ""

    def test_returns_empty_string_when_no_parts(self):
        body = {"candidates": [{"content": {}}]}
        assert VertexProvider._extract_text(body) == ""


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    def test_basic_text_completion(self):
        mock_resp = _mock_vertex_response(content="Hello there!")
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            result = VertexProvider._send_request(_make_request(), api_key="test-key")

        assert isinstance(result, RawResponse)
        assert result.content == "Hello there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

        # Verify the POST call went to the right URL
        call_args = mock_post.call_args
        url = call_args[0][0]
        assert "generateContent" in url
        assert "streamGenerateContent" not in url

    def test_uses_model_and_location_from_model_id(self):
        mock_resp = _mock_vertex_response()
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            request = _make_request(model_id="gemini-2.5-flash@europe-west4")
            VertexProvider._send_request(request, api_key="test-key")

        url = mock_post.call_args[0][0]
        assert "europe-west4-aiplatform.googleapis.com" in url
        assert "models/gemini-2.5-flash:" in url

    def test_generation_kwargs_forwarded(self):
        mock_resp = _mock_vertex_response()
        request = _make_request(generation_kwargs={"temperature": 0.9, "max_tokens": 10})
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            VertexProvider._send_request(request, api_key="test-key")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        gen_config = payload["generationConfig"]
        assert gen_config["temperature"] == 0.9
        assert gen_config["maxOutputTokens"] == 10

    def test_structured_output_payload(self):
        content = '{"name": "Alice", "age": 30}'
        mock_resp = _mock_vertex_response(content=content)
        request = _make_request(output_schema=Person)

        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            result = VertexProvider._send_request(request, api_key="test-key")

        assert result.content == content

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        gen_config = payload["generationConfig"]
        assert gen_config["responseMimeType"] == "application/json"
        assert "responseSchema" in gen_config

    def test_structured_output_nested_payload(self):
        content = json.dumps(
            {
                "ingredients": [
                    {"name": "tomato", "quantity": 5, "unit": "pieces"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"},
                ]
            }
        )
        mock_resp = _mock_vertex_response(content=content)
        request = _make_request(output_schema=Recipe)

        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            result = VertexProvider._send_request(request, api_key="test-key")

        assert result.content == content

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        gen_config = payload["generationConfig"]
        assert gen_config["responseMimeType"] == "application/json"
        schema = gen_config["responseSchema"]
        assert schema["type"] == "OBJECT"

    def test_auth_headers_sent(self):
        mock_resp = _mock_vertex_response()
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            VertexProvider._send_request(_make_request(), api_key="my-api-key")

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"] == {"x-goog-api-key": "my-api-key"}

    def test_filters_thought_parts_from_response(self):
        parts = [
            {"text": "Let me think...", "thought": True},
            {"text": "The answer is 42."},
        ]
        mock_resp = _mock_vertex_response(parts=parts)
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp),
        ):
            result = VertexProvider._send_request(_make_request(), api_key="test-key")

        assert result.content == "The answer is 42."

    def test_missing_usage_metadata_defaults_to_zero(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        }
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=resp),
        ):
            result = VertexProvider._send_request(_make_request(), api_key="test-key")

        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_resp = _mock_stream_response(["Hello", " ", "world"])
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp),
        ):
            tokens = list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["Hello", " ", "world"]

    def test_stream_url_used(self):
        mock_resp = _mock_stream_response(["ok"])
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        url = mock_post.call_args[0][0]
        assert "streamGenerateContent" in url
        assert "alt=sse" in url

    def test_stream_flag_in_request(self):
        mock_resp = _mock_stream_response(["ok"])
        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp) as mock_post,
        ):
            list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs.get("stream") is True

    def test_filters_thought_parts_in_stream(self):
        lines = [
            f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'thinking...', 'thought': True}]}}]})}",
            f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'Answer'}]}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp),
        ):
            tokens = list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["Answer"]

    def test_skips_empty_candidates(self):
        lines = [
            f"data: {json.dumps({'candidates': []})}",
            f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'hi'}]}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp),
        ):
            tokens = list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["hi"]

    def test_skips_empty_lines_and_non_data_lines(self):
        lines = [
            "",
            ": keep-alive",
            f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'hi'}]}}]})}",
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with (
            patch.dict("os.environ", {"GCP_PROJECT_ID": PROJECT_ID}),
            patch("lmdk.provider.requests.post", return_value=mock_resp),
        ):
            tokens = list(VertexProvider._stream_response(_make_request(), api_key="test-key"))

        assert tokens == ["hi"]
