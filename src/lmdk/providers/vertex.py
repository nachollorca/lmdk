"""Implements the provider to use models hosted in GCP Vertex API."""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse

DEFAULT_LOCATION = "global"

# Multi-region endpoints use a dedicated hostname that keeps processing within
# the named jurisdiction (see Vertex AI "multi-region endpoints" docs).
_MULTI_REGIONS = ("eu", "us")

# Maps common OpenAI-style generation parameter names to Vertex AI camelCase equivalents.
_GENERATION_KEY_MAP = {
    "max_tokens": "maxOutputTokens",
    "top_p": "topP",
    "top_k": "topK",
    "stop_sequences": "stopSequences",
    # Keys already in Vertex format pass through as-is.
    "temperature": "temperature",
    "candidateCount": "candidateCount",
    "maxOutputTokens": "maxOutputTokens",
    "topP": "topP",
    "topK": "topK",
    "stopSequences": "stopSequences",
    "thinkingConfig": "thinkingConfig",
}


class VertexProvider(Provider):
    """Provider for models hosted on the Google Vertex AI API (Gemini)."""

    required_env = ("VERTEX_API_KEY", "GCP_PROJECT_ID")

    # ── Auth ──────────────────────────────────────────────────────────────

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Vertex AI API-key authentication headers."""
        return {"x-goog-api-key": credentials["VERTEX_API_KEY"]}

    # ── Model / location parsing ──────────────────────────────────────────

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Split ``model_id`` into ``(model, location)``.

        The model string may contain an ``@location`` suffix, e.g.
        ``"gemini-2.5-flash@europe-west4"``.  When omitted the default
        location is ``us-central1``.
        """
        if "@" in model_id:
            model, location = model_id.rsplit("@", 1)
            return model, location
        return model_id, DEFAULT_LOCATION

    @classmethod
    def _build_url(cls, model: str, location: str, project_id: str, *, stream: bool) -> str:
        """Construct the Vertex AI ``generateContent`` endpoint URL.

        Three hostname shapes are supported:

        * ``global`` → ``aiplatform.googleapis.com``
        * multi-region (``eu`` / ``us``) → ``aiplatform.{loc}.rep.googleapis.com``
        * regional (``us-central1``, ``europe-west4``, …) →
          ``{loc}-aiplatform.googleapis.com``
        """
        action = "streamGenerateContent" if stream else "generateContent"
        if location == "global":
            host = "aiplatform.googleapis.com"
        elif location in _MULTI_REGIONS:
            host = f"aiplatform.{location}.rep.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/"
            f"projects/{project_id}/locations/{location}/"
            f"publishers/google/models/{model}:{action}"
        )
        if stream:
            url += "?alt=sse"
        return url

    # ── Request building ──────────────────────────────────────────────────

    @classmethod
    def _build_contents(cls, request: CompletionRequest) -> list[dict]:
        """Convert the message list to Vertex ``contents`` format.

        Vertex uses ``"user"`` and ``"model"`` roles with a ``parts``
        list containing ``{text: ...}`` objects.
        """
        contents: list[dict] = []
        for msg in request.prompt:
            role = "model" if msg.role == "assistant" else msg.role
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

    @classmethod
    def _vertex_thinking_level(cls, request: CompletionRequest) -> str | None:
        """Map ``thinking_effort`` to Vertex ``thinkingLevel`` for Gemini 3 models.

        Gemini 3 cannot fully disable thinking. ``"none"`` maps to the practical
        minimum: ``"minimal"`` on Flash / Flash-Lite models, ``"low"`` on Pro
        models (which reject ``"minimal"``). Returns ``None`` for non-Gemini-3
        models so legacy ``thinkingBudget`` behaviour is unchanged.
        """
        model, _ = cls._parse_model_id(request.model_id)
        if not model.startswith("gemini-3"):
            return None

        if request.thinking_effort == "none":
            return "low" if "pro" in model else "minimal"
        return request.thinking_effort

    @classmethod
    def _build_generation_config(cls, request: CompletionRequest) -> dict:
        """Build the ``generationConfig`` object.

        Translates common OpenAI-style parameter names (``max_tokens``,
        ``top_p``, …) to their Vertex AI camelCase equivalents and merges
        structured-output directives when an ``output_schema`` is present.

        ``thinking_effort`` maps to ``thinkingConfig.thinkingLevel`` on Gemini 3
        models. An explicit ``thinkingConfig`` in ``generation_kwargs``
        overrides the mapped value.
        """
        config: dict = {}

        for key, value in (request.generation_kwargs or {}).items():
            mapped_key = _GENERATION_KEY_MAP.get(key, key)
            config[mapped_key] = value

        thinking_level = cls._vertex_thinking_level(request)
        if thinking_level is not None:
            config.setdefault("thinkingConfig", {"thinkingLevel": thinking_level})

        if request.output_schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = cls._pydantic_schema_to_vertex(
                request.output_schema.model_json_schema()
            )

        return config

    @classmethod
    def _pydantic_schema_to_vertex(cls, schema: dict) -> dict:
        """Convert a Pydantic JSON Schema to the Vertex AI Schema format.

        Vertex AI schemas differ from standard JSON Schema in two ways:
        1. Type values must be uppercased (``STRING``, ``INTEGER``, …).
        2. ``$ref`` / ``$defs`` are not supported; all references must be
           inlined.

        This method recursively resolves ``$ref`` pointers and transforms
        the schema into the Vertex-native format.
        """
        defs = schema.get("$defs", {})
        return cls._convert_schema_node(schema, defs)

    @classmethod
    def _convert_schema_node(cls, node: dict, defs: dict) -> dict:
        """Recursively convert a single JSON Schema node to Vertex format."""
        # Resolve $ref first.
        if "$ref" in node:
            ref_path = node["$ref"]  # e.g. "#/$defs/Ingredient"
            ref_name = ref_path.rsplit("/", 1)[-1]
            return cls._convert_schema_node(defs[ref_name], defs)

        result: dict = {}

        # Type — uppercase it.
        if "type" in node:
            result["type"] = node["type"].upper()

        # Description.
        if "description" in node:
            result["description"] = node["description"]

        # Enum values.
        if "enum" in node:
            result["enum"] = node["enum"]

        # Object properties.
        if "properties" in node:
            result["properties"] = {
                k: cls._convert_schema_node(v, defs) for k, v in node["properties"].items()
            }
        if "required" in node:
            result["required"] = node["required"]

        # Array items.
        if "items" in node:
            result["items"] = cls._convert_schema_node(node["items"], defs)

        # Default value.
        if "default" in node:
            result["default"] = node["default"]

        return result

    @classmethod
    def _build_payload(cls, request: CompletionRequest) -> dict:
        """Assemble the full request payload for the Vertex API."""
        payload: dict = {
            "contents": cls._build_contents(request),
            "generationConfig": cls._build_generation_config(request),
        }

        if request.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}

        return payload

    # ── Response extraction ───────────────────────────────────────────────

    @classmethod
    def _extract_text(cls, body: dict) -> str:
        """Extract the response text from a Vertex API response body.

        Filters out ``thought`` parts produced by thinking models like
        ``gemini-2.5-flash``.  Returns an empty string when the candidate
        has no content (e.g. very low ``maxOutputTokens``).
        """
        candidate = body["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        return "".join(text_parts)

    # ── Provider interface implementation ─────────────────────────────────

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        model, location = cls._parse_model_id(request.model_id)
        project_id = credentials["GCP_PROJECT_ID"]

        response = cls._make_request(
            cls._build_url(model, location, project_id, stream=False),
            json=cls._build_payload(request),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        content = cls._extract_text(body)
        usage = body.get("usageMetadata", {})

        return RawResponse(
            content=content,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        model, location = cls._parse_model_id(request.model_id)
        project_id = credentials["GCP_PROJECT_ID"]

        response = cls._make_request(
            cls._build_url(model, location, project_id, stream=True),
            json=cls._build_payload(request),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        yield part["text"]
