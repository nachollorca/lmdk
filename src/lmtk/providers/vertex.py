"""Implements the provider to use models hosted in GCP Vertex API."""

import os
from collections.abc import Iterator

from lmtk.datatypes import CompletionRequest
from lmtk.errors import AuthenticationError
from lmtk.provider import Provider, RawResponse

DEFAULT_LOCATION = "us-central1"

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

    api_key_name = "VERTEX_API_KEY"

    # ── Auth ──────────────────────────────────────────────────────────────

    @classmethod
    def _build_auth_headers(cls, api_key: str) -> dict:
        """Return Vertex AI API-key authentication headers."""
        return {"x-goog-api-key": api_key}

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
    def _resolve_project_id(cls) -> str:
        """Read the GCP project ID from the environment."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise AuthenticationError(
                status_code=0,
                message="Environment variable GCP_PROJECT_ID not set.",
                provider=cls.__name__,
            )
        return project_id

    @classmethod
    def _build_url(cls, model: str, location: str, project_id: str, *, stream: bool) -> str:
        """Construct the Vertex AI ``generateContent`` endpoint URL."""
        action = "streamGenerateContent" if stream else "generateContent"
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
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
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else msg.role
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

    @classmethod
    def _build_generation_config(cls, request: CompletionRequest) -> dict:
        """Build the ``generationConfig`` object.

        Translates common OpenAI-style parameter names (``max_tokens``,
        ``top_p``, …) to their Vertex AI camelCase equivalents and merges
        structured-output directives when an ``output_schema`` is present.
        """
        config: dict = {}

        for key, value in (request.generation_kwargs or {}).items():
            mapped_key = _GENERATION_KEY_MAP.get(key, key)
            config[mapped_key] = value

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
        """Assemble the full request payload for the Vertex API.

        Thinking is disabled by default (``thinkingBudget: 0``) so that
        all output tokens go to visible content, matching non-thinking
        providers.  Users can opt in to thinking by passing
        ``generation_kwargs={"thinkingConfig": {"thinkingBudget": N}}``.
        """
        generation_config = cls._build_generation_config(request)

        # Allow users to override thinkingConfig via generation_kwargs;
        # default to disabled so maxOutputTokens behaves predictably.
        generation_config.setdefault("thinkingConfig", {"thinkingBudget": 0})

        payload: dict = {
            "contents": cls._build_contents(request),
            "generationConfig": generation_config,
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
    def _send_request(cls, request: CompletionRequest, api_key: str) -> RawResponse:
        model, location = cls._parse_model_id(request.model_id)
        project_id = cls._resolve_project_id()
        url = cls._build_url(model, location, project_id, stream=False)
        payload = cls._build_payload(request)

        response = cls._make_request(
            url,
            json=payload,
            headers=cls._build_auth_headers(api_key),
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
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        model, location = cls._parse_model_id(request.model_id)
        project_id = cls._resolve_project_id()
        url = cls._build_url(model, location, project_id, stream=True)
        payload = cls._build_payload(request)

        response = cls._make_request(
            url,
            json=payload,
            headers=cls._build_auth_headers(api_key),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        yield part["text"]
