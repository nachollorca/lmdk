"""Implements the provider to use Anthropic models hosted in AWS Bedrock.

Bedrock exposes Claude through ``InvokeModel``, which is the Anthropic Messages
API with a different envelope: bearer-token auth, the model in the URL instead
of the body, an ``anthropic_version`` field, and a binary event stream instead
of SSE. Everything else (payload shape, thinking, response blocks) is identical,
so this provider subclasses :class:`AnthropicProvider` and only patches the
differences.

Two Bedrock-specific deviations:

* Structured output is expressed as a forced tool call. Bedrock rejects
  Anthropic's native ``output_config.format``.
* Streaming responses use ``application/vnd.amazon.eventstream`` framing.
"""

import base64
import json
import struct
from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import RawResponse
from lmdk.providers.anthropic import AnthropicProvider

BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Name of the synthetic tool used to emulate structured output.
_SCHEMA_TOOL = "emit_structured_output"

# Default region per geo prefix of the model ID (``eu.anthropic.claude-opus-5``).
# A geo inference ID must be called from a region inside that geography.
_GEO_REGIONS = {"eu": "eu-west-1", "us": "us-east-1", "au": "ap-southeast-2"}
DEFAULT_REGION = "us-east-1"


class BedrockProvider(AnthropicProvider):
    """Provider for Anthropic models hosted on the AWS Bedrock runtime API."""

    required_env = "AWS_BEARER_TOKEN_BEDROCK"

    # ── Auth ──────────────────────────────────────────────────────────────

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Bedrock API-key (bearer token) authentication headers."""
        return {"Authorization": f"Bearer {credentials['AWS_BEARER_TOKEN_BEDROCK']}"}

    # ── Model / region parsing ────────────────────────────────────────────

    @classmethod
    def _build_url(cls, model_id: str, *, stream: bool) -> str:
        """Construct the ``invoke`` endpoint URL for *model_id*.

        The model string may carry an ``@region`` suffix, e.g.
        ``"eu.anthropic.claude-opus-5@eu-central-1"``. Without it the region is
        inferred from the geo prefix of the model ID (``eu``/``us``/``au``).
        """
        if "@" in model_id:
            model, region = model_id.rsplit("@", 1)
        else:
            model = model_id
            region = _GEO_REGIONS.get(model.split(".", 1)[0], DEFAULT_REGION)
        action = "invoke-with-response-stream" if stream else "invoke"
        return f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/{action}"

    # ── Request building ──────────────────────────────────────────────────

    @classmethod
    def _build_payload(cls, request: CompletionRequest, stream: bool = False) -> dict:
        """Adapt the Anthropic payload to the Bedrock ``InvokeModel`` body.

        The model and the stream flag move to the URL, ``anthropic_version`` is
        required, and ``output_config.format`` is rewritten as a forced tool
        call because Bedrock does not accept it.
        """
        payload = super()._build_payload(request, stream=stream)
        payload.pop("model", None)
        payload.pop("stream", None)
        payload["anthropic_version"] = BEDROCK_ANTHROPIC_VERSION

        output_config = payload.get("output_config", {})
        schema_format = output_config.pop("format", None)
        if schema_format:
            payload["tools"] = [
                {
                    "name": _SCHEMA_TOOL,
                    "description": "Return the answer as structured output.",
                    "input_schema": schema_format["schema"],
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": _SCHEMA_TOOL}
        if not output_config:
            payload.pop("output_config", None)

        return payload

    # ── Response extraction ───────────────────────────────────────────────

    @classmethod
    def _extract_text(cls, body: dict) -> str:
        """Extract the answer, unwrapping the structured-output tool call."""
        for block in body.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == _SCHEMA_TOOL:
                return json.dumps(block["input"])
        return super()._extract_text(body)

    # ── Provider interface implementation ─────────────────────────────────

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        response = cls._make_request(
            cls._build_url(request.model_id, stream=False),
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        usage = body.get("usage", {})
        return RawResponse(
            content=cls._extract_text(body),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            thinking=cls._extract_thinking(body),
            thinking_tokens=usage.get("output_tokens_details", {}).get("thinking_tokens", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        response = cls._make_request(
            cls._build_url(request.model_id, stream=True),
            json=cls._build_payload(request, stream=True),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_eventstream_chunks(response):
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]

    @classmethod
    def _iter_eventstream_chunks(cls, response) -> Iterator[dict]:
        """Decode an ``application/vnd.amazon.eventstream`` body into JSON chunks.

        Each message is framed as ``total_length | headers_length | prelude_crc
        | headers | payload | message_crc`` (lengths are big-endian uint32).
        The payload is ``{"bytes": <base64 Anthropic stream event>}``.
        """
        # ponytail: CRCs are not verified; requests already runs over TLS, and a
        # corrupt frame surfaces as a JSON decode error. Add zlib.crc32 checks if
        # silent truncation ever shows up.
        buffer = b""
        for data in response.iter_content(chunk_size=None):
            buffer += data
            while len(buffer) >= 12:
                total_length, headers_length = struct.unpack(">II", buffer[:8])
                if len(buffer) < total_length:
                    break
                payload = buffer[12 + headers_length : total_length - 4]
                buffer = buffer[total_length:]
                event = json.loads(payload)
                if "bytes" in event:
                    yield json.loads(base64.b64decode(event["bytes"]))
