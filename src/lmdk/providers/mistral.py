"""Implements the provider to use models hosted in Mistral API."""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse
from lmdk.providers._schema import prepare_schema

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Sampling controls Mistral rejects when ``reasoning_effort`` is enabled.
_REASONING_INCOMPATIBLE_KWARGS = ("temperature", "top_p")


class MistralProvider(Provider):
    """Provider for models hosted on the Mistral API."""

    required_env = "MISTRAL_API_KEY"

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return Mistral Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {credentials['MISTRAL_API_KEY']}"}

    @classmethod
    def _build_prompt_payload(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest."""
        api_messages: list[dict] = []
        if request.system_instruction:
            api_messages.append({"role": "system", "content": request.system_instruction})
        api_messages.extend(m.to_dict() for m in request.prompt)
        return api_messages

    @classmethod
    def _build_payload(cls, request: CompletionRequest, stream: bool = False) -> dict:
        """Build the full request payload for the Mistral API."""
        generation_kwargs = dict(request.generation_kwargs or {})
        # Mistral adjustable reasoning models only accept reasoning_effort
        # "none" or "high" (low/medium return 400). Map any non-"none" lmdk
        # thinking_effort to "high" so cross-provider effort levels still
        # enable reasoning. Callers can override via generation_kwargs.
        if request.thinking_effort != "none":
            # Mistral reasoning models reject sampling controls like
            # temperature/top_p (lmdk defaults temperature to 0). Drop them so
            # the request goes through cleanly.
            for key in _REASONING_INCOMPATIBLE_KWARGS:
                generation_kwargs.pop(key, None)
            # Caller-provided reasoning_effort wins; otherwise default to "high".
            generation_kwargs.setdefault("reasoning_effort", "high")

        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_prompt_payload(request),
            "stream": stream,
            **generation_kwargs,
        }

        if request.output_schema and not stream:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": prepare_schema(request.output_schema.model_json_schema()),
                    "strict": True,
                },
            }
        return payload

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return ``reasoning_effort`` from the outbound Mistral payload."""
        effort = cls._build_payload(request, stream=False).get("reasoning_effort")
        return effort if effort is not None else "none"

    @staticmethod
    def _extract_text(content: str | list | None) -> str:
        """Extract the answer text from a Mistral message ``content``.

        With ``reasoning_effort`` enabled, ``content`` is a list of chunks
        (``thinking`` + ``text``) instead of a plain string. Only the text
        chunks form the final answer.
        """
        if isinstance(content, list):
            return "".join(
                chunk.get("text", "") for chunk in content if chunk.get("type") == "text"
            )
        return content or ""

    @staticmethod
    def _extract_thinking(content: str | list | None) -> str | None:
        """Extract thinking/reasoning text from a Mistral message ``content``.

        With ``reasoning_effort`` enabled, thinking chunks look like
        ``{"type": "thinking", "thinking": [{"type": "text", "text": "..."}]}``.
        """
        if not isinstance(content, list):
            return None
        parts = []
        for chunk in content:
            if chunk.get("type") != "thinking":
                continue
            for sub in chunk.get("thinking", []):
                if sub.get("type") == "text":
                    parts.append(sub.get("text", ""))
        joined = "".join(parts)
        return joined if joined else None

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        response = cls._make_request(
            MISTRAL_API_URL,
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        message_content = body["choices"][0]["message"]["content"]
        usage = body["usage"]
        # Mistral surfaces thinking text in message chunks but does not report a
        # separate token breakdown in usage (no reasoning_tokens or
        # completion_tokens_details). completion_tokens includes thinking + answer.
        # thinking_tokens stays 0 per the contract when no breakdown is available.
        return RawResponse(
            content=cls._extract_text(message_content),
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            thinking=cls._extract_thinking(message_content),
            thinking_tokens=usage.get("reasoning_tokens", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        response = cls._make_request(
            MISTRAL_API_URL,
            json=cls._build_payload(request, stream=True),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            choices = chunk.get("choices", [])
            if choices:
                # With reasoning enabled, delta.content is a list of chunks
                # during the thinking phase and a plain string afterwards.
                # Only surface the text (answer) tokens.
                token = cls._extract_text(choices[0].get("delta", {}).get("content", ""))
                if token:
                    yield token
