"""Implements the provider to use models hosted in Mistral API."""

from collections.abc import Iterator

from lmtk.datatypes import CompletionRequest
from lmtk.provider import Provider, RawResponse

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider(Provider):
    """Provider for models hosted on the Mistral API."""

    api_key_name = "MISTRAL_API_KEY"

    @classmethod
    def _build_auth_headers(cls, api_key: str) -> dict:
        """Return Mistral Bearer-token authentication headers."""
        return {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def _build_messages(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest."""
        api_messages: list[dict] = []
        if request.system_instruction:
            api_messages.append({"role": "system", "content": request.system_instruction})
        api_messages.extend(m.to_dict() for m in request.messages)
        return api_messages

    @classmethod
    def _send_request(cls, request: CompletionRequest, api_key: str) -> RawResponse:
        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_messages(request),
            **(request.generation_kwargs or {}),
        }

        if request.output_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": request.output_schema.model_json_schema(),
                },
            }

        response = cls._make_request(
            MISTRAL_API_URL,
            json=payload,
            headers=cls._build_auth_headers(api_key),
        )

        body = response.json()
        return RawResponse(
            content=body["choices"][0]["message"]["content"],
            input_tokens=body["usage"]["prompt_tokens"],
            output_tokens=body["usage"]["completion_tokens"],
        )

    @classmethod
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_messages(request),
            "stream": True,
            **(request.generation_kwargs or {}),
        }

        response = cls._make_request(
            MISTRAL_API_URL,
            json=payload,
            headers=cls._build_auth_headers(api_key),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            choices = chunk.get("choices", [])
            if choices:
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token
