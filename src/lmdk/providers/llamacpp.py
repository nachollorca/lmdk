"""Implements the provider to use models hosted in llamacpp instance."""

from collections.abc import Iterator

from lmdk.datatypes import CompletionRequest
from lmdk.provider import Provider, RawResponse


class LlamacppProvider(Provider):
    """Provider for models hosted on a llamacpp server.

    Requires LLAMACPP_URL and LLAMACPP_PORT environment variables.
    The URL for the request will be built as:
    http://{LLAMACPP_URL}:{LLAMACPP_PORT}/v1/chat/completions
    """

    required_env = ("LLAMACPP_URL", "LLAMACPP_PORT")

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Llamacpp typically doesn't require auth, but we return empty headers."""
        return {}

    @classmethod
    def _get_base_url(cls, credentials: dict[str, str]) -> str:
        """Build the full API URL from credentials."""
        url = credentials["LLAMACPP_URL"].rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        port = credentials["LLAMACPP_PORT"]
        return f"{url}:{port}/v1/chat/completions"

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
        """Build the full request payload for the llamacpp API."""
        if request.output_schema:
            raise NotImplementedError("Structured output is not implemented for LlamacppProvider.")

        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_prompt_payload(request),
            "stream": stream,
            **(request.generation_kwargs or {}),
        }
        return payload

    @classmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        url = cls._get_base_url(credentials)
        response = cls._make_request(
            url,
            json=cls._build_payload(request, stream=False),
            headers=cls._build_auth_headers(credentials),
        )

        body = response.json()
        return RawResponse(
            content=body["choices"][0]["message"]["content"],
            input_tokens=body.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=body.get("usage", {}).get("completion_tokens", 0),
        )

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        url = cls._get_base_url(credentials)
        response = cls._make_request(
            url,
            json=cls._build_payload(request, stream=True),
            headers=cls._build_auth_headers(credentials),
            stream=True,
        )

        for chunk in cls._iter_sse_chunks(response):
            choices = chunk.get("choices", [])
            if choices:
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token
