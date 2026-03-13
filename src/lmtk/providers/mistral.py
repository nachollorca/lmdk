"""Implements the provider to use models hosted in Mistral API."""

import json
import time
import urllib.request
from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import Message, ModelResponse
from lmtk.provider import Provider

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider(Provider):
    """Provider for models hosted on the Mistral API."""

    model_ids = [
        "mistral-large-latest",
        "mistral-small-latest",
        "devstral-latest",
        "codestral-latest",
    ]
    api_key_name = "MISTRAL_API_KEY"

    @classmethod
    def _get_response(
        cls,
        model_id: str,
        messages: list[Message],
        api_key: str,
        system_instruction: str | None,
        output_schema: type[BaseModel] | None,
        generation_kwargs: dict,
    ) -> ModelResponse:
        """Send a chat completion request to the Mistral API."""
        api_messages: list[dict] = []
        if system_instruction:
            api_messages.append({"role": "system", "content": system_instruction})
        api_messages.extend(m.to_dict() for m in messages)

        payload = {
            "model": model_id,
            "messages": api_messages,
            **(generation_kwargs or {}),
        }

        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            MISTRAL_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        start = time.perf_counter()
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read())
        latency = time.perf_counter() - start

        return ModelResponse(
            content=body["choices"][0]["message"]["content"],
            input_tokens=body["usage"]["prompt_tokens"],
            output_tokens=body["usage"]["completion_tokens"],
            latency=latency,
        )

    @classmethod
    def _stream(
        cls,
        model_id: str,
        messages: list[Message],
        api_key: str,
        system_instruction: str | None,
        output_schema: type[BaseModel] | None,
        generation_kwargs: dict,
    ) -> Iterator[str]:
        """Not yet implemented."""
        raise NotImplementedError
