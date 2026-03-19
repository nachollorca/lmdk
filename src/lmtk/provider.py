"""Abstract base class for LLM providers."""

import importlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from lmtk.datatypes import CompletionRequest, CompletionResponse, RawResponse
from lmtk.errors import STATUS_TO_ERROR, AuthenticationError, ProviderError


class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define the following class attributes:
        api_key_name: The environment variable name for the provider's API key.

    The base class handles credential resolution, latency measurement,
    structured-output validation, and ``CompletionResponse`` construction.
    Concrete providers implement ``_send_request``, ``_stream_response``,
    and ``_build_auth_headers``, receiving the API key as a parameter.
    """

    api_key_name: str

    @classmethod
    def get_response(
        cls, request: CompletionRequest, stream: bool
    ) -> CompletionResponse | Iterator[str]:
        """Resolve API credentials and delegate to the provider implementation.

        For non-streaming calls the base class wraps the provider's
        ``_send_request`` with latency measurement, optional structured-output
        parsing, and ``CompletionResponse`` construction.

        See ``lmtk.core.get_response`` for parameter docs and defaults.
        """
        api_key = cls._resolve_api_key()

        if stream:
            return cls._stream_response(request, api_key)

        start = time.perf_counter()
        raw = cls._send_request(request, api_key)
        latency = time.perf_counter() - start

        parsed = None
        if request.output_schema:
            parsed = request.output_schema.model_validate_json(raw.content)

        return CompletionResponse(
            content=raw.content,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            latency=latency,
            parsed=parsed,
        )

    @classmethod
    @abstractmethod
    def _build_auth_headers(cls, api_key: str) -> dict:
        """Return provider-specific authentication headers."""
        ...

    @classmethod
    @abstractmethod
    def _send_request(cls, request: CompletionRequest, api_key: str) -> RawResponse:
        """Make the API call and return the raw content and token counts.

        Implementations should NOT measure latency, validate output schemas,
        or build ``CompletionResponse`` objects -- the base class handles that.
        """
        ...

    @classmethod
    @abstractmethod
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        """Stream chat completion tokens from the provider."""
        ...

    @classmethod
    def _iter_sse_chunks(cls, response: requests.Response) -> Iterator[dict]:
        """Parse a Server-Sent Events stream and yield JSON-decoded chunks.

        Handles the ``data: ...`` framing and the ``[DONE]`` sentinel that
        most LLM APIs use for streaming responses.
        """
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            yield json.loads(data)

    @classmethod
    def _make_request(
        cls, url: str, *, json: dict, headers: dict | None = None, **kwargs
    ) -> requests.Response:
        """Send a POST request and raise on non-200 responses.

        This is an ergonomic wrapper around ``requests.post`` that maps
        common HTTP status codes to specific ``ProviderError`` subclasses
        so callers get uniform error handling across providers.

        Returns the :class:`requests.Response` on success.
        """
        response = requests.post(url, json=json, headers=headers, **kwargs)

        if response.status_code != 200:
            error_cls = STATUS_TO_ERROR.get(response.status_code, ProviderError)
            raise error_cls(
                status_code=response.status_code,
                message=f"{cls.__name__}: HTTP {response.status_code} - {response.reason}",
                provider=cls.__name__,
                body=response.text,
            )

        return response

    @classmethod
    def _resolve_api_key(cls) -> str:
        """Read the API key from the environment.

        Raises :class:`AuthenticationError` when the variable is unset or empty.
        """
        api_key = os.getenv(cls.api_key_name)
        if not api_key:
            raise AuthenticationError(
                status_code=0,
                message=f"Environment variable {cls.api_key_name} not set.",
                provider=cls.__name__,
            )
        return api_key


def load_provider(name: str) -> type[Provider]:
    """Gets the appropriate Provider class for the given provider name.

    Imports ``<Name>Provider`` from ``lmtk.providers.<name>``
    (e.g. ``"mistral"`` -> ``MistralProvider``).

    Raises:
        ImportError: If no module matches *name*.
        AttributeError: If the module does not contain the expected class.
    """
    module = importlib.import_module(f"lmtk.providers.{name}")
    class_name = f"{name.capitalize()}Provider"
    return getattr(module, class_name)
