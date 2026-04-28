"""Abstract base class for LLM providers."""

import importlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from lmdk.datatypes import CompletionRequest, CompletionResponse, RawResponse
from lmdk.errors import STATUS_TO_ERROR, AuthenticationError, ProviderError


class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define class attribute ``required_env``:
    A string or tuple of environment variable names the provider needs. For example:
    ``"MISTRAL_API_KEY"`` or ``("VERTEX_API_KEY", "GCP_PROJECT_ID")``).

    The main method in the base class (``complete``) handles:
        - credential resolution (``_resolve_credentials``)
        - latency measurement
        - structured-output validation
        - ``CompletionResponse`` construction

    It also provides utilities to be used in the concrete providers:
        - ``_iter_sse_chunks`` to receive streamed responses
        - ``_make_request`` to make the POST HTTP request and handle errors

    Concrete providers implement:
        - ``_build_auth_headers`` to build the HTTP credentials required by the Provider
        - ``_send_request`` to build the HTTP body,  call ``Provider._make_request`` and
            optionally parse structured output
        - ``_stream_response`` to build the HTTP body and parse the streamed tokens
    """

    required_env: str | tuple[str, ...]  # one or multiple required environmental variables

    @classmethod
    def complete(
        cls, request: CompletionRequest, stream: bool
    ) -> CompletionResponse | Iterator[str]:
        """Resolve API credentials and delegate to the provider implementation.

        For non-streaming calls the base class wraps the provider's
        ``_send_request`` with latency measurement, optional structured-output
        parsing, and ``CompletionResponse`` construction.

        See ``lmdk.core.complete`` for parameter docs and defaults.
        """
        credentials = cls._resolve_credentials()

        if stream:
            return cls._stream_response(request, credentials)

        start = time.perf_counter()
        raw = cls._send_request(request, credentials)
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
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return provider-specific authentication headers."""
        ...

    @classmethod
    @abstractmethod
    def _send_request(cls, request: CompletionRequest, credentials: dict[str, str]) -> RawResponse:
        """Make the API call and return the raw content and token counts.

        Implementations should NOT measure latency, validate output schemas,
        or build ``CompletionResponse`` objects -- the base class handles that.
        """
        ...

    @classmethod
    @abstractmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
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
    def _resolve_credentials(cls) -> dict[str, str]:
        """Read all required environment variables.

        Returns a ``{var_name: value}`` dict for every name listed in
        ``required_env``.

        Raises :class:`AuthenticationError` when any variable is unset or empty.
        """
        # Normalize to a tuple if a single string is provided
        vars_to_resolve = (
            (cls.required_env,) if isinstance(cls.required_env, str) else cls.required_env
        )

        credentials: dict[str, str] = {}
        for var in vars_to_resolve:
            value = os.getenv(var)
            if not value:
                raise AuthenticationError(
                    status_code=0,
                    message=f"Environment variable {var} not set.",
                    provider=cls.__name__,
                )
            credentials[var] = value
        return credentials


def load_provider(name: str) -> type[Provider]:
    """Gets the appropriate Provider class for the given provider name.

    Imports ``<Name>Provider`` from ``lmdk.providers.<name>``
    (e.g. ``"mistral"`` -> ``MistralProvider``).

    Raises:
        ImportError: If no module matches *name*.
        AttributeError: If the module does not contain the expected class.
    """
    module = importlib.import_module(f"lmdk.providers.{name}")
    class_name = f"{name.capitalize()}Provider"
    return getattr(module, class_name)
