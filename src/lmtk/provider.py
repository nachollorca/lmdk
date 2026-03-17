"""Abstract base class for LLM providers."""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from lmtk.datatypes import CompletionRequest, CompletionResponse
from lmtk.errors import STATUS_TO_ERROR, AuthenticationError, ProviderError


class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define the following class attributes:
        api_key_name: The environment variable name for the provider's API key.

    The base class handles credential resolution and validation.
    Concrete providers only implement ``_get_response`` and ``_stream``,
    receiving the API key as a parameter.
    """

    api_key_name: str

    @classmethod
    def _check_response(cls, response: requests.Response) -> None:
        """Raise a ``ProviderError`` subclass for non-200 responses.

        Maps common HTTP status codes to specific exception types so that
        callers can handle authentication, rate-limiting, and other errors
        uniformly across providers.
        """
        if response.status_code == 200:
            return

        error_cls = STATUS_TO_ERROR.get(response.status_code, ProviderError)
        raise error_cls(
            status_code=response.status_code,
            message=f"{cls.__name__}: HTTP {response.status_code} - {response.reason}",
            provider=cls.__name__,
            body=response.text,
        )

    @classmethod
    def get_response(
        cls, request: CompletionRequest, stream: bool
    ) -> CompletionResponse | Iterator[str]:
        """Resolve API credentials and delegate to the provider implementation.

        See ``lmtk.core.get_response`` for parameter docs and defaults.
        """
        api_key = os.getenv(cls.api_key_name)
        if not api_key:
            raise AuthenticationError(
                status_code=0,
                message=f"Environment variable {cls.api_key_name} not set.",
                provider=cls.__name__,
            )

        if stream:
            return cls._stream(request, api_key)
        return cls._get_response(request, api_key)

    @classmethod
    @abstractmethod
    def _get_response(cls, request: CompletionRequest, api_key: str) -> CompletionResponse:
        """Provider-specific response generation."""
        ...

    @classmethod
    @abstractmethod
    def _stream(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        """Stream chat completion tokens from the provider."""
        ...
