"""Abstract base class for LLM providers."""

import dataclasses
import datetime
import importlib
import importlib.util
import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from email.utils import parsedate_to_datetime

import requests

from lmdk.datatypes import (
    CompletionRequest,
    CompletionResponse,
    DecisionRequest,
    DecisionResponse,
    RawResponse,
)
from lmdk.errors import STATUS_TO_ERROR, AuthenticationError, ProviderError, TruncatedResponseError

_TRUNCATED_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}
_PROBABILITY_DECIMALS = 5  # label probabilities are rounded to this many decimals


def _parse_retry_after(retry_after: str | None) -> float | None:
    """Parse HTTP Retry-After header value as seconds to sleep.

    Can handle both a numeric string and an HTTP GMT/naive date string.
    """
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        try:
            dt = parsedate_to_datetime(retry_after)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.UTC)
            now = datetime.datetime.now(datetime.UTC)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None


def _extract_error_message(data: dict | None) -> str | None:
    """Find a human-readable message in a JSON error body of unknown shape."""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, dict):
        return err.get("message")
    if isinstance(err, str):
        return err
    if "message" in data:
        return data["message"]
    if "detail" in data:
        detail = data["detail"]
        return detail if isinstance(detail, str) else str(detail)
    return None


def _calculate_backoff(
    attempt: int, initial_delay: float, backoff_factor: float, max_delay: float
) -> float:
    """Calculate exponential backoff with full randomized jitter."""
    calculated_delay = min(max_delay, initial_delay * (backoff_factor**attempt))
    return random.uniform(0, calculated_delay)


class Provider(ABC):
    """Interface that all model providers must implement.

    Subclasses may define class attribute ``required_env``:
    A string or tuple of environment variable names the provider needs. For example:
    ``"MISTRAL_API_KEY"`` or ``("VERTEX_API_KEY", "GCP_PROJECT_ID")``). Providers
    that need no credentials (e.g. a shape addressed via ``@<address>``) can leave
    it as the default empty tuple.

    The main methods in the base class (``complete`` and ``decide``) handle:
        - credential resolution (``_resolve_credentials``)
        - latency measurement
        - structured-output validation
        - ``CompletionResponse`` construction

    It also provides utilities to be used in the concrete providers:
        - ``_iter_sse_chunks`` to receive streamed responses
        - ``_make_request`` to make the POST HTTP request and handle errors

    Concrete providers implement ``_build_auth_headers`` to build the HTTP credentials
    required by the Provider, plus the hooks for whatever they offer:
        - completion (decoders): ``_send_completion_request`` to build the HTTP body, call
            ``Provider._make_request`` and optionally parse structured output, and
            ``_stream_response`` to build the HTTP body and parse the streamed tokens
        - decision (encoders): ``_send_decision_request`` to build the HTTP body, call
            ``Provider._make_request`` and parse the label probabilities
    Hooks a provider does not implement raise ``NotImplementedError``.

    Location handling: the ``@`` suffix of a model identifier is either an
    address (``@<host>[:<port>]``, see :func:`_address_override`) or a region
    (bedrock, vertex). On a user-supplied address the provider's own key is
    neither required nor sent.
    """

    # zero, one, or multiple required environment variables (empty = none required)
    required_env: str | tuple[str, ...] = ()

    # retry configuration for `_make_request`
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    # HTTP statuses retried with backoff (rate limited, overloaded, ...)
    retry_statuses: frozenset[int] = frozenset({429, 529})
    # read timeout in seconds passed to requests when set (None = wait forever)
    request_timeout: float | None = None

    @classmethod
    def complete(
        cls, request: CompletionRequest, stream: bool
    ) -> CompletionResponse | Iterator[str]:
        """Resolve API credentials and delegate to the provider implementation.

        For non-streaming calls the base class wraps the provider's
        ``_send_completion_request`` with latency measurement, optional structured-output
        parsing, and ``CompletionResponse`` construction.

        See ``lmdk.completion.complete`` for parameter docs and defaults.
        """
        credentials = cls._resolve_credentials(request)

        if stream:
            return cls._stream_response(request, credentials)

        start = time.perf_counter()
        raw = cls._send_completion_request(request, credentials)
        latency = time.perf_counter() - start

        parsed = None
        if request.output_schema:
            reason = (raw.finish_reason or "").lower()
            if reason in _TRUNCATED_FINISH_REASONS:
                raise TruncatedResponseError(
                    f"Response truncated: model hit ``max_tokens`` limit with ``output_schema`` "
                    f"({raw.output_tokens} tokens generated). "
                    "Therefore, the output JSON is malformed."
                )
            parsed = request.output_schema.model_validate_json(raw.content)

        return CompletionResponse(
            content=raw.content,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            thinking=raw.thinking,
            thinking_tokens=raw.thinking_tokens,
            finish_reason=raw.finish_reason,
            latency=latency,
            parsed=parsed,
        )

    @classmethod
    def decide(cls, request: DecisionRequest) -> DecisionResponse:
        """Resolve API credentials, delegate to ``_send_decision_request`` and measure latency.

        Label probabilities are rounded to ``_PROBABILITY_DECIMALS`` decimals for every
        provider, so callers never see raw floating-point noise.

        See ``lmdk.decision.decide`` for parameter docs.
        """
        credentials = cls._resolve_credentials(request)
        start = time.perf_counter()
        response = cls._send_decision_request(request, credentials)
        probabilities = {
            name: {label: round(p, _PROBABILITY_DECIMALS) for label, p in probs.items()}
            for name, probs in response.probabilities.items()
        }
        return dataclasses.replace(
            response, probabilities=probabilities, latency=time.perf_counter() - start
        )

    @classmethod
    def request_reasoning_level(cls, request: CompletionRequest) -> str:
        """Return the reasoning level string sent to the provider for telemetry.

        Maps to ``gen_ai.request.reasoning.level``. Returns the provider-specific
        level when the outbound request includes a reasoning or thinking control,
        otherwise ``"none"``.
        """
        return "none"

    @classmethod
    @abstractmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return provider-specific authentication headers."""
        ...

    @classmethod
    def _send_completion_request(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> RawResponse:
        """Make the API call and return the raw content and token counts.

        Implementations should NOT measure latency, validate output schemas,
        or build ``CompletionResponse`` objects -- the base class handles that.
        """
        raise NotImplementedError(f"{cls.__name__} does not support complete().")

    @classmethod
    def _stream_response(
        cls, request: CompletionRequest, credentials: dict[str, str]
    ) -> Iterator[str]:
        """Stream chat completion tokens from the provider."""
        raise NotImplementedError(f"{cls.__name__} does not support streaming.")

    @classmethod
    def _send_decision_request(
        cls, request: DecisionRequest, credentials: dict[str, str]
    ) -> DecisionResponse:
        """Make the API call and return the label probabilities and token counts.

        Implementations should NOT measure latency -- the base class handles that.
        """
        raise NotImplementedError(f"{cls.__name__} does not support decide().")

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
        cls,
        url: str,
        *,
        json: dict,
        headers: dict | None = None,
        **kwargs,
    ) -> requests.Response:
        """Send a POST request and raise on non-200 responses.

        This is an ergonomic wrapper around ``requests.post`` that maps
        common HTTP status codes to specific ``ProviderError`` subclasses
        so callers get uniform error handling across providers.

        Returns the :class:`requests.Response` on success.
        """
        if cls.request_timeout is not None:
            kwargs.setdefault("timeout", cls.request_timeout)

        for attempt in range(cls.max_retries + 1):
            response = requests.post(url, json=json, headers=headers, **kwargs)

            if response.status_code == 200:
                return response

            if response.status_code in cls.retry_statuses and attempt < cls.max_retries:
                delay = _parse_retry_after(response.headers.get("Retry-After"))
                if delay is None or delay < 0:
                    delay = _calculate_backoff(
                        attempt, cls.initial_delay, cls.backoff_factor, cls.max_delay
                    )
                delay = min(delay, cls.max_delay)

                time.sleep(delay)
                continue

            error_cls, status_code, message = cls._extract_error_details(response)
            raise error_cls(
                status_code=status_code,
                message=message,
                provider=cls.__name__,
                body=response.text,
            )

        raise AssertionError("unreachable")

    @classmethod
    def _extract_error_details(
        cls, response: requests.Response
    ) -> tuple[type[ProviderError], int, str]:
        """Extract error class, status code, and descriptive message from a failed response."""
        data: dict | None = None
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            pass

        server_msg = _extract_error_message(data)
        if server_msg and not isinstance(server_msg, str):
            server_msg = str(server_msg)

        status_code = response.status_code or 0
        error_cls, final_status_code = cls._classify_error(status_code, data, server_msg)
        message = (
            f"{cls.__name__}: HTTP {final_status_code} - {server_msg or response.reason or 'Error'}"
        )
        return error_cls, final_status_code, message

    @classmethod
    def _classify_error(
        cls, status_code: int, data: dict | None, message: str | None
    ) -> tuple[type[ProviderError], int]:
        """Map HTTP status and response payload to an error class. Overridden by some providers."""
        return STATUS_TO_ERROR.get(status_code, ProviderError), status_code

    @classmethod
    def _address_override(cls, model_id: str) -> str | None:
        """Return the ``@<address>`` suffix when it is a user-supplied server address.

        The base implementation treats any ``@`` suffix as an address.
        Providers whose ``@`` means something else (a region, e.g. bedrock and
        vertex) override this to return ``None``.
        """
        if "@" in model_id:
            return model_id.rsplit("@", 1)[1]
        return None

    @classmethod
    def _resolve_credentials(cls, request: CompletionRequest | DecisionRequest) -> dict[str, str]:
        """Read all required environment variables.

        Returns a ``{var_name: value}`` dict for every name listed in
        ``required_env``.

        When the model id carries a user-supplied ``@<address>``
        (:func:`_address_override`), the provider's own key is neither required
        nor ever sent to that address, so no credentials are resolved.

        Raises :class:`AuthenticationError` when any variable is unset or empty.
        """
        if cls._address_override(request.model_id) is not None:
            return {}

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

    Looks up ``<Name>Provider`` in ``lmdk.providers.shapes.<name>`` first and
    then in ``lmdk.providers.vendors.<name>`` (e.g. ``"chat"`` ->
    ``ChatProvider``, ``"mistral"`` -> ``MistralProvider``). Shape and vendor
    names must not overlap.

    Raises:
        ModuleNotFoundError: If no shape or vendor matches *name*.
        ValueError: If *name* matches both a shape and a vendor.
        AttributeError: If the module does not contain the expected class.
    """
    modules = [
        f"lmdk.providers.{package}.{name}"
        for package in ("shapes", "vendors")
        if importlib.util.find_spec(f"lmdk.providers.{package}.{name}") is not None
    ]
    if len(modules) > 1:
        raise ValueError(
            f"Provider name '{name}' matches both a shape and a vendor: {', '.join(modules)}. "
            "Shape and vendor names must not overlap."
        )
    if not modules:
        raise ModuleNotFoundError(
            f"No lmdk provider named '{name}' "
            "(looked in lmdk.providers.shapes and lmdk.providers.vendors)."
        )
    module = importlib.import_module(modules[0])
    class_name = f"{name.capitalize()}Provider"
    return getattr(module, class_name)


def resolve_model(model: str) -> tuple[str, str, type[Provider]]:
    """Split ``"<provider>:<model_id>"`` and load the provider class.

    Any ``@location`` suffix stays in ``model_id`` for the provider to parse.

    Returns:
        ``(provider_name, model_id, provider_cls)``.
    """
    provider_name, model_id = model.split(":", maxsplit=1)
    return provider_name, model_id, load_provider(name=provider_name)
