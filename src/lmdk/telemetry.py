"""Optional OpenTelemetry instrumentation for lmdk completions."""

import importlib
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from lmdk.datatypes import CompletionRequest, CompletionResponse

# Targeted OpenTelemetry GenAI Semantic Conventions version: v1.41.0.
# TODO: Revisit the targeted semconv version on each lmdk release.
# TODO: Add finish reasons if provider response contracts expose them later.

TelemetryMode = Literal["off", "metadata", "content"]

_OFF_VALUES = {"", "off", "0", "false"}
_METADATA_VALUES = {"metadata", "on", "1", "true"}
_CONTENT_VALUES = {"content"}
_REQUEST_ATTRIBUTE_NAMES = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "stop_sequences": "gen_ai.request.stop_sequences",
}


class _NoopCompletionTelemetry:
    """Completion telemetry handle used when instrumentation is disabled."""

    def record_response(self, response: CompletionResponse) -> None:
        """Accept a response without recording telemetry."""


class _CompletionTelemetry:
    """Completion telemetry handle backed by an active OpenTelemetry span."""

    def __init__(
        self,
        *,
        span: Any,
        token_usage_histogram: Any,
        metric_attributes: dict[str, Any],
    ) -> None:
        self._span = span
        self._token_usage_histogram = token_usage_histogram
        self._metric_attributes = metric_attributes

    def record_response(self, response: CompletionResponse) -> None:
        """Record response token usage on the current span and meter."""
        self._record_token_count(response.input_tokens, "input")
        self._record_token_count(response.output_tokens, "output")
        self._span.set_attribute("gen_ai.usage.input_tokens", response.input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", response.output_tokens)

    def _record_token_count(self, token_count: int | None, token_type: str) -> None:
        if token_count is None:
            return
        attributes = {**self._metric_attributes, "gen_ai.token.type": token_type}
        self._token_usage_histogram.record(token_count, attributes=attributes)


@contextmanager
def traced_completion(
    provider_name: str,
    model_id: str,
    request: CompletionRequest,
    fallback_index: int,
) -> Generator[_NoopCompletionTelemetry | _CompletionTelemetry]:
    """Trace a non-streaming completion attempt if lmdk telemetry is enabled."""
    mode = _get_telemetry_mode()
    if mode == "off":
        yield _NoopCompletionTelemetry()
        return

    otel = _load_otel()
    if otel is None:
        yield _NoopCompletionTelemetry()
        return

    trace, metrics, SpanKind, Status, StatusCode = otel
    model_name, location = _split_model_and_location(model_id)
    span_attributes = _span_attributes(provider_name, model_name, location, request, fallback_index)
    if mode == "content":
        span_attributes.update(_content_attributes(request))

    metric_attributes = _metric_attributes(provider_name, model_name)
    meter = metrics.get_meter("lmdk")
    duration_histogram = meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="s",
    )
    token_usage_histogram = meter.create_histogram(
        "gen_ai.client.token.usage",
        unit="{token}",
    )

    start = time.perf_counter()
    error_type: str | None = None
    tracer = trace.get_tracer("lmdk")
    with tracer.start_as_current_span(
        f"chat {model_id}",
        kind=SpanKind.CLIENT,
        attributes=span_attributes,
    ) as span:
        telemetry = _CompletionTelemetry(
            span=span,
            token_usage_histogram=token_usage_histogram,
            metric_attributes=metric_attributes,
        )
        try:
            yield telemetry
        except Exception as exc:
            error_type = type(exc).__name__
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration_attributes = dict(metric_attributes)
            if error_type is not None:
                duration_attributes["error.type"] = error_type
            duration_histogram.record(
                time.perf_counter() - start,
                attributes=duration_attributes,
            )


def _get_telemetry_mode() -> TelemetryMode:
    value = os.getenv("LMDK_TELEMETRY", "").strip().lower()
    if value in _CONTENT_VALUES:
        return "content"
    if value in _METADATA_VALUES:
        return "metadata"
    return "off"


def _load_otel() -> tuple[Any, Any, Any, Any, Any] | None:
    try:
        trace = importlib.import_module("opentelemetry.trace")
        metrics = importlib.import_module("opentelemetry.metrics")
    except ImportError:
        return None

    return (
        trace,
        metrics,
        trace.SpanKind,
        trace.Status,
        trace.StatusCode,
    )


def _split_model_and_location(model_id: str) -> tuple[str, str | None]:
    model_name, separator, location = model_id.rpartition("@")
    if not separator or not model_name or not location:
        return model_id, None
    return model_name, location


def _span_attributes(
    provider_name: str,
    model_name: str,
    location: str | None,
    request: CompletionRequest,
    fallback_index: int,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "gen_ai.provider.name": provider_name,
        "gen_ai.request.model": model_name,
        "lmdk.fallback_index": fallback_index,
    }
    if location is not None:
        attributes["lmdk.location"] = location

    for kwarg_name, attribute_name in _REQUEST_ATTRIBUTE_NAMES.items():
        value = request.generation_kwargs.get(kwarg_name)
        if value is not None:
            attributes[attribute_name] = value

    return attributes


def _metric_attributes(provider_name: str, model_name: str) -> dict[str, Any]:
    return {
        "gen_ai.provider.name": provider_name,
        "gen_ai.request.model": model_name,
    }


def _content_attributes(request: CompletionRequest) -> dict[str, str]:
    attributes = {
        "gen_ai.input.messages": json.dumps(
            [
                {
                    "role": message.role,
                    "parts": [{"type": "text", "content": message.content}],
                }
                for message in request.prompt
            ],
        )
    }
    if request.system_instruction is not None:
        attributes["gen_ai.system_instructions"] = json.dumps(
            [{"type": "text", "content": request.system_instruction}]
        )
    return attributes
