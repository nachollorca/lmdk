"""Optional OpenTelemetry instrumentation for lmdk completions."""

import importlib
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, CompletionResponse
from lmdk.provider import Provider

# Targeted OpenTelemetry GenAI Semantic Conventions version: v1.41.0.
# TODO: Revisit the targeted semconv version on each lmdk release.

TelemetryMode = Literal["off", "metadata", "content"]

_REQUEST_ATTRIBUTE_NAMES = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "stop_sequences": "gen_ai.request.stop_sequences",
}


class _CompletionTelemetry:
    """Completion telemetry handle. A no-op when ``span`` is ``None``."""

    def __init__(
        self,
        *,
        span: Any = None,
        token_usage_histogram: Any = None,
        metric_attributes: dict[str, Any] | None = None,
        capture_content: bool = False,
    ) -> None:
        self._span = span
        self._token_usage_histogram = token_usage_histogram
        self._metric_attributes = metric_attributes or {}
        self._capture_content = capture_content

    def record_response(self, response: CompletionResponse) -> None:
        """Record response token usage on the current span and meter."""
        if self._span is None:
            return
        for token_count, token_type in (
            (response.input_tokens, "input"),
            (response.output_tokens, "output"),
        ):
            if token_count is not None:
                self._token_usage_histogram.record(
                    token_count,
                    attributes={**self._metric_attributes, "gen_ai.token.type": token_type},
                )
        self._span.set_attribute("gen_ai.usage.input_tokens", response.input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", response.output_tokens)
        if self._capture_content:
            self._span.set_attribute(
                "gen_ai.output.messages",
                json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": response.content}],
                        }
                    ]
                ),
            )
            # The OTel GenAI semconv has no attribute for structured output(yet)
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                self._span.set_attribute("lmdk.parsed", json.dumps(_to_jsonable(parsed)))
                self._span.set_attribute("lmdk.output", json.dumps(_to_jsonable(response.output)))


@contextmanager
def traced_completion(
    provider: type[Provider],
    provider_name: str,
    model_id: str,
    request: CompletionRequest,
    fallback_index: int,
) -> Generator[_CompletionTelemetry]:
    """Trace a non-streaming completion attempt if lmdk telemetry is enabled."""
    mode = _get_telemetry_mode()
    otel = _load_otel() if mode != "off" else None
    if otel is None:
        yield _CompletionTelemetry()
        return

    trace, metrics = otel
    model_name, location = _split_model_and_location(model_id)
    span_attributes = _span_attributes(
        provider, provider_name, model_name, location, request, fallback_index
    )
    if mode == "content":
        span_attributes.update(_content_attributes(request))

    metric_attributes = {
        "gen_ai.provider.name": provider_name,
        "gen_ai.request.model": model_name,
    }
    meter = metrics.get_meter("lmdk")
    duration_histogram = meter.create_histogram("gen_ai.client.operation.duration", unit="s")
    token_usage_histogram = meter.create_histogram("gen_ai.client.token.usage", unit="{token}")

    start = time.perf_counter()
    error_type: str | None = None
    tracer = trace.get_tracer("lmdk")
    with tracer.start_as_current_span(
        f"chat {model_id}",
        kind=trace.SpanKind.CLIENT,
        attributes=span_attributes,
        record_exception=False,
    ) as span:
        try:
            yield _CompletionTelemetry(
                span=span,
                token_usage_histogram=token_usage_histogram,
                metric_attributes=metric_attributes,
                capture_content=mode == "content",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            span.set_attribute("error.type", error_type)
            raise
        finally:
            duration_attributes = dict(metric_attributes)
            if error_type is not None:
                duration_attributes["error.type"] = error_type
            duration_histogram.record(time.perf_counter() - start, attributes=duration_attributes)


def _to_jsonable(value: Any) -> Any:
    """Convert pydantic models (and containers thereof) to JSON-serializable data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _get_telemetry_mode() -> TelemetryMode:
    value = os.getenv("LMDK_TELEMETRY", "").strip().lower()
    if value == "content":
        return "content"
    if value in {"metadata", "on", "1", "true"}:
        return "metadata"
    return "off"


def _load_otel() -> tuple[Any, Any] | None:
    try:
        trace = importlib.import_module("opentelemetry.trace")
        metrics = importlib.import_module("opentelemetry.metrics")
    except ImportError:
        return None
    return trace, metrics


def _split_model_and_location(model_id: str) -> tuple[str, str | None]:
    model_name, _, location = model_id.rpartition("@")
    if model_name and location:
        return model_name, location
    return model_id, None


def _span_attributes(
    provider: type[Provider],
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

    attributes["gen_ai.request.reasoning.level"] = provider.request_reasoning_level(request)

    return attributes


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
