"""Tests for optional OpenTelemetry telemetry."""

import importlib
import json

import pytest

from lmdk.core import complete
from lmdk.datatypes import CompletionRequest, RawResponse, UserMessage
from lmdk.telemetry import traced_completion


@pytest.fixture()
def otel_setup():
    pytest.importorskip("opentelemetry")
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import _internal as metrics_internal
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._set_tracer_provider(tracer_provider, log=False)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False
    metrics_internal._set_meter_provider(meter_provider, log=False)

    yield span_exporter, metric_reader

    span_exporter.clear()
    tracer_provider.shutdown()
    meter_provider.shutdown()
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False
    metrics_internal._set_meter_provider(metrics.NoOpMeterProvider(), log=False)


def _request() -> CompletionRequest:
    return CompletionRequest(
        model_id="model",
        prompt=[UserMessage(content="hello")],
        system_instruction="system secret",
        output_schema=None,
        generation_kwargs={"temperature": 0.2, "max_tokens": 7},
    )


def _metric_points(metric_reader, metric_name: str):
    metrics_data = metric_reader.get_metrics_data()
    if metrics_data is None:
        return []

    points = []
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)
    return points


def test_telemetry_off_does_not_import_opentelemetry(monkeypatch):
    calls = []

    def fail_on_import(name):
        calls.append(name)
        raise AssertionError("OpenTelemetry should not be imported when telemetry is off")

    monkeypatch.delenv("LMDK_TELEMETRY", raising=False)
    monkeypatch.setattr(importlib, "import_module", fail_on_import)

    with traced_completion("fake", "model", _request(), fallback_index=0) as telemetry:
        telemetry.record_response(RawResponse(content="ok", input_tokens=1, output_tokens=2))

    assert calls == []


def test_enabled_telemetry_gracefully_noops_when_opentelemetry_is_missing(monkeypatch):
    def fail_for_otel(name):
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return importlib.import_module(name)

    monkeypatch.setenv("LMDK_TELEMETRY", "metadata")
    monkeypatch.setattr(importlib, "import_module", fail_for_otel)

    with traced_completion("fake", "model", _request(), fallback_index=0) as telemetry:
        telemetry.record_response(RawResponse(content="ok", input_tokens=1, output_tokens=2))


def test_metadata_mode_records_span_attributes_and_metrics(
    monkeypatch, patch_load_provider, otel_setup
):
    span_exporter, metric_reader = otel_setup
    monkeypatch.setenv("LMDK_TELEMETRY", "metadata")

    result = complete(
        model="fake:model@eu-west1",
        prompt="hello",
        system_instruction="do not leak",
        generation_kwargs={"temperature": 0.3, "max_tokens": 8, "stop_sequences": ["STOP"]},
    )

    assert result.content == "fake response"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat model@eu-west1"
    assert span.kind.name == "CLIENT"
    assert span.attributes["gen_ai.provider.name"] == "fake"
    assert span.attributes["gen_ai.request.model"] == "model"
    assert span.attributes["lmdk.location"] == "eu-west1"
    assert span.attributes["lmdk.fallback_index"] == 0
    assert span.attributes["gen_ai.request.temperature"] == 0.3
    assert span.attributes["gen_ai.request.max_tokens"] == 8
    assert span.attributes["gen_ai.request.stop_sequences"] == ("STOP",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.system_instructions" not in span.attributes

    duration_points = _metric_points(metric_reader, "gen_ai.client.operation.duration")
    assert len(duration_points) == 1
    assert duration_points[0].attributes == {
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "model",
    }
    assert duration_points[0].count == 1
    assert duration_points[0].sum >= 0

    token_points = _metric_points(metric_reader, "gen_ai.client.token.usage")
    assert {point.attributes["gen_ai.token.type"]: point.sum for point in token_points} == {
        "input": 10,
        "output": 5,
    }


def test_content_mode_records_prompt_and_system_instruction(monkeypatch, otel_setup):
    span_exporter, _ = otel_setup
    monkeypatch.setenv("LMDK_TELEMETRY", "content")

    with traced_completion("fake", "model", _request(), fallback_index=2) as telemetry:
        telemetry.record_response(RawResponse(content="ok", input_tokens=1, output_tokens=2))

    span = span_exporter.get_finished_spans()[0]
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert json.loads(span.attributes["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "system secret"}
    ]


def test_fallback_records_one_span_per_non_streaming_attempt(
    monkeypatch, patch_load_provider, otel_setup
):
    span_exporter, _ = otel_setup
    monkeypatch.setenv("LMDK_TELEMETRY", "metadata")
    call_count = {"n": 0}

    def fail_then_succeed(request, api_key):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first model down")
        return RawResponse(content="from fallback", input_tokens=3, output_tokens=4)

    patch_load_provider.response_fn = fail_then_succeed

    result = complete(model=["fake:first", "fake:second"], prompt="hello")

    assert result.content == "from fallback"
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["chat first", "chat second"]
    assert [span.attributes["lmdk.fallback_index"] for span in spans] == [0, 1]
    assert spans[0].attributes["error.type"] == "RuntimeError"
    assert spans[0].status.status_code.name == "ERROR"
    assert "error.type" not in spans[1].attributes


def test_streaming_completions_are_not_instrumented(monkeypatch, patch_load_provider, otel_setup):
    span_exporter, metric_reader = otel_setup
    monkeypatch.setenv("LMDK_TELEMETRY", "metadata")

    result = complete(model="fake:model", prompt="hello", stream=True)

    assert list(result) == ["chunk1", "chunk2"]
    assert span_exporter.get_finished_spans() == ()
    assert _metric_points(metric_reader, "gen_ai.client.operation.duration") == []
