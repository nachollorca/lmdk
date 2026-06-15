# Language Model Development Kit

What it offers:
- **Simplest interface to call different Language Model APIs**
- Minimal dependencies: HTTP requests only, no third party packages
- Streaming
- Comfy structured outputs via Pydantic models, **only if the provider / model supports it natively**
- Parallel completions
- Unified HTTP error handling
- Easy location config (for providers with multiple datacenters like AWS Bedrock, GCP Vertex and Azure)
- Model fallbacks
- Bring Your Own Key (for each provider)
- Optional Telemetry following OpenTelemetry GenAI Semantic Conventions
- In-process observation hook (`observe()`) to capture request/response pairs from wrapped code

What it does **NOT** offer:
- Tools / function calling / MCP
- Agents
- Multimodality (only text-in, text-out)
- Shady under-the-hood prompt modification (e.g. to force structured output)
- API gateways

If you are looking for a more constrained but out-of-the-box agent interface, I'd recommend [pydantic-ai](https://ai.pydantic.dev) or [haystack-ai](https://docs.haystack.deepset.ai/docs/generators).
If you are looking to keep granular control but extend on tools or multimodality, I'd recommend [litellm](https://docs.litellm.ai/docs/) or leveraging the OpenAI-compatible endpoints that providers normally set up.
If you want a unified a token for all providers and are willing to give away telemetry data, check Gateways like [openrouter](https://openrouter.ai).

## Installation
`uv add lmdk`

Optional OpenTelemetry support:

```bash
uv add 'lmdk[telemetry]'
```

## Usage
```python
from lmdk import complete

model = "mistral:mistral-small-2603"
# supports locations as in "vertex:gemini-2.5-flash@europe-west4"
```

<details>
<summary>Single prompt</summary>

```python
response = complete(model=model, prompt="Tell me a joke")
```
</details>

<details>
<summary>Multi-turn conversation</summary>

```python
messages = [
    UserMessage("My name is Alice."),
    AssistantMessage("Nice to meet you, Alice!"),
    UserMessage("What is my name?"),
]
response = complete(model=model, prompt=messages)
```
</details>

<details>
<summary>System prompt and generation kwargs</summary>

```python
response = complete(
    model=model,
    prompt="Hi!",
    system_instruction="Talk like a pirate",
    generation_kwargs={"temperature": 0.9, "max_tokens": 10}
)
```
</details>

<details>
<summary>Streaming</summary>

```python
token_iter = complete(model=model, prompt="Count from 1 to 5.", stream=True)
```
</details>

<details>
<summary>Model fallbacks</summary>

```python
response = complete(model=["mistral:nonexistent-model", model], prompt="Hi")
# first request will raise NotFoundError bc model does not exist, second will work
```
</details>

<details>
<summary>Structured output</summary>

```python
class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""

class Recipe(BaseModel):
    ingredients: list[Ingredient]

response = complete(model=model, prompt="How do I make cheescake?", output_schema=Recipe)
# response.parsed will have a Recipe instance
```
</details>

<details>
<summary>Reasoning / thinking</summary>

```python
# "none" (default) | "low" | "medium" | "high"
response = complete(model=model, prompt="Solve this carefully...", thinking_effort="high")

# Works alongside structured output where the provider supports both:
response = complete(
    model=model,
    prompt="Plan a 3-day trip to Lisbon.",
    output_schema=Trip,
    thinking_effort="medium",
)
```

`thinking_effort` is mapped per provider:
- OpenAI: `reasoning.effort`
- Vertex (Gemini 3): `thinkingConfig.thinkingLevel` (`"low"` / `"medium"` / `"high"`). `"none"` is a no-op since Gemini 3 can't disable thinking.
- Anthropic: `thinking={type: "enabled", budget_tokens: ...}` (low=1024, medium=8192, high=16384). Sampling kwargs (`temperature`, `top_p`, `top_k`) are dropped to satisfy the API constraint.
- Mistral: `reasoning_effort` (the API only accepts `"high"`, so any non-`"none"` level maps to `"high"`)

`generation_kwargs` still wins as an escape hatch when you need exact provider-native values (e.g. `generation_kwargs={"thinkingConfig": {"thinkingBudget": 256}}` on Vertex).
</details>

<details>
<summary>Parallel calls</summary>

```python
from lmdk import complete_batch

batch = complete_batch(model=model, prompt_list=["Greet in english", "Saluda en espanyol."])
# `batch` is a CompletionBatch. Iterate it to handle each outcome:
for result in batch:
    if isinstance(result, Exception):
        ...  # this prompt failed
    else:
        ...  # CompletionResponse

# Aggregates over successful responses:
batch.input_tokens, batch.output_tokens, batch.latency
batch.responses  # successes only
batch.errors     # exceptions only
```
</details>

<details>
<summary>Template Rendering</summary>

```python
from lmdk import render_template

# Render a template string with variables
result = render_template(
    template="Hello, {{ name }}!",
    name="World"
)
# Output: "Hello, World!"

# Render a template from a jinja file
result = render_template(
    path="path/to/template.jinja2",
    name="World"
)
```
</details>

<details>
<summary>Observing wrapped code</summary>

```python
from lmdk import observe

with observe() as obs:
    answer = my_function_that_calls_complete()

for record in obs.records:
    record.request    # CompletionRequest sent to the LM
    record.response   # CompletionResponse returned
```

Useful for tests, evals, and debug tooling where the wrapped function only
returns its own result but you also want to inspect the underlying LM calls.
Streaming completions are not recorded.
</details>

## Telemetry

Telemetry is off by default and adds no required dependencies to the default install.
To enable **OpenTelemetry**-based spans and metrics, install the optional extra and set `LMDK_TELEMETRY`:

```bash
uv add 'lmdk[telemetry]'
export LMDK_TELEMETRY=metadata  # spans/metrics without prompt text
# export LMDK_TELEMETRY=content  # also records prompt, system-instruction, and response text
```

We follows the experimental [**Gen AI semconv**](https://opentelemetry.io/docs/specs/semconv/gen-ai/) v1.41.0. We only instrument non-streaming responses for now.

`lmdk` only emits telemetry through the OpenTelemetry SDK. Your application owns exporter, processor, reader, collector endpoint, i.e.: you decide how and where to send the emitted traces.

Below are some minimal exporter setups. Call them once at process start before invoking `complete` / `complete_batch`.

<details>
<summary>Console (debugging)</summary>

Prints spans to stdout. Useful to verify instrumentation locally without any backend.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def configure_console_traces() -> None:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
```
</details>

<details>
<summary>Pydantic Logfire</summary>

Logfire installs itself as the global `TracerProvider`, so spans emitted by `lmdk` are forwarded automatically. Requires `uv add logfire` and a `LOGFIRE_TOKEN`.

```python
import os
import logfire


def configure_logfire_traces() -> None:
    logfire.configure(
        token=os.environ["LOGFIRE_TOKEN"],
        service_name="my-app",
        # lmdk already controls prompt/response redaction via LMDK_TELEMETRY;
        # don't let Logfire second-guess scrubbing of content.
        scrubbing=False,
        send_to_logfire=True,
    )
```
</details>

<details>
<summary>Grafana (OTLP / Tempo)</summary>

Ship spans over OTLP to Grafana Cloud (or a self-hosted Tempo + OTel Collector). Requires `uv add opentelemetry-exporter-otlp`.

```python
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_grafana_traces() -> None:
    # For Grafana Cloud OTLP, set:
    #   OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
    #   OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64(instanceID:token)>
    exporter = OTLPSpanExporter(
        endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] + "/v1/traces",
    )
    provider = TracerProvider(resource=Resource.create({"service.name": "my-app"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
```
</details>


## Development

### Structure
```text
src/lmdk/
├── core.py         # Entry points: complete, complete_batch
├── datatypes.py    # Common message and response schemas
├── provider.py     # Base Provider class and registry
├── providers/      # Concrete implementations (Mistral, Vertex, etc.)
├── errors.py       # Unified HTTP and API error handling
└── utils.py        # Shared helper functions
```

### Tooling
We use `just` for development tasks. Use:
- `just sync`: Updates lockfile and syncs environment.
- `just format`: Lints and formats with `ruff`.
- `just check-types`: Static analysis with `ty`.
- `just check-complexity`: Cyclomatic complexity checks with `complexipy`.
- `just test`: Runs pytest with 90% coverage threshold.

See [`justfile`](justfile) for a complete list of dev commands.

### Contribute
1. **Hooks**: Install pre-commit hooks via `just install-hooks`. PRs will fail CI if linting/formatting is not applied.
2. **Issues**: Open an issue first using the default template.
3. **PRs**: Link your PR to the relevant issue using the PR template.

You can use `just validate <model>` (runs `example.py`) to verify which features run properly and which do not for a new provider / model.
**Not all of them have to pass to open a PR:** some providers do not even support native structured output. Do at least the normal non-structured, non-streamed completion. The rest can raise `NotImplementedError`.

## License
MIT

_Made with [`mold`](https://github.com/nachollorca/mold) template_
