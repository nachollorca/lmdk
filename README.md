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

What it does **NOT** offer:
- Tools / function calling / MCP
- Agents
- Multimodality (only text-in, text-out)
- Shady under-the-hood prompt modification (e.g. to force structured output)
- API gateways

If you are looking for a more constrained but out-of-the-box agent interface, I'd recommend [pydantic-ai](https://ai.pydantic.dev) or [haystack-ai](https://docs.haystack.deepset.ai/docs/generators).
If you are looking to keep granular control but extend on tools or multimodality, I'd recommend [litellm](https://docs.litellm.ai/docs/) or leveraging the OpenAI-compatible endpoints that providers normally set up.
If you want a unified a token for all providers and are willing to give away telemetry data, check Gateways like [openrouter](https://openrouter.ai).

## Install
`uv add lmdk`

## Basic usage
```python
from lmdk import complete

model = "mistral:mistral-small-2603"
# supports locations as in "vertex:gemini-2.5-flash@europe-west4"
```

<details>
<summary>Single prompt</summary>

```python
response = complete(model=model, messages="Tell me a joke")
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
response = complete(model=model, messages=messages)
```
</details>

<details>
<summary>System prompt and generation kwargs</summary>

```python
response = complete(
    model=model,
    messages="Hi!",
    system_instruction="Talk like a pirate",
    generation_kwargs={"temperature": 0.9, "max_tokens": 10}
)
```
</details>

<details>
<summary>Streaming</summary>

```python
token_iter = complete(model=model, messages="Count from 1 to 5.", stream=True)
```
</details>

<details>
<summary>Model fallbacks</summary>

```python
response = complete(model=["mistral:nonexistent-model", model], messages="Hi")
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

response = complete(model=model, messages="How do I make cheescake?", output_schema=Recipe)
# response.parsed will have a Recipe instance
```
</details>

<details>
<summary>Parallel calls</summary>

```python
results = complete_batch(model=model, messages_list=["Greet in english", "Saluda en espanyol."])
# results will be al list of CompletionResult
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
- `just analyze-complexity`: Cyclomatic complexity checks with `complexipy`.
- `just test`: Runs pytest with 90% coverage threshold.

### Contribute
1. **Hooks**: Install pre-commit hooks via `just install-hooks`. PRs will fail CI if linting/formatting is not applied.
2. **Issues**: Open an issue first using the default template.
3. **PRs**: Link your PR to the relevant issue using the PR template.

You can use `just validate <model>` (runs `example.py`) to verify which features run properly and which do not for a new provider / model.
**Not all of them have to pass to open a PR:** some providers do not even support native structured output. Do at least the normal non-structured, non-streamed completion. The rest can raise `NotImplementedError`.

## License
MIT
