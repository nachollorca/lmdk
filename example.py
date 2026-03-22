"""Example usage of the lmdk library.

This script serves two purposes:
  1. A quick-start guide showing every feature of the public API.
  2. A provider conformance checker -- run it against a new provider to see
     which capabilities are implemented and which ones still raise errors.

Usage:
    # Set the API key for your provider, then run:
    just validate                                   # uses default model
    just validate mistral:mistral-small-2603        # specify a model

Each section is independent and wrapped in try/except so a failure in one
section never blocks the rest. Look for [OK] and [FAILED] in the output.
"""

import argparse

from pydantic import BaseModel

from lmdk import CompletionResponse, complete, complete_batch
from lmdk.datatypes import AssistantMessage, UserMessage

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "mistral:mistral-small-2603"
SEPARATOR = "=" * 60


# ── Helpers ────────────────────────────────────────────────────────────────
def print_response(label: str, response: CompletionResponse) -> None:
    """Print a CompletionResponse in a consistent, debug-friendly format."""
    print(f"[OK] {label}")
    print(f"  .content      = {response.content!r}")
    print(f"  .parsed       = {response.parsed!r}")
    print(f"  .output       = {response.output!r}")
    print(f"  .input_tokens = {response.input_tokens}")
    print(f"  .output_tokens= {response.output_tokens}")
    print(f"  .latency      = {response.latency:.3f}s")


def section(number: int, title: str) -> None:
    """Print a section header."""
    print(f"\n{SEPARATOR}")
    print(f"  Section {number}: {title}")
    print(SEPARATOR)


# ── Pydantic schemas (defined at module level for reuse) ──────────────────


class Person(BaseModel):
    name: str
    age: int


class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""


class Recipe(BaseModel):
    ingredients: list[Ingredient]


class Summary(BaseModel):
    text: str


class City(BaseModel):
    name: str
    country: str
    population_million: float


# ── Main ──────────────────────────────────────────────────────────────────


def main(model: str) -> None:
    """Run all example sections against the given model."""

    # ── Section 1: Basic text completion ──────────────────────────────────
    # The simplest possible call: a model string and a plain text prompt.
    # The string is automatically wrapped into a UserMessage.
    section(1, "Basic text completion")
    try:
        response = complete(model=model, prompt="Say hello in one sentence.")
        print_response("Basic text completion", response)
    except Exception as e:
        print(f"[FAILED] Basic text completion -> {type(e).__name__}: {e}")

    # ── Section 2: Multi-turn conversation ────────────────────────────────
    # Instead of a plain string, pass a list of Message objects to simulate
    # a conversation with history.  UserMessage and AssistantMessage set the
    # role automatically.
    section(2, "Multi-turn conversation")
    try:
        prompt = [
            UserMessage("My name is Alice."),
            AssistantMessage("Nice to meet you, Alice!"),
            UserMessage("What is my name?"),
        ]
        response = complete(model=model, prompt=prompt)
        print_response("Multi-turn conversation", response)
    except Exception as e:
        print(f"[FAILED] Multi-turn conversation -> {type(e).__name__}: {e}")

    # ── Section 3: System instruction ─────────────────────────────────────
    # A system instruction is prepended to the conversation. Useful for setting
    # the tone, persona, or constraints of the model.
    section(3, "System instruction")
    try:
        response = complete(
            model=model,
            prompt="Hi!",
            system_instruction="You are a pirate. Always answer in pirate speak.",
        )
        print_response("System instruction", response)
    except Exception as e:
        print(f"[FAILED] System instruction -> {type(e).__name__}: {e}")

    # ── Section 4: Generation kwargs ──────────────────────────────────────
    # Pass provider-specific generation parameters like temperature and
    # max_tokens.  Default is {"temperature": 0} when not specified.
    section(4, "Generation kwargs")
    try:
        response = complete(
            model=model,
            prompt="Write a poem.",
            generation_kwargs={"temperature": 0.9, "max_tokens": 10},
        )
        print_response("Generation kwargs", response)
    except Exception as e:
        print(f"[FAILED] Generation kwargs -> {type(e).__name__}: {e}")

    # ── Section 5: Streaming ─────────────────────────────────────────────
    # stream=True returns an iterator of string tokens instead of a
    # CompletionResponse.  Note: streaming and output_schema are mutually
    # exclusive.
    section(5, "Streaming")
    try:
        token_iter = complete(model=model, prompt="Count from 1 to 5.", stream=True)
        print("[OK] Streaming")
        print("  tokens: ", end="")
        for token in token_iter:
            print(token, end="", flush=True)
        print()  # newline after stream
    except Exception as e:
        print(f"[FAILED] Streaming -> {type(e).__name__}: {e}")

    # ── Section 6: Model fallback ────────────────────────────────────────
    # Pass a list of models. lmdk tries each in order, falling back to the
    # next on failure.  Here the first model uses a non-existent model ID
    # (same provider), so the API call should fail and the second should
    # succeed.
    section(6, "Model fallback")
    provider = model.split(":")[0]
    try:
        response = complete(
            model=[f"{provider}:nonexistent-model-12345", model],
            prompt="Say 'fallback worked' and nothing else.",
        )
        print_response("Model fallback", response)
    except Exception as e:
        print(f"[FAILED] Model fallback -> {type(e).__name__}: {e}")

    # ── Section 7: Structured output (simple) ────────────────────────────
    # Pass a Pydantic BaseModel as output_schema.  The provider must return
    # JSON that validates against this schema.  The parsed object is available
    # in response.parsed, and response.output applies unwrapping logic.
    section(7, "Structured output (simple)")
    try:
        response = complete(
            model=model,
            prompt="My coworker Jesus is 33 years old.",
            output_schema=Person,
        )
        print_response("Structured output (simple)", response)
        print(f"  type(.parsed) = {type(response.parsed).__name__}")
        print(f"  type(.output) = {type(response.output).__name__}")
    except Exception as e:
        print(f"[FAILED] Structured output (simple) -> {type(e).__name__}: {e}")

    # ── Section 8: Structured output (compound / nested) ─────────────────
    # Nested Pydantic models work too.  This tests that the provider can
    # handle more complex JSON schemas.
    section(8, "Structured output (compound)")
    try:
        response = complete(
            model=model,
            prompt="How do I make gazpacho?",
            output_schema=Recipe,
        )
        print_response("Structured output (compound)", response)
    except Exception as e:
        print(f"[FAILED] Structured output (compound) -> {type(e).__name__}: {e}")

    # ── Section 9: Single-field unwrapping ────────────────────────────────
    # When the output_schema has exactly ONE field, response.output returns
    # just that field's value (not the whole BaseModel).  This is a convenience
    # for common patterns like wrapping a single list or string in a schema.
    #
    # Compare .parsed (the full BaseModel) vs .output (the unwrapped value):
    #   .parsed  -> Summary(text="...")
    #   .output  -> "..."
    section(9, "Single-field unwrapping")
    try:
        response = complete(
            model=model,
            prompt="Summarize the theory of relativity in one sentence.",
            output_schema=Summary,
        )
        print_response("Single-field unwrapping", response)
        print(f"  type(.parsed) = {type(response.parsed).__name__}  (full BaseModel)")
        print(f"  type(.output) = {type(response.output).__name__}  (unwrapped field)")
    except Exception as e:
        print(f"[FAILED] Single-field unwrapping -> {type(e).__name__}: {e}")

    # ── Section 10: Batch responses ───────────────────────────────────────
    # complete_batch sends multiple prompts in parallel using a thread
    # pool.  Each result is either a CompletionResponse or an Exception.
    section(10, "Batch responses")
    try:
        results = complete_batch(
            model=model,
            prompt_list=["Say 'hello' and nothing else.", "Say 'hola' and nothing else."],
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  [{i}] [FAILED] {type(result).__name__}: {result}")
            else:
                print(
                    f"  [{i}] [OK] content={result.content!r}  "
                    f"tokens={result.input_tokens}+{result.output_tokens}  "
                    f"latency={result.latency:.3f}s"
                )
    except Exception as e:
        print(f"[FAILED] Batch responses -> {type(e).__name__}: {e}")

    # ── Section 11: Batch with structured output ──────────────────────────
    # Batch mode also supports output_schema.  Each response in the list
    # will have .parsed and .output populated.
    section(11, "Batch with structured output")
    try:
        results = complete_batch(
            model=model,
            prompt_list=[
                "Tell me about Tokyo.",
                "Tell me about Paris.",
            ],
            output_schema=City,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  [{i}] [FAILED] {type(result).__name__}: {result}")
            else:
                print(f"  [{i}] [OK] parsed={result.parsed!r}  output={result.output!r}")
    except Exception as e:
        print(f"[FAILED] Batch with structured output -> {type(e).__name__}: {e}")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  All sections executed. Check [OK] / [FAILED] above.")
    print(SEPARATOR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="lmdk example / provider conformance checker",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model ID in provider:model format (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    main(args.model)
