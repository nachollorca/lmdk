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
from collections.abc import Callable

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


def run_section(number: int, title: str, action: Callable[[], None]) -> None:
    """Run one example section, printing failures without stopping the script."""
    section(number, title)
    try:
        action()
    except Exception as e:
        print(f"[FAILED] {title} -> {type(e).__name__}: {e}")


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


# ── Section runners ───────────────────────────────────────────────────────


def _basic_text_completion(model: str) -> None:
    response = complete(model=model, prompt="Say hello in one sentence.")
    print_response("Basic text completion", response)


def _multi_turn_conversation(model: str) -> None:
    prompt = [
        UserMessage("My name is Alice."),
        AssistantMessage("Nice to meet you, Alice!"),
        UserMessage("What is my name?"),
    ]
    response = complete(model=model, prompt=prompt)
    print_response("Multi-turn conversation", response)


def _system_instruction(model: str) -> None:
    response = complete(
        model=model,
        prompt="Hi!",
        system_instruction="You are a pirate. Always answer in pirate speak.",
    )
    print_response("System instruction", response)


def _generation_kwargs(model: str) -> None:
    response = complete(
        model=model,
        prompt="Write a poem.",
        generation_kwargs={"temperature": 0.9, "max_tokens": 10},
    )
    print_response("Generation kwargs", response)


def _streaming(model: str) -> None:
    token_iter = complete(model=model, prompt="Count from 1 to 5.", stream=True)
    print("[OK] Streaming")
    print("  tokens: ", end="")
    for token in token_iter:
        print(token, end="", flush=True)
    print()


def _model_fallback(model: str) -> None:
    provider = model.split(":")[0]
    response = complete(
        model=[f"{provider}:nonexistent-model-12345", model],
        prompt="Say 'fallback worked' and nothing else.",
    )
    print_response("Model fallback", response)


def _structured_output_simple(model: str) -> None:
    response = complete(
        model=model,
        prompt="My coworker Jesus is 33 years old.",
        output_schema=Person,
    )
    print_response("Structured output (simple)", response)
    print(f"  type(.parsed) = {type(response.parsed).__name__}")
    print(f"  type(.output) = {type(response.output).__name__}")


def _structured_output_compound(model: str) -> None:
    response = complete(
        model=model,
        prompt="How do I make gazpacho?",
        output_schema=Recipe,
    )
    print_response("Structured output (compound)", response)


def _single_field_unwrapping(model: str) -> None:
    response = complete(
        model=model,
        prompt="Summarize the theory of relativity in one sentence.",
        output_schema=Summary,
    )
    print_response("Single-field unwrapping", response)
    print(f"  type(.parsed) = {type(response.parsed).__name__}  (full BaseModel)")
    print(f"  type(.output) = {type(response.output).__name__}  (unwrapped field)")


def _batch_responses(model: str) -> None:
    batch = complete_batch(
        model=model,
        prompt_list=["Say 'hello' and nothing else.", "Say 'hola' and nothing else."],
    )
    for i, result in enumerate(batch):
        if isinstance(result, Exception):
            print(f"  [{i}] [FAILED] {type(result).__name__}: {result}")
        else:
            print(
                f"  [{i}] [OK] content={result.content!r}  "
                f"tokens={result.input_tokens}+{result.output_tokens}  "
                f"latency={result.latency:.3f}s"
            )
    print(
        f"  batch totals: input={batch.input_tokens}  output={batch.output_tokens}  "
        f"latency={batch.latency:.3f}s  errors={len(batch.errors)}"
    )


def _batch_with_structured_output(model: str) -> None:
    batch = complete_batch(
        model=model,
        prompt_list=[
            "Tell me about Tokyo.",
            "Tell me about Paris.",
        ],
        output_schema=City,
    )
    for i, result in enumerate(batch):
        if isinstance(result, Exception):
            print(f"  [{i}] [FAILED] {type(result).__name__}: {result}")
        else:
            print(f"  [{i}] [OK] parsed={result.parsed!r}  output={result.output!r}")


# ── Main ──────────────────────────────────────────────────────────────────


def main(model: str) -> None:
    """Run all example sections against the given model."""
    run_section(1, "Basic text completion", lambda: _basic_text_completion(model))
    run_section(2, "Multi-turn conversation", lambda: _multi_turn_conversation(model))
    run_section(3, "System instruction", lambda: _system_instruction(model))
    run_section(4, "Generation kwargs", lambda: _generation_kwargs(model))
    run_section(5, "Streaming", lambda: _streaming(model))
    run_section(6, "Model fallback", lambda: _model_fallback(model))
    run_section(7, "Structured output (simple)", lambda: _structured_output_simple(model))
    run_section(8, "Structured output (compound)", lambda: _structured_output_compound(model))
    run_section(9, "Single-field unwrapping", lambda: _single_field_unwrapping(model))
    run_section(10, "Batch responses", lambda: _batch_responses(model))
    run_section(11, "Batch with structured output", lambda: _batch_with_structured_output(model))

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
