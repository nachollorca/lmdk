"""Run the example.py conformance checker against every provider.

This drives example.py once per provider/model so you can validate the whole
provider matrix with a single command:

    just validate-all          # or: uv run --env-file .env validate_all.py

Make sure the API key for each provider is set in your environment (.env).
"""

import subprocess
import sys
from pathlib import Path

# ── Provider matrix ────────────────────────────────────────────────────────
# Maps a human-readable provider name to its model ID in provider:model format.
MODELS = {
    "anthropic": "anthropic:claude-sonnet-4-6",
    "vertex": "vertex:gemini-3.1-flash-lite",
    "openai": "openai:gpt-5.4-mini",
    "mistral": "mistral:mistral-medium-3-5",
}

EXAMPLE = Path(__file__).parent / "example.py"
SEPARATOR = "#" * 70


def main() -> int:
    """Run example.py for each model, returning a non-zero code on any failure."""
    failures: list[str] = []

    for provider, model in MODELS.items():
        print(f"\n{SEPARATOR}")
        print(f"#  Provider: {provider}  ({model})")
        print(SEPARATOR)

        result = subprocess.run([sys.executable, str(EXAMPLE), model], check=False)
        if result.returncode != 0:
            failures.append(f"{provider} ({model}) exited with code {result.returncode}")

    print(f"\n{SEPARATOR}")
    print("#  Summary")
    print(SEPARATOR)
    if failures:
        for failure in failures:
            print(f"  [FAILED] {failure}")
        return 1
    print(f"  [OK] All {len(MODELS)} providers ran to completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
