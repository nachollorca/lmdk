set dotenv-load := true

# Python environment ----------------------------------------------------------

# Refresh the lockfile and sync (use when you intentionally want newer deps).
upgrade:
    uv lock --upgrade
    uv sync --all-extras --all-groups

# Install from the committed lockfile (matches CI).
install:
    uv sync --frozen --all-extras --all-groups

# Nuke the venv and reinstall from the lockfile (last-resort env reset).
reset-env:
    rm -rf .venv
    uv sync --frozen --all-extras --all-groups

# Pre-commit hooks ------------------------------------------------------------
install-hooks:
    prek install

update-hooks:
    prek auto-update

run-hooks:
    prek run --show-diff-on-failure --color=always -a

# Code quality ----------------------------------------------------------------
format *paths=".":
    uv run --frozen ruff check --fix {{ paths }}
    uv run --frozen ruff format {{ paths }}

check-types *paths=".":
    uv run --frozen ty check {{ paths }}

check-complexity *paths=".":
    uv run --frozen complexipy {{ paths }}

# Test and run ----------------------------------------------------------------
test target="":
    uv run --frozen pytest --cov --cov-fail-under=90 {{ target }}

run file:
    uv run --frozen --env-file .env {{ file }}

# Project specific commands ---------------------------------------------------
ipython:
    uv run --frozen ipython

validate model="":
    uv run --frozen --env-file .env example.py {{ model }}
