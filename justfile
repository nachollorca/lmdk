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
format:
    uv run ruff check --fix .
    uv run ruff format .

check-types:
    uv run ty check src

check-complexity:
    uv run complexipy src

# Test and run ----------------------------------------------------------------
test target="":
    uv run pytest --cov --cov-fail-under=90 {{ target }}

run file:
    uv run --env-file .env {{ file }}

# Project specific commands ---------------------------------------------------
ipython:
    uv run ipython

validate model="":
    uv run --env-file .env example.py {{ model }}
