set dotenv-load

sync:
    uv lock --upgrade
    uv sync --all-extras

reset-env:
    rm -rf .venv
    uv sync

install-hooks:
    prek install
    prek install --hook-type commit-msg

format:
    uvx ruff check --select I --fix .
    uvx ruff format .

test target="":
    uv run pytest --cov --cov-fail-under=90 {{target}}

check-types:
    uvx ty check src

ipython:
    uv run ipython

analyze-complexity:
    uvx complexipy src

run file:
    uv run --env-file .env {{file}}

validate model="":
    uv run --env-file .env example.py {{model}}
