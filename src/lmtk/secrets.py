"""Secrets management.

Stores and retrieves API keys from `$XDG_CONFIG_HOME/lmtk/secrets.yaml`
Defaults to `~/.config/lmtk/secrets.yaml`
Environment variables always take precedence over stored secrets
"""

# TODO: does it make sense that env vars take precedence over stored secrets? If not, change it

import os
import stat
from pathlib import Path

import yaml


def secrets_path() -> Path:
    """Return the path to the secrets file.

    Uses `$XDG_CONFIG_HOME/lmtk/secrets.yaml` if set,
    otherwise `~/.config/lmtk/secrets.yaml`.

    Returns:
        Path to the secrets file.
    """
    config_home = os.environ.get("XDG_CONFIG_HOME", "")
    if config_home:
        return Path(config_home) / "lmtk" / "secrets.yaml"
    return Path.home() / ".config" / "lmtk" / "secrets.yaml"


# TODO: do we really need to pass a custom path? Don't we always want the one made in `secrets_path`? Remove otherwise
# If we do not need it, remove it also from the other functions
def load_secret(key: str, path: Path | None = None) -> str | None:
    """Load a single secret value by key.

    Args:
        key: The secret key to look up (e.g. `"MISTRAL_API_KEY"`).
        path: Override path for the secrets file.  Defaults to the
            standard XDG location.

    Returns:
        The secret value, or `None` if the file or key does not exist.
    """
    if path is None:
        path = secrets_path()

    if not path.is_file():
        return None

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    value = data.get(key)
    return str(value) if value is not None else None


def save_secret(key: str, value: str, path: Path | None = None) -> None:
    """Save a secret to the secrets file.

    Creates the file and parent directories if they do not exist.
    Sets file permissions to owner-only (`0600`) for security.

    Args:
        key: The secret key to store.
        value: The secret value.
        path: Override path for the secrets file.  Defaults to the
            standard XDG location.
    """
    if path is None:
        path = secrets_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, str] = {}
    if path.is_file():
        with open(path) as f:
            existing = yaml.safe_load(f)
        if isinstance(existing, dict):
            data = existing

    data[key] = value
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def resolve_api_key(env_var: str, path: Path | None = None) -> str | None:
    """Resolve an API key by checking the environment first, then the secrets file.

    Args:
        env_var: The environment variable name to check.
        path: Override path for the secrets file.

    Returns:
        The API key string, or `None` if not found in either location.
    """
    value = os.environ.get(env_var)
    if value:
        return value
    return load_secret(env_var, path)
