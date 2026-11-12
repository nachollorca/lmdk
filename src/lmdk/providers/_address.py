"""Shared address rule for provider endpoints.

An *address* is ``host[:port]`` or ``scheme://host[:port]``; ``http://`` is
assumed when no scheme is given. The endpoint URL is the normalized address
plus the provider shape's path (e.g. ``/v1/chat/completions``).
"""


def normalize_address(address: str) -> str:
    """Return ``scheme://host[:port]`` for an address, assuming ``http://`` without a scheme."""
    base = address.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return base


def build_url(address: str, path: str) -> str:
    """Build an endpoint URL from an *address* and a shape *path* (e.g. ``/v1/responses``)."""
    return f"{normalize_address(address)}{path}"
