"""Base class for API-shape providers (``providers/shapes/``).

A *shape* is a wire format only: no default URL and no API key environment
variable. It is always addressed explicitly through the ``@<address>`` suffix
of the model identifier::

    chat:<model>@<host>[:<port>]

The endpoint URL is built with the shared address rule
(:func:`lmdk.providers._address.build_url`): ``host[:port]`` or
``scheme://host[:port]`` (``http://`` assumed) plus the shape's ``path``.

Authentication on a shape sends the optional ``LOCAL_API_KEY`` environment
variable in the shape's auth header format (see ``_key_headers``).

Vendors (``providers/vendors/``) subclass a shape and add a hosted default
location (``base_url``) and their own API key (``required_env``). For them the
``@`` suffix is optional: without it the hosted default is used and the vendor
key is sent; with it the address overrides the default and only the optional
``LOCAL_API_KEY`` is sent -- the vendor key is never sent to a user-supplied
address.
"""

import os

from lmdk.errors import ProviderError
from lmdk.provider import Provider
from lmdk.providers._address import build_url

# Optional API key sent to shapes and to user-supplied @<address> endpoints.
LOCAL_API_KEY_ENV = "LOCAL_API_KEY"


class ShapeProvider(Provider):
    """Base for wire-format shapes; requires ``@<address>`` unless ``base_url`` is set."""

    # Wire-format endpoint path appended to an address, e.g. ``/v1/chat/completions``.
    path: str = ""
    # Hosted default location. Shapes leave it empty (an @<address> is required);
    # vendors set it so the ``@`` suffix becomes optional.
    base_url: str = ""
    # Shapes need no credentials of their own; vendors set their API key env var.
    required_env: str | tuple[str, ...] = ()

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Split ``model_id`` into ``(model, location)``.

        An ``@<address>`` suffix always wins. Without it, vendors fall back to
        their hosted ``base_url``; a bare shape raises ``ProviderError`` because
        it has no default location.
        """
        address = cls._address_override(model_id)
        if address is not None:
            return model_id.rsplit("@", 1)[0], address
        if cls.base_url:
            return model_id, cls.base_url
        raise ProviderError(
            status_code=0,
            message=(
                f"{cls.__name__}: model must include an address as "
                f"'<model>@<host>[:<port>]' (got '{model_id}')."
            ),
            provider=cls.__name__,
        )

    @classmethod
    def _build_url(cls, location: str) -> str:
        """Build the endpoint URL: normalized *location* plus the shape's ``path``."""
        return build_url(location, cls.path)

    @classmethod
    def _key_headers(cls, api_key: str) -> dict:
        """Auth header format for an API key. ``Bearer`` unless the shape overrides it."""
        return {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def _local_key_headers(cls) -> dict:
        """Headers for the optional ``LOCAL_API_KEY``, in this provider's format."""
        api_key = os.getenv(LOCAL_API_KEY_ENV)
        return cls._key_headers(api_key) if api_key else {}

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Build the authentication headers for the current location.

        A vendor on its default location sends its own key (resolved from
        ``required_env`` into *credentials*). A shape, or a vendor with an
        ``@<address>`` override, sends the optional ``LOCAL_API_KEY`` in this
        provider's auth header format instead -- the vendor key is never sent
        to an address the user supplied.
        """
        if cls.required_env:
            env_vars = (
                (cls.required_env,) if isinstance(cls.required_env, str) else cls.required_env
            )
            key = next((credentials[var] for var in env_vars if var in credentials), None)
            if key is not None:
                return cls._key_headers(key)
        return cls._local_key_headers()
