"""Provider for self-hosted Laya servers (``laya-serve``).

Laya speaks TypeSafe's System One wire protocol, so this reuses
:class:`~lmdk.providers.typesafe.TypesafeProvider` and only changes where the
request goes and how it authenticates:

    ``laya:<model>@<host>[:<port>]``

``<model>`` is a Laya checkpoint (``english``, ``multilingual``,
``typed-decisions``); any other value lets the server's router pick one. The
location is mandatory and defaults to ``http://`` when it has no scheme. Servers
started with ``LAYA_API_KEY`` expect it as a ``Bearer`` token; set the same
variable on the client.

See https://github.com/NandhaKishorM/laya#self-hosting-http-server-jev-compatible.
"""

import os

from lmdk.errors import ProviderError
from lmdk.providers.typesafe import TypesafeProvider

_API_KEY_ENV = "LAYA_API_KEY"


class LayaProvider(TypesafeProvider):
    """Provider for a Laya server at the ``@location`` of the model identifier."""

    required_env = ()

    @classmethod
    def _build_auth_headers(cls, credentials: dict[str, str]) -> dict:
        """Return a Bearer header when ``LAYA_API_KEY`` is set, else no auth."""
        api_key = os.getenv(_API_KEY_ENV)
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @classmethod
    def _parse_model_id(cls, model_id: str) -> tuple[str, str]:
        """Split ``"<model>@<host>[:<port>]"`` into ``(model, base_url)``."""
        if "@" not in model_id:
            raise ProviderError(
                status_code=0,
                message=(
                    f"{cls.__name__}: model must include an endpoint as "
                    f"'<model>@<host>[:<port>]' (got '{model_id}')."
                ),
                provider=cls.__name__,
            )
        model, location = model_id.rsplit("@", 1)
        base_url = location.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        return model, base_url
