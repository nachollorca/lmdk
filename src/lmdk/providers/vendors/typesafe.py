"""Vendor for TypeSafe's System One API (hosted Jev models).

Subclasses the ``systemone`` shape with the hosted default location and the
``TYPESAFE_API_KEY`` credential; an ``@<address>`` suffix overrides the
default location (then only the optional ``LOCAL_API_KEY`` is sent as a
``Bearer`` token, never the vendor key)::

    decide("typesafe:jev-latest", state=..., questions=...)

See https://docs.typesafe.ai/api.
"""

from lmdk.providers.shapes.systemone import SystemoneProvider

TYPESAFE_BASE_URL = "https://api.typesafe.ai"


class TypesafeProvider(SystemoneProvider):
    """Provider for TypeSafe's hosted System One models (e.g. ``typesafe:jev-latest``)."""

    required_env = "TYPESAFE_API_KEY"
    base_url = TYPESAFE_BASE_URL
