"""JSON-Schema normalization shared by OpenAI-style strict structured output.

OpenAI's Responses API and Mistral's chat-completions ``response_format`` both
require the same JSON-Schema subset for strict structured output:

* every object node must declare ``additionalProperties: false``
* every object node must list all of its properties in ``required``
* ``default`` keys (which Pydantic emits for fields with defaults) are not
  part of the supported subset and must be stripped

This module is private to the ``lmdk.providers`` package.
"""

from copy import deepcopy
from typing import Any

_SCHEMA_COMBINATORS = ("anyOf", "oneOf", "allOf")


def prepare_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Pydantic JSON schema for OpenAI-style strict structured output.

    The input schema is deep-copied and never mutated.
    """
    prepared = deepcopy(schema)
    _prepare_schema_in_place(prepared)
    return prepared


def _prepare_schema_in_place(node: Any) -> None:
    """Recursively normalize a JSON Schema node in place."""
    if not isinstance(node, dict):
        return

    node.pop("default", None)

    _prepare_object_schema(node)
    _prepare_recursive_schemas(node)


def _prepare_object_schema(node: dict[str, Any]) -> None:
    """Handle object-specific JSON Schema normalization."""
    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
        node["additionalProperties"] = False
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            for prop in properties.values():
                _prepare_schema_in_place(prop)


def _prepare_recursive_schemas(node: dict[str, Any]) -> None:
    """Handle recursive JSON Schema structures."""
    if "$defs" in node:
        for definition in node["$defs"].values():
            _prepare_schema_in_place(definition)

    if "items" in node:
        _prepare_schema_in_place(node["items"])

    additional_properties = node.get("additionalProperties")
    if isinstance(additional_properties, dict):
        _prepare_schema_in_place(additional_properties)

    for key in _SCHEMA_COMBINATORS:
        for option in node.get(key, []) or []:
            _prepare_schema_in_place(option)
