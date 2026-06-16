"""Tests for lmdk.providers._schema — strict JSON-Schema normalization."""

from pydantic import BaseModel

from lmdk.providers._schema import prepare_schema


class Person(BaseModel):
    name: str
    age: int


class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""


class Recipe(BaseModel):
    ingredients: list[Ingredient]


class TestPrepareSchema:
    def test_adds_strict_object_constraints(self):
        result = prepare_schema(Person.model_json_schema())
        assert result["additionalProperties"] is False
        assert result["required"] == ["name", "age"]

    def test_handles_nested_defs_and_defaults(self):
        result = prepare_schema(Recipe.model_json_schema())
        ingredient = result["$defs"]["Ingredient"]
        assert result["additionalProperties"] is False
        assert ingredient["additionalProperties"] is False
        assert ingredient["required"] == ["name", "quantity", "unit"]
        assert "default" not in ingredient["properties"]["unit"]

    def test_does_not_mutate_original_schema(self):
        schema = Recipe.model_json_schema()
        prepare_schema(schema)
        assert "additionalProperties" not in schema
        assert "default" in schema["$defs"]["Ingredient"]["properties"]["unit"]

    def test_handles_arrays_and_combinators(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "anyOf": [{"type": "object", "properties": {"name": {"type": "string"}}}]
                    },
                }
            },
        }
        result = prepare_schema(schema)
        nested = result["properties"]["items"]["items"]["anyOf"][0]
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["name"]
