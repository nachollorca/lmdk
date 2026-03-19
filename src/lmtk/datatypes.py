"""Contains the data contracts used across the app."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import chain
from typing import Any

from pydantic import BaseModel


@dataclass
class Message:
    """Represents a single message in a conversation."""

    content: str
    role: str

    def to_dict(self) -> dict:
        """Converts the message to a dictionary."""
        return asdict(self)


@dataclass
class UserMessage(Message):
    """Wrapper for a message sent by the user."""

    role: str = "user"


@dataclass
class AssistantMessage(Message):
    """Wrapper for a message sent by the assistant."""

    role: str = "assistant"


@dataclass(frozen=True)
class CompletionRequest:
    """Bundles the common parameters for a completion call.

    Built by ``lmtk.core.get_response`` and threaded through the provider
    layer so that adding a new parameter is a single-field change here.
    """

    model_id: str
    messages: Sequence[Message]
    system_instruction: str | None
    output_schema: type[BaseModel] | None
    generation_kwargs: dict


@dataclass
class RawResponse:
    """Lightweight intermediate result returned by provider implementations.

    Carries the extracted content and token counts so the base class can
    handle timing, schema validation, and ``CompletionResponse`` construction.
    """

    content: str
    input_tokens: int
    output_tokens: int


@dataclass
class CompletionResponse(RawResponse):
    """The result of a completion call, including usage and parsed pydantic objects.

    Attributes:
        content: The raw string response from the LLM.
        input_tokens: The number of tokens consumed in the input/prompt.
        output_tokens: The number of tokens generated in the response.
        latency: The time in seconds taken to generate the response.
        parsed: Optional parsed structured output as a BaseModel instance, list of models, or None
            if no output schema was specified.
    """

    latency: float = 0.0
    parsed: BaseModel | list | None = None

    @property
    def message(self):
        """Converts the response to an AssistantMessage object."""
        return AssistantMessage(self.content)

    @property
    def output(self):
        """The most useful representation of the response output.

        This property intelligently extracts the most relevant part of the response:
        - For single responses with no parsed output: returns the string content
        - For single responses with a multi-field BaseModel: returns the entire model
        - For single responses with a single-field BaseModel: returns just that field's value
        - For aggregated responses (created via from_list): returns processed list of outputs

        This handles common patterns where a BaseModel wraps a single value (e.g., a summary
        string or a list of segments) for schema validation, automatically unwrapping such
        single-field models to make the output more convenient to work with.

        Returns:
            Any: The extracted output - varies based on response structure and whether it's
                aggregated. Could be a string, BaseModel, list, or any other type.
        """
        if isinstance(self.parsed, list):
            return self._output_for_aggregated_responses()
        return self._output_for_single_response()

    @classmethod
    def from_list(cls, responses: list["CompletionResponse"]) -> "CompletionResponse":
        """Aggregates multiple completion responses into a single response object.

        Combines multiple responses by:
        - Collecting all parsed outputs into a list
        - Summing input and output tokens from all responses
        - Using the maximum latency value from all responses

        Args:
            responses: A list of CompletionResponse objects to aggregate.

        Returns:
            CompletionResponse: An aggregated response with empty content, combined token counts,
                max latency, and all parsed outputs collected in a list.
        """
        aggregated_response: CompletionResponse = CompletionResponse(
            content="", input_tokens=0, output_tokens=0, latency=0, parsed=[]
        )

        for response in responses:
            if isinstance(response, CompletionResponse):
                assert isinstance(aggregated_response.parsed, list)
                if response.parsed is not None and aggregated_response.parsed is not None:
                    aggregated_response.parsed.append(response.parsed)

                # update stats
                aggregated_response.input_tokens += response.input_tokens
                aggregated_response.output_tokens += response.output_tokens
                aggregated_response.latency = (
                    response.latency
                    if aggregated_response.latency < response.latency
                    else aggregated_response.latency
                )

        return aggregated_response

    def _output_for_single_response(self) -> Any:
        """Extracts the output for a single response object.

        Returns the most useful representation of the parsed output:
        - If there is no parsed structured output, returns the string content.
        - If there is a parsed object with more than one field, returns the entire object.
        - If there is a parsed object with just one field, returns the content of that field.

        This unwrapping behavior is useful for cases where a BaseModel wraps a single value
        (e.g., a summary string or a list of segments) for schema validation purposes.

        Returns:
            Any: The extracted output - either the content string, a BaseModel, or the value
                of a single field from a BaseModel.
        """
        if self.parsed is None:
            return self.content

        assert isinstance(self.parsed, BaseModel)
        fields = type(self.parsed).model_fields

        if len(fields) == 1:
            field_name = next(iter(fields.keys()))
            return getattr(self.parsed, field_name)

        return self.parsed

    def _output_for_aggregated_responses(self) -> Any:
        """Extracts the output for aggregated responses (when parsed is a list from from_list).

        Processes a list of parsed outputs by:
        1. Computing individual outputs using the unwrapping logic in _output_for_single_response
        2. Flattening the results if all individual outputs are lists

        This is useful when aggregating multiple responses where each contains a list of items,
        allowing you to get a single flat list rather than a list of lists.

        Returns:
            Any: An empty list if parsed is empty. Otherwise, either the individual outputs
                (possibly flattened if all are lists) or a single flattened list if all
                individual outputs are lists.
        """
        if not self.parsed:
            return []

        # Compute what each individual response.output would be
        individual_outputs = []
        for obj in self.parsed:
            if isinstance(obj, BaseModel) and len(fields := type(obj).model_fields) == 1:
                # Single field: unwrap it
                field_name = next(iter(fields.keys()))
                individual_outputs.append(getattr(obj, field_name))

            else:
                individual_outputs.append(obj)

        # Flatten if all individual outputs are lists
        if individual_outputs and all(isinstance(out, list) for out in individual_outputs):
            return list(chain.from_iterable(individual_outputs))

        return individual_outputs
