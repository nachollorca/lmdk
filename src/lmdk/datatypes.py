"""Contains the data contracts used across the app."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


@dataclass(frozen=True)
class Message:
    """Represents a single message in a conversation."""

    content: str
    role: str

    def to_dict(self) -> dict:
        """Converts the message to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class UserMessage(Message):
    """Wrapper for a message sent by the user."""

    role: str = "user"


@dataclass(frozen=True)
class AssistantMessage(Message):
    """Wrapper for a message sent by the assistant."""

    role: str = "assistant"


@dataclass(frozen=True)
class CompletionRequest:
    """Bundles the common parameters for a completion call.

    Built by ``lmdk.core.complete`` and threaded through the provider
    layer so that adding a new parameter is a single-field change here.
    """

    model_id: str
    prompt: Sequence[Message]
    system_instruction: str | None
    output_schema: type[BaseModel] | None
    generation_kwargs: dict


@dataclass(frozen=True)
class RawResponse:
    """Lightweight intermediate result returned by provider implementations.

    Carries the extracted content and token counts so the base class can
    handle timing, schema validation, and ``CompletionResponse`` construction.
    """

    content: str
    input_tokens: int
    output_tokens: int


T = TypeVar("T", bound=BaseModel | None)


@dataclass(frozen=True)
class CompletionResponse(RawResponse, Generic[T]):  # noqa: UP046
    """The result of a single completion call.

    You can hint the type of the expected object in ``.parsed`` field using annotation
    ``ParsedResponse[MyPydanticModel]``.

    Attributes:
        content: The raw string response from the LLM.
        input_tokens: The number of tokens consumed in the input/prompt.
        output_tokens: The number of tokens generated in the response.
        latency: The time in seconds taken to generate the response.
        parsed: Optional parsed structured output as a BaseModel instance, or None
            if no output schema was specified.
    """

    latency: float = 0.0
    parsed: T | None = None

    @property
    def message(self) -> AssistantMessage:
        """Converts the response to an AssistantMessage object."""
        return AssistantMessage(self.content)

    @property
    def output(self) -> Any:
        """The most useful representation of the response output.

        - If there is no parsed structured output, returns the string content.
        - If ``parsed`` is a BaseModel with more than one field, returns the model itself.
        - If ``parsed`` is a BaseModel with exactly one field, returns that field's value.

        The single-field unwrapping is useful when a schema is used purely to coerce a
        scalar/list value (e.g. a summary string or a list of segments).
        """
        if self.parsed is None:
            return self.content

        assert isinstance(self.parsed, BaseModel)
        fields = type(self.parsed).model_fields

        if len(fields) == 1:
            field_name = next(iter(fields.keys()))
            return getattr(self.parsed, field_name)

        return self.parsed


@dataclass(frozen=True)
class CompletionBatch:
    """An aggregate of multiple :class:`CompletionResponse` objects.

    Use this when you fan out a prompt over many inputs (e.g. via
    ``complete_batch``) and want a single object that summarises token usage,
    latency and parsed outputs across the batch.

    Attributes:
        responses: The individual responses that make up the batch.
    """

    responses: list[CompletionResponse] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        """Total input tokens across all responses."""
        return sum(r.input_tokens for r in self.responses)

    @property
    def output_tokens(self) -> int:
        """Total output tokens across all responses."""
        return sum(r.output_tokens for r in self.responses)

    @property
    def latency(self) -> float:
        """The slowest response's latency (the batch is bounded by its tail)."""
        return max((r.latency for r in self.responses), default=0.0)

    @property
    def parsed(self) -> list[BaseModel]:
        """The parsed outputs of each response, skipping responses without one."""
        return [r.parsed for r in self.responses if r.parsed is not None]

    @property
    def output(self) -> list[Any]:
        """The per-response ``.output`` values, flattened if they are all lists.

        Each response's :pyattr:`CompletionResponse.output` is computed individually
        (so single-field BaseModels are unwrapped). If every resulting value is a
        list, the lists are concatenated into one flat list; otherwise the list of
        per-response outputs is returned as-is.
        """
        if not self.responses:
            return []

        individual = [r.output for r in self.responses]

        if individual and all(isinstance(out, list) for out in individual):
            return list(chain.from_iterable(individual))

        return individual
