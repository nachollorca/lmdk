"""Contains the data contracts used across the app."""

from dataclasses import asdict, dataclass

from pydantic import BaseModel

# TODO: see if this module could actually be merged into core.py


@dataclass
class Message:
    content: str
    role: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UserMessage(Message):
    role: str = "user"


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"


@dataclass
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency: float
    object: BaseModel | None = None

    @property
    def message(self):
        return AssistantMessage(self.content)
