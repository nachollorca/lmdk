"""Contains the main logic to call language model APIs."""

from pydantic import BaseModel

from lmtk.datatypes import Message, ModelResponse


# TODO: think if this should be get_response or generate_response
def get_response(
    model: str,
    messages: list[Message],
    system_instruction: str | None = None,
    output_schema: BaseModel | None = None,
    stream: bool = False,
    generation_kargs: dict = {"temperature": 0},
) -> ModelResponse:  # type: ignore
    """Docstring."""
    if output_schema and stream:
        raise RuntimeError(f"Only `stream` or `output_schema` can be set, not both.")
    ...


class Chat:
    """Docstring."""

    def __init__(
        self,
        model: str,
        system_instruction: str,
        output_schema: BaseModel | None = None,
        stream=True,
        generation_kwargs: dict = {"temperature": 0},
    ):
        """Docstring."""
        self.model = model
        self.system_instruction = system_instruction
        self.output_schema = output_schema
        self.stream = stream
        self.generation_kargs = generation_kwargs
        self.messages: list[Message] = []

    def get_response(self, message: Message) -> ModelResponse:
        """Docstring."""
        self.messages.append(message)
        response = get_response(
            model=self.model,
            messages=self.messages,
            system_instruction=self.system_instruction,
            stream=self.stream,
            generation_kargs=self.generation_kargs,
        )
        self.messages.append(response.message)
        return response
