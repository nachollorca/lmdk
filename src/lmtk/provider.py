"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import Message, ModelResponse
from lmtk.secrets import resolve_api_key


class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define the following class attributes:

        model_ids: A list of supported model identifiers.
            This is just for the selector of the TUI, nothing is enforced.
            You can try to run a model that is not in the list.
        api_key_name: The environment variable name for the provider's API key.

    The base class handles credential resolution and validation.
    Concrete providers only implement ``_get_response`` and ``_stream``,
    receiving the API key as a parameter.
    """

    model_ids: list[str]
    api_key_name: str

    @classmethod
    def get_response(
        cls,
        model_id: str,
        messages: list[Message],
        system_instruction: str | None = None,
        output_schema: type[BaseModel] | None = None,
        stream: bool = False,
        generation_kwargs: dict | None = None,
    ) -> ModelResponse | Iterator[str]:
        """Generate a response from the provider.

        Resolves API credentials and delegates to the provider implementation.

        Args:
            model_id: The model identifier (e.g. ``"devstral-latest"``).
            messages: The conversation history.
            system_instruction: Optional system prompt.
            output_schema: Optional Pydantic model class for structured output.
            stream: Whether to stream the response.
            generation_kwargs: Additional generation parameters.
                Defaults to ``{"temperature": 0}``.

        Returns:
            A ModelResponse with the generated content and metadata.
        """
        # ensure the options selected are possible
        if stream and output_schema:
            raise ValueError("Only one between `stream` and `output_schema` can be active.")

        # resolve key
        api_key = resolve_api_key(cls.api_key_name)
        if not api_key:
            raise ValueError(f"{cls.api_key_name} not found in .conf/lmtk/secrets.yaml or env vars")

        # this pattern avoids mutable default arg
        if generation_kwargs is None:
            generation_kwargs = {"temperature": 0}

        # call child implementation
        params = dict(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            system_instruction=system_instruction,
            output_schema=output_schema,
            generation_kwargs=generation_kwargs,
        )
        if stream:
            return cls._stream(**params)  # type: ignore[arg-type]
        return cls._get_response(**params)  # type: ignore[arg-type]

    @classmethod
    @abstractmethod
    def _get_response(
        cls,
        model_id: str,
        messages: list[Message],
        api_key: str,
        system_instruction: str | None,
        output_schema: type[BaseModel] | None,
        generation_kwargs: dict,
    ) -> ModelResponse:
        """Provider-specific response generation. See ``get_response`` for details."""
        ...

    @classmethod
    @abstractmethod
    def _stream(
        cls,
        model_id: str,
        messages: list[Message],
        api_key: str,
        system_instruction: str | None,
        output_schema: type[BaseModel] | None,
        generation_kwargs: dict,
    ) -> Iterator[str]:
        """Stream chat completion tokens from the provider.

        Yields:
            Individual content tokens as they arrive.
        """
        ...
