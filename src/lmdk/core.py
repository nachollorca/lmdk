"""Contains the main logic to call language model APIs."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal, overload

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, CompletionResponse, Message, UserMessage
from lmdk.errors import AllModelsFailedError
from lmdk.provider import load_provider
from lmdk.utils import parallelize_function

# @overload stubs let type checkers infer the return type of ``complete`` based on ``stream``


@overload
def complete(
    model: str | list[str],
    messages: Sequence[Message] | str,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    *,
    stream: Literal[True],
    generation_kwargs: dict | None = None,
) -> Iterator[str]: ...  # stream=True  -> yields tokens one by one


@overload
def complete(
    model: str | list[str],
    messages: Sequence[Message] | str,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: Literal[False] = False,
    generation_kwargs: dict | None = None,
) -> CompletionResponse: ...  # stream=False (default) -> complete response


def complete(
    model: str | list[str],
    messages: Sequence[Message] | str,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: bool = False,
    generation_kwargs: dict | None = None,
) -> CompletionResponse | Iterator[str]:
    """Generate a response from a language model.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``)
            or a list of identifiers to try in order as fallbacks.
        messages: The conversation history, or a plain string which is
            interpreted as a single user message.
        system_instruction: Optional system prompt prepended to the conversation.
        output_schema: Optional Pydantic model class for structured output.
            Mutually exclusive with *stream*.
        stream: If ``True``, return an iterator of content tokens instead of
            a complete ``ModelResponse``. Mutually exclusive with *output_schema*.
        generation_kwargs: Additional generation parameters forwarded to the
            provider (e.g. ``temperature``, ``max_tokens``).
            Defaults to ``{"temperature": 0}``.

    Returns:
        A ``CompletionResponse`` with the generated content and metadata, or an
        iterator of string tokens when *stream* is ``True``.

    Raises:
        AllModelsFailedError: If every model in the list fails.
    """
    # early stop
    if output_schema and stream:
        raise ValueError("Only `stream` or `output_schema` can be set, not both.")

    # set defaults and normalize overloaded params
    models = [model] if isinstance(model, str) else model
    if isinstance(messages, str):
        messages = [UserMessage(content=messages)]
    if generation_kwargs is None:
        generation_kwargs = {"temperature": 0}

    # fallback loop
    errors: dict[str, Exception] = {}
    for m in models:
        provider_name, model_id = m.split(":")
        provider = load_provider(name=provider_name)
        request = CompletionRequest(
            model_id=model_id,
            messages=messages,
            system_instruction=system_instruction,
            output_schema=output_schema,
            generation_kwargs=generation_kwargs,
        )
        try:
            # call provider and hopefully return response
            return provider.complete(request=request, stream=stream)
        except Exception as exc:
            errors[m] = exc

    # raise paricular error if one model or error summary if many
    if len(errors) == 1:
        raise next(iter(errors.values()))
    raise AllModelsFailedError(errors)


def complete_batch(
    model: str | list[str],
    messages_list: Sequence[Sequence[Message] | str],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    max_workers: int = 10,
) -> list[CompletionResponse | Exception]:
    """Generate responses for multiple conversations in parallel.

    Each conversation in *messages_list* is dispatched to :func:`complete`
    concurrently via a thread pool. Streaming is not supported in batch mode.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``)
            or a list of identifiers to try in order as fallbacks.
        messages_list: A list of conversations. Each element is either a
            message list or a plain string (interpreted as a single user message).
        system_instruction: Optional system prompt applied to every conversation.
        output_schema: Optional Pydantic model class for structured output.
        generation_kwargs: Additional generation parameters forwarded to the
            provider (e.g. ``temperature``, ``max_tokens``).
        max_workers: Maximum number of concurrent threads.

    Returns:
        A list with one entry per conversation, in the same order as
        *messages_list*.  Each entry is either a ``CompletionResponse`` on
        success or the ``Exception`` that was raised on failure.
    """
    shared_kwargs: dict[str, Any] = {
        "model": model,
        "system_instruction": system_instruction,
        "output_schema": output_schema,
        "stream": False,
        "generation_kwargs": generation_kwargs,
    }
    params_list = [{**shared_kwargs, "messages": messages} for messages in messages_list]

    return parallelize_function(
        function=complete,
        params_list=params_list,
        max_workers=max_workers,
        catch_exceptions=True,
    )
