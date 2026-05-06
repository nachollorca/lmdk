"""Contains the main logic to call language model APIs."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal, overload

from pydantic import BaseModel

from lmdk.datatypes import CompletionRequest, CompletionResponse, Message, UserMessage
from lmdk.errors import AllModelsFailedError
from lmdk.provider import load_provider
from lmdk.telemetry import traced_completion
from lmdk.utils import parallelize_function

# @overload stubs let type checkers infer the return type of ``complete`` based on ``stream``


@overload
def complete(
    model: str | list[str],
    prompt: str | Sequence[Message],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    *,
    stream: Literal[True],
    generation_kwargs: dict | None = None,
    return_request: bool = False,
) -> Iterator[str]: ...  # stream=True  -> yields tokens one by one


@overload
def complete(
    model: str | list[str],
    prompt: str | Sequence[Message],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: Literal[False] = False,
    generation_kwargs: dict | None = None,
    return_request: bool = False,
) -> CompletionResponse: ...  # stream=False (default) -> complete response


def complete(
    model: str | list[str],
    prompt: str | Sequence[Message],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: bool = False,
    generation_kwargs: dict | None = None,
    return_request: bool = False,
) -> CompletionResponse | Iterator[str]:
    """Generate a response from a language model.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``)
            or a list of identifiers to try in order as fallbacks.
        prompt: The string to complete or a conversation history as a list of messages.
        system_instruction: Optional system prompt prepended to the conversation.
        output_schema: Optional Pydantic model class for structured output.
            Mutually exclusive with *stream*.
        stream: If ``True``, return an iterator of content tokens instead of
            a complete ``ModelResponse``. Mutually exclusive with *output_schema*.
        generation_kwargs: Additional generation parameters forwarded to the
            provider (e.g. ``temperature``, ``max_tokens``).
            Defaults to ``{"temperature": 0}``.
        return_request: If ``True``, attaches the generated ``CompletionRequest`` to the
            returned ``CompletionResponse``. Ignored if *stream* is ``True``.

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
    prompt = _normalize_prompt(prompt)
    generation_kwargs = _default_generation_kwargs(generation_kwargs)

    # fallback loop
    errors: dict[str, Exception] = {}
    for i, m in enumerate(models):
        try:
            return _complete_model(
                model=m,
                prompt=prompt,
                system_instruction=system_instruction,
                output_schema=output_schema,
                stream=stream,
                generation_kwargs=generation_kwargs,
                return_request=return_request,
                fallback_index=i,
            )
        except Exception as exc:
            errors[m] = exc

    # raise particular error if one model or error summary if many
    if len(errors) == 1:
        raise next(iter(errors.values()))
    raise AllModelsFailedError(errors)


def _normalize_prompt(prompt: str | Sequence[Message]) -> Sequence[Message]:
    if isinstance(prompt, str):
        return [UserMessage(content=prompt)]
    return prompt


def _default_generation_kwargs(generation_kwargs: dict | None) -> dict:
    if generation_kwargs is None:
        return {"temperature": 0}
    return generation_kwargs


def _complete_model(
    *,
    model: str,
    prompt: Sequence[Message],
    system_instruction: str | None,
    output_schema: type[BaseModel] | None,
    stream: bool,
    generation_kwargs: dict,
    return_request: bool,
    fallback_index: int,
) -> CompletionResponse | Iterator[str]:
    provider_name, model_id = model.split(":", maxsplit=1)
    provider = load_provider(name=provider_name)
    request = CompletionRequest(
        model_id=model_id,
        prompt=prompt,
        system_instruction=system_instruction,
        output_schema=output_schema,
        generation_kwargs=generation_kwargs,
    )

    if stream:
        return provider.complete(request=request, stream=True)

    with traced_completion(
        provider_name, model_id, request, fallback_index=fallback_index
    ) as telemetry:
        response = provider.complete(request=request, stream=False)
        if isinstance(response, CompletionResponse):
            telemetry.record_response(response)
            if return_request:
                response.request = request
        return response


def complete_batch(
    model: str | list[str],
    prompt_list: Sequence[str | Sequence[Message]],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    max_workers: int = 10,
    return_request: bool = False,
) -> list[CompletionResponse | Exception]:
    """Generate responses for multiple conversations in parallel.

    Each conversation in *prompt_list* is dispatched to :func:`complete`
    concurrently via a thread pool. Streaming is not supported in batch mode.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``)
            or a list of identifiers to try in order as fallbacks.
        prompt_list: A list of conversations. Each element is either a
            message list or a plain string (interpreted as a single user message).
        system_instruction: Optional system prompt applied to every conversation.
        output_schema: Optional Pydantic model class for structured output.
        generation_kwargs: Additional generation parameters forwarded to the
            provider (e.g. ``temperature``, ``max_tokens``).
        max_workers: Maximum number of concurrent threads.
        return_request: If ``True``, attaches the generated ``CompletionRequest`` to each
            returned ``CompletionResponse``.

    Returns:
        A list with one entry per conversation, in the same order as
        *prompt_list*.  Each entry is either a ``CompletionResponse`` on
        success or the ``Exception`` that was raised on failure.
    """
    shared_kwargs: dict[str, Any] = {
        "model": model,
        "system_instruction": system_instruction,
        "output_schema": output_schema,
        "stream": False,
        "generation_kwargs": generation_kwargs,
        "return_request": return_request,
    }
    params_list = [{**shared_kwargs, "prompt": prompt} for prompt in prompt_list]

    return parallelize_function(
        function=complete,
        params_list=params_list,
        max_workers=max_workers,
        catch_exceptions=True,
    )
