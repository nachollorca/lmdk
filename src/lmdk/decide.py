"""Simplified wrapper of the TypeSafe SDK."""

from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


@dataclass
class DecisionResponse:
    """The decision from the model with usage metadata."""

    input_tokens: int | None
    output_tokens: int | None
    probabilities: dict[str, dict[str, float]]
    """Root key is the name of the question, nested dict is keyed by the label name,
    among the possible ones, the float is the probability assigned to it.
    """


@dataclass
class Question:
    """A question for the model to decide upon.

    Attrs:
        text(str): the actual text of the question.
        labels(dict[str, str]): the possible labels to choose from, with optional descriptions.
        is_ordered(bool): signals if the given labales follow an ordinal scale.
    """

    text: str
    labels: dict[str, str | None]
    is_ordered: bool

    def to_typesafe(self) -> Noul | Choice | Score:
        """Maps our questions to the corresponding TypeSafe type."""
        return NotImplemented


def decide(
    context: str, questions: dict[str, Question], model: str = "typesafe:jev-latest"
) -> DecisionResponse:
    """Requests an encoder to output the probability of each label to answer the questions.

    Args:
        context(str): the context shared across all questions.
        questions(dict[str, Question]): named questions for the model to answer.
        model(str): the string id of the model, including the provider and optional base url.
    """
    # preprocess: map our questions to the type-safe accepted format
    questions = {k: v.to_typesafe() for k, v in questions.items()}

    # process: start client and make the request
    client = TypeSafeClient()
    response = client.system_one(state=context, questions=questions, model=model)

    # postprocess: map typesafe's answers to our format
    probabilities = {... for k, v in response.answers.items()}

    return DecisionResponse(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        probabilities=probabilities,
    )
