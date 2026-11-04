from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    AfterValidator,
    Field,
)

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.extractors.default_models import DEFAULT_NLI_MODEL
from exasol.ai.text.extractors.hf_model import HuggingFaceModel


def topics_are_unique(topics: list[str]) -> list[str]:
    if len(topics) != len(set(topics)):
        raise ValueError("All topics in the list must be unique")
    return topics


class HftTopicSettings(TextAIBase):
    """
    A topic extractor using the ZeroShotClassificationPipeline from the
    Huggingface transformers library.

    Parameters:
        The descriptions of the parameters are copied from the Huggingface Transformers
        documentation.

        setting_type:
            This is an earmark of the algorithm or a particular setup for the algorithm
            where the settings are applicable. This field allows switching between
            different algorithms doing a similar job.
        topics:
            The list of possible class labels to classify each sequence into.
        hypothesis_template:
            The template used to turn each label into an NLI-style hypothesis. This
            template must include a {} or similar syntax for the candidate label to be
            inserted into the template. The default template is "This example is {}.".
            With the candidate label "sports", this would be fed into the model like
            "<cls> sequence to classify <sep> This example is sports . <sep>". The
            default template works well in many cases, but it may be worthwhile to
            experiment with different templates depending on the task setting.
        multi_label:
            Indicates if multiple candidate labels can be true. If False, the scores
            are normalized such that the sum of the label likelihoods for each sequence
            is 1. If True, the labels are considered independent and probabilities are
            normalized for each candidate by doing a softmax of the entailment score vs.
            the contradiction score. The default value is False.
    """

    setting_type: Literal["hft-topic-settings"] = "hft-topic-settings"
    ranking_model: HuggingFaceModel = HuggingFaceModel(name=DEFAULT_NLI_MODEL)
    topics: Annotated[list[str], AfterValidator(topics_are_unique)]
    hypothesis_template: str | None = None
    multi_label: bool = False


class TopicBase(TextAIBase):
    # In case of multiple arguments the `topic_settings` should be a
    # discriminating union.
    topic_settings: HftTopicSettings


class TopicClassifierExtractor(NLPExtractor, TopicBase):
    extractor_type: Literal["TopicClassifierExtractor"] = Field(
        default="TopicClassifierExtractor", repr=False
    )
