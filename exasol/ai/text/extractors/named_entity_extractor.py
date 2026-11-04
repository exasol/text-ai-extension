from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.extractors.default_models import DEFAULT_NAMED_ENTITY_MODEL
from exasol.ai.text.extractors.hf_model import HuggingFaceModel


class HftNamedEntitySettings(TextAIBase):
    """
    A named entity extractor using the TokenClassificationPipeline from the
    Huggingface transformers library.

    Parameters:
        The descriptions of some of the parameters are copied from the Huggingface
        Transformers documentation.

        setting_type:
            This is an earmark of the algorithm or a particular setup for the algorithm
            where the settings are applicable. This field allows switching between
            different algorithms doing a similar job.
        ignore_labels:
            A list of labels to ignore.
        aggregation_strategy:
            The strategy to fuse (or not) tokens based on the model prediction.
            “none”:
                Will simply not do any aggregation and simply return raw results from
                the model
            “simple”:
                Will attempt to group entities following the default schema. (A, B-TAG),
                (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being
                [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”},
                {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will
                end up as different entities. On word based languages, we might end up
                splitting words undesirably : Imagine Microsoft being tagged as
                [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}].
                Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate
                words (on languages that support that meaning, which is basically tokens
                separated by a space). These mitigations will only work on real words,
                “New york” might still be tagged with two different entities.
            “first”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. Words will simply use the
                tag of the first token of the word when there is ambiguity.
            “average”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. scores will be averaged
                first across tokens, and then the maximum label is applied.
            “max”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. Word entity will simply
                be the token with the maximum score.

            Defaults to "simple".
    """

    setting_type: Literal["hft-named-entity-settings"] = "hft-named-entity-settings"
    model: HuggingFaceModel = HuggingFaceModel(name=DEFAULT_NAMED_ENTITY_MODEL)
    ignore_labels: list[str] | None = None
    aggregation_strategy: str = "simple"


class NamedEntityBase(BaseModel):
    model_config = {"frozen": True}

    # In case of multiple arguments the `named_entity_settings` should be a
    # discriminating union.
    named_entity_settings: HftNamedEntitySettings


class NamedEntityExtractor(NLPExtractor, NamedEntityBase):
    extractor_type: Literal["NamedEntityExtractor"] = Field(
        default="NamedEntityExtractor", repr=False
    )
