from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.base_model import TextAIBase


class SpacyTokenizerSettings(TextAIBase):
    """
    A token extractor using the Spacy Tokenizer.
    """

    setting_type: Literal["spacy-tokenizer-settings"] = "spacy-tokenizer-settings"
    # Currently we don't allow setting the model name, since we don't have any
    # means of installing a new SpaCy model into the SLC or the Bucket-FS.
    # model_name: str | None = None


class TokenizerBase(TextAIBase):
    pass

    # In case of multiple arguments the `tokenizer_settings` should be a
    # discriminating union.
    tokenizer_settings: SpacyTokenizerSettings


class TokenizerExtractor(NLPExtractor, TokenizerBase):
    extractor_type: Literal["TokenizerExtractor"] = Field(
        default="TokenizerExtractor", repr=False
    )
