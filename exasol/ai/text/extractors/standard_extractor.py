from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    AfterValidator,
    Field,
)

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.default_models import (
    DEFAULT_FEATURE_EXTRACTION_MODEL,
    DEFAULT_NAMED_ENTITY_MODEL,
    DEFAULT_NLI_MODEL,
)
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.extractors.topic_classifier_extractor import topics_are_unique


class StandardExtractor(NLPExtractor):
    extractor_type: Literal["StandardExtractor"] = Field(
        default="StandardExtractor", repr=False
    )
    named_entity_recognition_model: HuggingFaceModel | None = HuggingFaceModel(
        name=DEFAULT_NAMED_ENTITY_MODEL
    )
    topic_classification_model: HuggingFaceModel | None = HuggingFaceModel(
        name=DEFAULT_NLI_MODEL
    )
    keyword_search_model: HuggingFaceModel | None = HuggingFaceModel(
        name=DEFAULT_FEATURE_EXTRACTION_MODEL
    )
    topics: Annotated[list[str], AfterValidator(topics_are_unique)] | None = None
    entity_db_output: str | None = None
    topic_db_output: str | None = None
    keyword_db_output: str | None = None
    tokenizer_db_output: str | None = None
    entity_tokenized_db_output: str | None = None
    keyword_tokenized_db_output: str | None = None
