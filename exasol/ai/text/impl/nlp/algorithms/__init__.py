from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.keyword_search import KeywordSearchModel
from exasol.ai.text.impl.nlp.algorithms.keywords import KeywordExtractorModel
from exasol.ai.text.impl.nlp.algorithms.named_entity import NamedEntityRecognitionModel
from exasol.ai.text.impl.nlp.algorithms.testing import TestingModel
from exasol.ai.text.impl.nlp.algorithms.text_search import TextSearchModel
from exasol.ai.text.impl.nlp.algorithms.tokenizer import TokenizerModel
from exasol.ai.text.impl.nlp.algorithms.topics import TopicClassificationModel

# All types of algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
AlgorithmModel = Annotated[
    (
        TestingModel
        | TextSearchModel
        | TokenizerModel
        | KeywordExtractorModel
        | KeywordSearchModel
        | NamedEntityRecognitionModel
        | TopicClassificationModel
    ),
    Field(discriminator="algorithm_name"),
]
