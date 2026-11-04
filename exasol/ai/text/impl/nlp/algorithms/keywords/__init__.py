from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import (
    KeywordExtractor as KeywordExtractor,
)
from exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor import (
    PatternRankKeywordExtractor,
)

# All keyword extraction algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
KeywordExtractorModel = Annotated[
    PatternRankKeywordExtractor, Field(discriminator="algorithm_name")
]
