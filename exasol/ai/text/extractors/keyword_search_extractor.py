from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.keyword_extractor import KeywordBase
from exasol.ai.text.extractors.text_search_extractor import TextSearchBase


class KeywordSearchExtractor(NLPExtractor, KeywordBase, TextSearchBase):
    """
    A keyword extractor combined with a text search extractor. The output should include
    the keywords and their locations in the text.
    """

    extractor_type: Literal["KeywordSearchExtractor"] = Field(
        default="KeywordSearchExtractor", repr=False
    )
