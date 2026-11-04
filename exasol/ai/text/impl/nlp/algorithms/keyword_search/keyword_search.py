from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.keywords import KeywordExtractorModel
from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import KeywordOutput
from exasol.ai.text.impl.nlp.algorithms.text_search import TextSearchModel
from exasol.ai.text.impl.nlp.algorithms.udf_algo_pipeline import (
    UDFAlgoPipeline,
    map_fields,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput


@dataclass(frozen=True)
class KeywordSearchOutput(KeywordOutput):
    name: str = "KEYWORD_SEARCH"

    keyword_span: SpanColumns = SpanColumns.create("KEYWORD")
    """
    The character span of an extracted keyword. The span is relative to the source
    document, not to the text chunk the keyword was extracted from.
    """


class KeywordSearch(UDFAlgoPipeline):
    """
    This is a combination of a keyword extraction and text search algorithms. The latter
    adds the character spans of the occurrences of the keyword in the original text.

    algorithm discriminator: keyword-search-pipeline

    Parameters:
        algorithms: The pair of algorithms.
    """

    inputs: TextInput = TextInput()
    outputs: KeywordSearchOutput = KeywordSearchOutput()
    algorithms: tuple[KeywordExtractorModel, TextSearchModel]

    algorithm_name: Literal["keyword-search-pipeline"] = Field(
        "keyword-search-pipeline", repr=False
    )

    def map_io(self) -> None:
        keyword_extractor, text_search = self.algorithms
        text_search.inputs = map_fields(
            text_search.inputs, {"phrase": keyword_extractor.outputs.keyword}
        )
        text_search.outputs = map_fields(
            text_search.outputs,
            {"phrase": self.outputs.keyword, "phrase_span": self.outputs.keyword_span},
        )
