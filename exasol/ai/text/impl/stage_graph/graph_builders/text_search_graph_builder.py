from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.text_search_extractor import TextSearchBase
from exasol.ai.text.impl.db_schema import (
    IOColumnFlags,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.text_search.text_search import (
    TextSearch as TextSearchAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.text_search import (
    TextSearchInput,
    TextSearchOutput,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch as TokenizedTextSearchAlgorithm,
)
from exasol.ai.text.impl.stage_graph.graph_builders import ExtractorContext
from exasol.ai.text.impl.stage_graph.graph_builders.nlp_graph_builder import (
    NLPGraphBuilder,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec


def algorithm_from_extractor(
    extractor: NLPExtractor,
    phrase_column_name: str = "PHRASE",
    phrase_column_flags: IOColumnFlags = IOColumnFlags.NONE,
) -> TextSearchAlgorithm:
    """
    Creates TextSearchAlgorithm from a NLPExtractor
    Args:
        extractor:
            NLP Extractor
        phrase_column_name:
            Name of the phrase column. This will be set for both input and output.
        phrase_column_flags:
            IOColumnFlags for the phrase output column. The text search algorithm can
            be used in combinations with other algorithms. In different scenarios we may
            want to use different column flags for the searched phrase. However, within
            the same scenario (i.e. same assembly of algorithms) the flag should be
            consistent between subsequent extractions. Therefore, it cannot be provided
            in the extractor settings. Instead, it should be explicitly set in the call
            to this function.

    Returns:TextSearchAlgorithm
    """
    if isinstance(extractor, TextSearchBase):
        # At the moment there is only one text search algorithm
        # in use. Once there is a choice of such algorithms this code should
        # switch on the discriminator of the `text_search_settings`.
        ts_settings = cast(TextSearchBase, extractor).text_search_settings
        phrase_spec = varchar_column_spec(phrase_column_name, flags=phrase_column_flags)
        return TokenizedTextSearchAlgorithm(
            fuzziness=ts_settings.fuzziness,
            ignore_space=ts_settings.ignore_spaces,
            inputs=TextSearchInput(phrase=phrase_spec),
            outputs=TextSearchOutput(
                name=extractor.name,
                phrase=phrase_spec,
                phrase_span=SpanColumns.create(phrase_column_name),
            ),
        )
    raise TypeError(
        f"The extractor of type {type(extractor)} does not "
        f"inherit from {TextSearchBase.__name__}."
    )


class TextSearchGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for a text search extractor.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        algorithm = algorithm_from_extractor(
            extractor, phrase_column_flags=IOColumnFlags.Normalisation
        )
        return cast(AlgorithmModel, algorithm)
