from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.impl.db_schema import IOColumnFlags
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearch,
    KeywordSearchOutput,
)
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.keyword_graph_builder import (
    algorithm_from_extractor as keyword_algorithm_factory,
)
from exasol.ai.text.impl.stage_graph.graph_builders.nlp_graph_builder import (
    NLPGraphBuilder,
)
from exasol.ai.text.impl.stage_graph.graph_builders.text_search_graph_builder import (
    algorithm_from_extractor as text_search_algorith_factory,
)


class KeywordSearchGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for an extractor that is a combination of a keyword
    extraction with a subsequent text search.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        keyword_algorithm = keyword_algorithm_factory(extractor, context)
        search_algorithm = text_search_algorith_factory(
            extractor,
            phrase_column_name=keyword_algorithm.keyword_column,
            phrase_column_flags=IOColumnFlags.Normalisation,
        )
        kw_search_algorithm = KeywordSearch(
            algorithms=(keyword_algorithm, search_algorithm),
            outputs=KeywordSearchOutput(name=extractor.name),
        )
        return cast(AlgorithmModel, kw_search_algorithm)
