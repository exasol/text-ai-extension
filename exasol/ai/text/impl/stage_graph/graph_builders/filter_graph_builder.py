from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.extractors.filter_extractor import FilterExtractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.filter_query_handler import (
    FilterSqlStage,
)


class FilterGraphBuilder:
    """
    Stage graph builder for the Filter Extractor.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:

        if not isinstance(extractor, FilterExtractor):
            raise ValueError(
                "Wrong type of extractor for the Filter Graph Builder. "
                f"Expected {FilterExtractor.__name__}. "
                f"Got {type(extractor).__name__}."
            )

        stage = FilterSqlStage(extractor.allowed_names)
        return SQLStageGraph(stage, stage, set())
