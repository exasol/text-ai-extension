from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.extractors.splitter_extractor import SplitterExtractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.splitter_query_handler import (
    SplitterSqlStage,
)


class SplitterGraphBuilder:
    """
    Stage graph builder for the Splitter Extractor.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:

        if not isinstance(extractor, SplitterExtractor):
            raise ValueError(
                "Wrong type of extractor for the Splitter Graph Builder. "
                f"Expected {SplitterExtractor.__name__}. "
                f"Got {type(extractor).__name__}."
            )

        stage = SplitterSqlStage(extractor.new_names)
        return SQLStageGraph(stage, stage, set())
