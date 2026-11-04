from typing import cast

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.extractors.source_table_extractor import SourceTableExtractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.graph_builders.utils import (
    create_log_span_for_extractor,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table_sql_stage import (
    SourceTableSQLStage,
)
from exasol.ai.text.impl.utils.stage_graph_utils import get_output_table


class SourceTableGraphBuilder:
    """
    Builds a single-node graph for the source table extractor.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:
        if not isinstance(extractor, SourceTableExtractor):
            raise TypeError(
                "Wrong type of extractor. "
                f"Expected {SourceTableExtractor.__name__}. "
                f"Got {type(extractor).__name__}."
            )

        document_table = get_output_table(extractor, context)
        log_span = create_log_span_for_extractor(extractor)
        stage = SourceTableSQLStage(
            sources=cast(SourceTableExtractor, extractor).sources,
            document_table=document_table,
            log_span=log_span,
        )
        return SQLStageGraph(stage, stage, set())
