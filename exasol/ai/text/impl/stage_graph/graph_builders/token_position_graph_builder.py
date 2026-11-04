from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors import TokenPositionExtractor
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.graph_builders.utils import (
    create_log_span_for_extractor,
)
from exasol.ai.text.impl.stage_graph.query_handlers.token_position_sql_stage import (
    TokenPositionSQLStage,
)
from exasol.ai.text.impl.utils.stage_graph_utils import get_output_table


class TokenPositionGraphBuilder:
    """
    Stage graph builder for the TokenPosition Extractor.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:
        if not isinstance(extractor, TokenPositionExtractor):
            raise ValueError(
                "Wrong type of extractor for the Token Position Graph Builder. "
                f"Expected {TokenPositionExtractor.__name__}. "
                f"Got {type(extractor).__name__}."
            )

        # This is subject to change, once the output table strategy is decided.
        output_view = get_output_table(extractor, context)
        output_name = extractor.name
        log_span = create_log_span_for_extractor(extractor)
        stage = TokenPositionSQLStage(
            output_view=output_view, output_name=output_name, log_span=log_span
        )
        return SQLStageGraph(stage, stage, set())
