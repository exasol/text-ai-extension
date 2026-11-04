from exasol.ai.text.extractors import TokenPositionExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import TokenPositionGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.token_position_sql_stage import (
    TokenPositionSQLStage,
)


def test_filter_graph_builder(extractor_context):
    extractor = TokenPositionExtractor()
    builder = TokenPositionGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, TokenPositionSQLStage)
    # Validation of the output_view and the output_name will be added
    # once the output table strategy is decided.
