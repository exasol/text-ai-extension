from exasol.ai.text.extractors import CoOccurrenceExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import CoOccurrenceGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.co_occurrence_sql_stage import (
    CoOccurrenceSQLStage,
)


def test_filter_graph_builder(extractor_context):
    extractor = CoOccurrenceExtractor()
    builder = CoOccurrenceGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, CoOccurrenceSQLStage)
    # Validation of the output_view and the output_name will be added
    # once the output table strategy is decided.
