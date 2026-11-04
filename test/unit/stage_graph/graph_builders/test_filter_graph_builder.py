from typing import cast

from exasol.ai.text.extractors import FilterExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import FilterGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.filter_query_handler import (
    FilterSqlStage,
)


def test_filter_graph_builder(extractor_context):
    allowed_names = {"alpha", "beta", "gamma"}
    extractor = FilterExtractor(allowed_names=allowed_names)
    builder = FilterGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, FilterSqlStage)
    stage = cast(FilterSqlStage, graph.start_node)
    assert stage.allowed_names == allowed_names
