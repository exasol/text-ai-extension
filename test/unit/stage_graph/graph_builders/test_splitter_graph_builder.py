from typing import cast

from exasol.ai.text.extractors import SplitterExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import SplitterGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.splitter_query_handler import (
    SplitterSqlStage,
)


def test_splitter_graph_builder(extractor_context):
    new_names = {"alpha", "beta", "gamma"}
    extractor = SplitterExtractor(new_names=new_names)
    builder = SplitterGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, SplitterSqlStage)
    stage = cast(SplitterSqlStage, graph.start_node)
    assert stage.new_names == new_names
