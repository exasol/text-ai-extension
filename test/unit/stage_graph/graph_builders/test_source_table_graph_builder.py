from typing import cast

from exasol.ai.text.extractors import SourceTableExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import SourceTableGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.source_table_sql_stage import (
    SourceTableSQLStage,
)
from test.utils.helpers import source_tables


def test_source_table_graph_builder(extractor_context):
    sources = source_tables()
    extractor = SourceTableExtractor(sources=sources)
    builder = SourceTableGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, SourceTableSQLStage)
    stage = cast(SourceTableSQLStage, graph.start_node)
    assert stage.sources == sources
    assert graph.end_node is stage
