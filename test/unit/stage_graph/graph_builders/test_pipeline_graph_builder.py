from unittest.mock import (
    call,
    create_autospec,
)

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors import PipelineExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import (
    GraphBuilder,
    PipelineGraphBuilder,
)
from exasol.ai.text.impl.utils.stage_graph_utils import Edge
from test.unit.sql_stage_mock import sql_stage_graph
from test.utils.helpers import named_entity_extractor


def test_non_empty_pipeline(extractor_context):
    # define pipeline with two steps s1 and s2:
    s1 = named_entity_extractor()
    s2 = named_entity_extractor()
    pipeline = PipelineExtractor(steps=[s1, s2])

    # mock downstream_builder simulating to generate
    # SQLStageGraphs g1 and g2 for the two pipeline steps
    downstream_builder = create_autospec(GraphBuilder)
    g1 = sql_stage_graph(1, 2)
    g2 = sql_stage_graph(3, 4)
    downstream_builder.build.side_effect = [g1, g2]

    pl_builder = PipelineGraphBuilder()
    actual = pl_builder.build(pipeline, extractor_context, downstream_builder)

    def expected_edges() -> set[Edge]:
        interlinking = {(g1.end_node, g2.start_node)}
        return g1.edges() | g2.edges() | interlinking

    expected = SQLStageGraph(g1.start_node, g2.end_node, expected_edges())

    # Assert downstream_builder has been called for each step of the pipeline
    # and actual generated SQLStageGraph is as expected.
    assert (
        downstream_builder.build.call_args_list
        == [
            call(s1, extractor_context, downstream_builder),
            call(s2, extractor_context, downstream_builder),
        ]
        and expected == actual
    )
