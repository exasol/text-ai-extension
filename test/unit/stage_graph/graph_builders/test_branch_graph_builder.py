from unittest.mock import (
    call,
    create_autospec,
)

from exasol.ai.text.extractors import BranchExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import (
    BranchGraphBuilder,
    GraphBuilder,
)
from exasol.ai.text.impl.stage_graph.query_handlers.identity_query_handler import (
    IdentitySqlStage,
)
from exasol.ai.text.impl.stage_graph.query_handlers.list_collector_query_handler import (
    ListCollectorSqlStage,
)
from exasol.ai.text.impl.utils.stage_graph_utils import Edge
from test.unit.sql_stage_mock import sql_stage_graph
from test.utils.helpers import named_entity_extractor


def test_non_empty_branch(extractor_context):
    # define branch with two paths p1 and p2:
    p1 = named_entity_extractor()
    p2 = named_entity_extractor()
    branch = BranchExtractor(paths=[p1, p2])

    # mock downstream_builder simulating to generate
    # SQLStageGraphs g1 and g2 for the two branch paths
    downstream_builder = create_autospec(GraphBuilder)
    g1 = sql_stage_graph(1, 2)
    g2 = sql_stage_graph(3, 4)
    downstream_builder.build.side_effect = [g1, g2]

    actual = BranchGraphBuilder().build(branch, extractor_context, downstream_builder)

    def expected_edges() -> set[Edge]:
        inbound = {(actual.start_node, g.start_node) for g in (g1, g2)}
        outbound = {(g.end_node, actual.end_node) for g in (g1, g2)}
        return g1.edges() | g2.edges() | inbound | outbound

    # Assert downstream_builder has been called for each path of the branch
    # and actual generated SQLStageGraph is as expected.
    assert isinstance(actual.start_node, IdentitySqlStage)
    assert isinstance(actual.end_node, ListCollectorSqlStage)
    assert (
        downstream_builder.build.call_args_list
        == [
            call(p1, extractor_context, downstream_builder),
            call(p2, extractor_context, downstream_builder),
        ]
        and expected_edges() == actual.edges()
    )
