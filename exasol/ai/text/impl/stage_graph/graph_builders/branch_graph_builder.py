from typing import cast

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import (
    BranchExtractor,
    Extractor,
)
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.identity_query_handler import (
    IdentitySqlStage,
)
from exasol.ai.text.impl.stage_graph.query_handlers.list_collector_query_handler import (
    ListCollectorSqlStage,
)
from exasol.ai.text.impl.utils.stage_graph_utils import (
    all_edges,
)


class BranchGraphBuilder:
    """
    Builds an execution graph for a list of extractors branching from a
    common input, see protocol :class:`GraphBuilder` for more details.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:

        assert isinstance(extractor, BranchExtractor)
        branch = cast(BranchExtractor, extractor)

        # BranchExtractor.paths is constrained by pydantic to min_length 1.
        # So branch cannot be empty.
        if downstream_builder is None:
            raise ValueError(
                "A valid downstream graph builder must be provided "
                f"in a call to {self.__class__.__name__}.build()."
            )
        subgraphs = [
            downstream_builder.build(cast(Extractor, path), context, downstream_builder)
            for path in branch.paths
        ]

        start_node = IdentitySqlStage()
        end_node = ListCollectorSqlStage()

        # link overall start_node to each branch's start_node
        inbound = {(start_node, g.start_node) for g in subgraphs}
        # link each branch's end_node to the overall end_node
        outbound = {(g.end_node, end_node) for g in subgraphs}

        edges = all_edges(subgraphs) | inbound | outbound
        return SQLStageGraph(start_node, end_node, edges)
