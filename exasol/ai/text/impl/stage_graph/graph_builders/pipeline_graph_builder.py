from typing import cast

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import (
    Extractor,
    PipelineExtractor,
)
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.utils.stage_graph_utils import (
    all_edges,
    linear_interlinks,
)


class PipelineGraphBuilder:
    """
    Builds an execution graph for a sequence of extractors.

    The provided `downstream_builder` will be used to build a graph for each of the
    extractors in the sequence. Unless all extractors in the sequence are of the
    same type, the `downstream_builder` should be a `DispatchingGraphBuilder`. All the
    extractors in the sequence should have their graph builders registered in the
    dispatching graph builder. If any of the extractors are composite extractors
    themselves the graph builders for the nested extractors should also be registered
    in the dispatching graph builder.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:

        if not isinstance(extractor, PipelineExtractor):
            raise RuntimeError(
                f"{PipelineGraphBuilder.__name__} expects an instance "
                f"of {PipelineExtractor.__name__} in its build method. "
                f"Instead {extractor.__class__.__name__} was provided."
            )
        pipeline = cast(PipelineExtractor, extractor)

        # Pipeline.steps is constrained by pydantic to min_length 1.
        # So pipeline cannot be empty.
        if downstream_builder is None:
            raise ValueError(
                "A valid downstream graph builder must be provided "
                f"in a call to {self.__class__.__name__}.build()."
            )
        subgraphs = [
            downstream_builder.build(cast(Extractor, step), context, downstream_builder)
            for step in pipeline.steps
        ]
        start_node = subgraphs[0].start_node
        end_node = subgraphs[-1].end_node
        edges = all_edges(subgraphs) | linear_interlinks(subgraphs)
        return SQLStageGraph(start_node, end_node, edges)
