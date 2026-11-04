from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder


class NoGraphBuilderRegisteredError(Exception):
    """
    Raised when trying to build a graph for an Exporter without registering a
    corresponding GraphBuilder.
    """


class DispatchingGraphBuilder:
    """
    Builds the execution graph by dispatching the call to one of the registered
    graph builders. The class keeps a map of the graph builders indexed by the extractor
    types.

    The `downstream_builder` is forwarded to the graph builder selected to handle
    the call. A common pattern for processing a composite extractor is to call an
    instance of the DispatchingGraphBuilder providing the root extractor and the same
    or a different instance of the DispatchingGraphBuilder as the `downstream_builder`.
    For this to work, all the required graph builders must be registered in the
    dispatching graph builder.

    NoGraphBuilderRegisteredError will be raised if no graph builder is found for the
    provided extractor.
    """

    def __init__(self):
        self._builders: dict[type[Extractor], GraphBuilder] = {}

    def register(self, key: type[Extractor], graph_builder: GraphBuilder):
        self._builders[key] = graph_builder

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:
        extractor_type = type(extractor)
        builder = self._builders.get(extractor_type)
        if builder is None:
            raise NoGraphBuilderRegisteredError(f"Extractor of type {extractor_type}")
        return builder.build(extractor, context, downstream_builder)
