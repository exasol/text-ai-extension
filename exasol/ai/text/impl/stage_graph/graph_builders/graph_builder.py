from typing import (
    Optional,
    Protocol,
)

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)


class GraphBuilder(Protocol):
    """
    An interface for a class that builds an SQLStageGraph based on the provided
    configuration and context.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: Optional["GraphBuilder"] = None,
    ) -> SQLStageGraph:
        """
        Builds the execution graph for a given extractor.

        :param extractor:
            Definition of the extractor and its configuration.
        :param context:
            The context contains "static" parameters provided at the graph design time.
            An example of such a parameter is the database schema.
        :param downstream_builder:
            The extractor may contain nested extractors. An example of such an extractor
            is the Pipeline, which is a sequence of extractors. This parameter is the
            graph builder for the nested extractors. The interface might seem limiting
            as there is only one downstream graph builder. The DispatchingGraphBuilder
            at least partially solves this limitation.
        """
