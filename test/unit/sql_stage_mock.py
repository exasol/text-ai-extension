from dataclasses import dataclass
from unittest.mock import create_autospec

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)


@dataclass(eq=True, frozen=True)
class SqlStageMock(SQLStage):
    """
    Simulate an SQLStage which is identified by an int ID.  The stage only
    creates a mock of an SQLStageQueryHandler.
    """

    id: int

    def __repr__(self) -> str:
        return str(self.id)

    def create_query_handler(
        self,
        input: SQLStageQueryHandlerInput,
        context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return create_autospec(SQLStageQueryHandler)


def sql_stage_graph(start_id: int, end_id: int) -> SQLStageGraph:
    """
    Create an SQLStageGraph with mocked start and end nodes, each
    identified by the specified id.
    """
    start_node: SQLStage = SqlStageMock(start_id)
    end_node: SQLStage = SqlStageMock(end_id)
    edges = {(start_node, end_node)}
    return SQLStageGraph(start_node, end_node, edges)
