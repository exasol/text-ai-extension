from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.result import Finish

from exasol.ai.text.impl.stage_graph.query_handlers.common import (
    ContinueOrFinish,
    IllegalMethodCallError,
)


class IdentityQueryHandler(SQLStageQueryHandler):
    """
    Query handler for :class:``IdentitySqlStage`` expecting exactly one
    input.

    For a usecase please see :class:`BranchGraphBuilder`.
    """

    def __init__(
        self, parameter: SQLStageQueryHandlerInput, context: ScopeQueryHandlerContext
    ):
        super().__init__(parameter, context)
        self._parameter = parameter
        assert len(self._parameter.sql_stage_inputs) == 1

    def start(self) -> ContinueOrFinish:
        return Finish(self._parameter.sql_stage_inputs[0])

    def handle_query_result(self, query_result: QueryResult) -> ContinueOrFinish:
        raise IllegalMethodCallError(
            "Method IdentityQueryHandler.handle_query_result()"
            " is not intended to be called."
        )


class IdentitySqlStage(SQLStage):
    """
    An IdentitySqlStage serves as start node for an SQLStageGraph
    representing a branch structure. It creates a query handler that providing
    its input as output without any modification.

    For a usecase please see :class:`BranchGraphBuilder`.
    """

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        qh_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return IdentityQueryHandler(stage_input, qh_context)
