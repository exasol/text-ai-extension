from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.schema import TableLikeName

from exasol.ai.text.impl.stage_graph.query_handlers.token_position_query_handler import (
    TokenPositionQueryHandler,
)


class TokenPositionSQLStage(SQLStage):
    """
    The factory of the token position query handler.
    For the description of parameters see the TokenPositionQueryHandler docstring.
    """

    def __init__(
        self,
        output_view: TableLikeName,
        log_span: LogSpan,
        output_name: str | None = None,
    ) -> None:
        self.log_span = log_span
        self.output_view = output_view
        self.output_name = output_name

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return TokenPositionQueryHandler(
            stage_input=stage_input,
            query_handler_context=query_handler_context,
            output_view=self.output_view,
            output_name=self.output_name,
            log_span=self.log_span,
        )
