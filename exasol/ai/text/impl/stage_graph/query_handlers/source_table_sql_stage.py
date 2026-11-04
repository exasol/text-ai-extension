from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.schema import TableLikeName

from exasol.ai.text.extractors.source_table_extractor import SchemaSource
from exasol.ai.text.impl.stage_graph.query_handlers.source_table_query_handler import (
    SourceTableQueryHandler,
)


class SourceTableSQLStage(SQLStage):
    """
    The factory of the source table query handler.
    """

    def __init__(
        self,
        sources: list[SchemaSource],
        document_table: TableLikeName,
        log_span: LogSpan,
    ) -> None:
        self.log_span = log_span
        self.sources = sources
        self.document_table = document_table

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        """
        Create an instance of the query handler.
        For information on the required input see the docstring of the
        SourceTableQueryHandler class.
        """
        return SourceTableQueryHandler(
            sources=self.sources,
            document_table_name=self.document_table,
            parameter=stage_input,
            query_handler_context=query_handler_context,
            log_span=self.log_span,
        )
