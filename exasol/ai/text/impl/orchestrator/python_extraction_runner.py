from typing import Any

from exasol.analytics.audit.audit_query_handler import AuditQueryHandler
from exasol.analytics.audit.columns import BaseAuditColumns
from exasol.analytics.query_handler.context.top_level_query_handler_context import (
    ScopeQueryHandlerContext,
    TopLevelQueryHandlerContext,
)
from exasol.analytics.query_handler.python_query_handler_runner import (
    PythonQueryHandlerRunner,
)
from exasol.analytics.schema import varchar_column
from exasol.analytics.sql_executor.interface import SQLExecutor

from exasol.ai.text.extraction import AbstractExtraction
from exasol.ai.text.impl.orchestrator.noop_pathlike import NoOpPathLike
from exasol.ai.text.impl.orchestrator.orchestrator_query_handler import (
    OrchestratorParameter,
    OrchestratorQueryHandler,
)


def temporary_db_object_name_prefix(sql_executor: SQLExecutor):
    result = sql_executor.execute("SELECT CURRENT_SESSION").fetchall()
    session_id = result[0][0]
    return f"tmp_{session_id}"


def query_handler_context(
    temp_schema: str,
    temp_prefix: str,
) -> TopLevelQueryHandlerContext:
    """
    temp_schema: database schema for temporary objects
    """

    def raise_not_implemented_error(arg: str):
        raise NotImplementedError()

    return TopLevelQueryHandlerContext(
        temporary_bucketfs_location=NoOpPathLike.default_instance(),
        temporary_schema_name=temp_schema,
        temporary_db_object_name_prefix=temp_prefix,
        connection_lookup=raise_not_implemented_error,
    )


class PythonExtractionRunner:
    """
    Implementation of :class:`ExtractionRunner` running the specified
    extraction in the local python interpreter without using a UDF.
    """

    def run(
        self,
        extraction: AbstractExtraction,
        sql_executor: SQLExecutor,
        temporary_db_object_schema: str,
        language_alias: str,
    ) -> None:
        def audit_schema(parameter: OrchestratorParameter) -> str:
            return extraction.output.db_schema

        def audit_table_prefix(parameter: OrchestratorParameter) -> str:
            p = extraction.output.table_prefix
            return f"{p}_TXAIE" if p else "TXAIE"

        def factory(
            parameter: OrchestratorParameter,
            context: ScopeQueryHandlerContext,
        ) -> AuditQueryHandler[OrchestratorParameter, Any]:
            BaseAuditColumns.ERROR_MESSAGE = varchar_column(
                "ERROR_MESSAGE", size=2000000
            )
            BaseAuditColumns.all[-1] = BaseAuditColumns.ERROR_MESSAGE
            return AuditQueryHandler(
                parameter=parameter,
                context=context,
                query_handler_factory=OrchestratorQueryHandler,
                schema_getter=audit_schema,
                table_name_prefix_getter=audit_table_prefix,
                additional_columns=[],
            )

        temp_prefix = temporary_db_object_name_prefix(sql_executor)
        context = query_handler_context(temporary_db_object_schema, temp_prefix)
        parameter = OrchestratorParameter(
            extraction=extraction,
            language_alias=language_alias,
        )
        PythonQueryHandlerRunner(
            sql_executor=sql_executor,
            top_level_query_handler_context=context,
            parameter=parameter,
            query_handler_factory=factory,
        ).run()
