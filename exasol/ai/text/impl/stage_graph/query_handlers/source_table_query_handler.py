"""
The Query Handler for the Source Table Extractor. For more information please refer to
the docstring of the SourceTableQueryHandler class below.
"""

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.interface import Query
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query.select import (
    AuditQuery,
    LogSpan,
    SelectQuery,
    SelectQueryWithColumnDefinition,
)
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)
from exasol.analytics.schema import (
    Column,
    TableLikeName,
)

from exasol.ai.text.extractors.source_table_extractor import SchemaSource
from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.stage_graph.input_output.source_table_output import (
    SourceTableOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.documnet_table import (
    DocumentTable,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.mapping_table import (
    MappingTable,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.utils import (
    IS_KEY_COLUMN,
    METADATA_CAST_SIZE,
    get_source_tables,
    select_metadata_query,
    validate_source_tables,
)
from exasol.ai.text.impl.utils.sql_utils import (
    boolean_column_spec,
    varchar_column_spec,
)


class SourceTableQueryHandler(SQLStageQueryHandler):
    """
    Implements a node-level AAF Query Handler.

    Performs copying text from multiple sources into a single output table hereafter
    called Document table. The sources can be scattered across different tables located
    in different schemas. Each source table can have one or more columns where text
    should be taken from. Apart from the text columns, a source table should define one
    or more key columns that can be used to identify the rows. Alternatively, the
    primary keys can be used for that.

    Updating the Document table is a continuous process. The node is expected to run
    periodically. At each run the new data found in the source tables will be copied to
    the Document table.

    Apart from the Document table, the node maintains a list of Mapping tables. There is
    one mapping table for every source table. A mapping table provides a link between
    the Document table and one of the source tables, as illustrated below.
    |------- -------------------|      |----------------------|        |----------------|
    |       Source Table        | keys |    Mapping Table     | doc_id | Document Table |
    |(key1, key2, text1, text2) |------| (key1, key2, doc_id) |--------| (doc_id, text) |
    |---------------------------|      |----------------------|        |----------------|
    (The diagram doesn't show all columns).
    The mapping tables cannot be merged in one because the source tables have different
    sets of keys. These tables serve a dual purpose. Primarily, they are used to track
    the rows that have already been copied, so only the new ones are selected. Also,
    they can be used to link the Document with the sources to trace back the origins of
    the texts.

    Document table columns:
    text_doc_id:
        The text ID (PRIMARY KEY).
    text_char_begin:
        Starting index of the text span (PRIMARY KEY). For the texts extracted from
        Exasol database, this should be zero.
    text_char_end:
        Ending index of the text span (PRIMARY KEY). For the texts extracted from
        Exasol database, this should be set to the length of the text.
    text:
        The extracted text.
    The exact names are constructed by the `utils.sql_utils.get_span_columns` function
    and the value of the `TEXT_COLUMN` constant.

    A mapping table columns:
    keys:
        These are the same keys defined in the correspondent source table.
    text_doc_id:
        The text ID, as in the Document table. The exact name is defined by the
        `DOC_ID_COLUMN` constant.
    source:
        The name of the column in the source table the text is extracted from. The
        exact name is defined by the `SOURCE_COLUMN` constant.

    The node works in two steps.
    1. Obtain the source table names and the names and types of their text and key
       columns (all names can be defined as a pattern). This is achieved by querying
       the metadata tables of the DB - the SYS.EXA_ALL_COLUMNS and
       SYS.EXA_ALL_CONSTRAINT_COLUMNS.
    2. Extract the texts from the sources. This involves several queries:
       2.1. Ensure the Document table exists.
       2.2. Ensure the Mapping tables exist.
       2.3. Update the mapping tables with the new rows found in the source tables,
            allocating IDs for the new texts.
       2.4. Update the Document table with the new texts.
       2.5. Commit the changes.

    Parameters:
        sources:
            The text sources defined by the extractor.
        document_table_name:
            The name of the Document table.
        parameter:
            The node input data. This query handler doesn't use it.
        query_handler_context:
            The context object.
    """

    def __init__(
        self,
        sources: list[SchemaSource],
        document_table_name: TableLikeName,
        parameter: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
        log_span: LogSpan,
    ) -> None:
        super().__init__(parameter, query_handler_context)
        self.log_span = log_span
        self.sources = sources
        self.document_table = DocumentTable(
            document_table_name=document_table_name, log_span=self.log_span
        )
        self.parameter = parameter
        # Source/Mapping tables received from the last metadata query.
        self.latest_mapping: dict[TableLikeName, TableLikeName] = {}

    @property
    def select_metadata_output(self) -> list[Column]:
        """
        List of AAF Columns defining the result of the metadata query.
        """
        # The characterSet should be provided as a parameter to the `varchar_column_spec`
        # See https://github.com/exasol/text-ai-extension/issues/258
        return [
            boolean_column_spec(IS_KEY_COLUMN).to_column(),
            varchar_column_spec("COLUMN_SCHEMA", METADATA_CAST_SIZE).to_column(
                characterSet="UTF8"
            ),
            varchar_column_spec("COLUMN_TABLE", METADATA_CAST_SIZE).to_column(
                characterSet="UTF8"
            ),
            varchar_column_spec("COLUMN_NAME", METADATA_CAST_SIZE).to_column(
                characterSet="UTF8"
            ),
            varchar_column_spec("COLUMN_TYPE", METADATA_CAST_SIZE).to_column(
                characterSet="UTF8"
            ),
        ]

    @property
    def commit_output_query(self) -> Query:
        """
        Query committing the data.
        """
        return SelectQuery("COMMIT")

    @property
    def start_audit_query(self) -> Query:
        return AuditQuery(audit_fields={"EVENT_NAME": "Start"}, log_span=self.log_span)

    @property
    def end_audit_query(self) -> Query:
        return AuditQuery(audit_fields={"EVENT_NAME": "End"}, log_span=self.log_span)

    @property
    def final_output(self) -> SourceTableOutput:
        return SourceTableOutput(
            datasource=self.document_table.name,
            columns=self.document_table.text_inputs,
            source_mapping=self.latest_mapping,
        )

    def start(self) -> Continue | Finish[SQLStageInputOutput]:
        # Request the metadata first
        return Continue(
            query_list=[self.start_audit_query],
            input_query=SelectQueryWithColumnDefinition(
                query_string=self.select_metadata_query,
                output_columns=self.select_metadata_output,
            ),
        )

    @property
    def select_metadata_query(self):
        return select_metadata_query(self.sources)

    def handle_query_result(
        self, query_result: QueryResult
    ) -> Continue | Finish[SQLStageInputOutput]:
        if len(query_result.columns()) == 1:
            # We are done
            return Finish(result=self.final_output)

        # This must be the result of the metadata query
        source_tables = get_source_tables(query_result)
        validate_source_tables(source_tables)

        query_list = [self.document_table.create_document_query]
        self.latest_mapping.clear()
        for tbl in source_tables:
            mapping_table = self.get_mapping_table(tbl)
            query_list.append(mapping_table.get_create_mapping_query())
            query_list.append(mapping_table.get_mapping_update_query())
            query_list.append(
                self.document_table.get_document_update_query(mapping_table)
            )
            query_list.append(mapping_table.get_mapping_clean_query())
            query_list.append(self.document_table.get_source_view_query(mapping_table))
            self.latest_mapping[tbl.table] = mapping_table.name
        query_list.append(self.end_audit_query)
        query_list.append(self.commit_output_query)

        output_column = ColumnSpec(name="ONE", sql_type="DECIMAL(1,0)").to_column()
        return Continue(
            query_list=query_list,
            input_query=SelectQueryWithColumnDefinition(
                query_string=(
                    "SELECT CAST(1 AS DECIMAL(1,0)) "
                    f"AS {output_column.name.quoted_name}"
                ),
                output_columns=[output_column],
            ),
        )

    def get_mapping_table(self, tbl):
        return MappingTable(
            tbl,
            self.document_table.name,
            self.document_table.doc_id_name,
            log_span=self.log_span,
        )
