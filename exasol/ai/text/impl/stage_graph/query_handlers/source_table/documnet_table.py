from textwrap import dedent

from exasol.analytics.query_handler.query.interface import Query
from exasol.analytics.query_handler.query.select import (
    LogSpan,
    ModifyQuery,
)
from exasol.analytics.schema import (
    DbObjectType,
    DbOperationType,
    TableLikeName,
)

from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.mapping_table import (
    MappingTable,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.metadata_table import (
    SOURCE_COLUMN,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.utils import (
    get_source_view,
    render_columns,
    render_create_columns,
)
from exasol.ai.text.impl.utils.pandas_utils import ensure_unique_name


class DocumentTable:
    """
    Class that keeps all information regarding the Documents table and can be used to generate all
    corresponding queries to this table, such as creation, update and views.
    """

    def __init__(self, document_table_name: TableLikeName, log_span: LogSpan):
        self.log_span = log_span
        self.name = document_table_name
        self.text_inputs = TextInput()

    @property
    def doc_id_name(self) -> str:
        """
        The name of the document id column in the document table.
        """
        return self.text_inputs.text_span.doc_id.name

    @property
    def document_columns(self) -> list[ColumnSpec]:
        """
        The document table columns.
        """
        return self.text_inputs.order

    @property
    def create_document_query(self) -> Query:
        """
        Query creating the output table if it doesn't exist.
        """
        query = (
            f"CREATE TABLE IF NOT EXISTS {self.name.fully_qualified}"
            f"({render_create_columns(self.document_columns)}, "
            f'DISTRIBUTE BY "{self.doc_id_name}")'
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.name,
            db_operation_type=DbOperationType.CREATE_IF_NOT_EXISTS,
            audit=True,
            parent_log_span=self.log_span,
        )

    def get_document_update_query(self, mapping_table: MappingTable) -> Query:
        """
        Query updating the document table with new data found in the provided source
        table.
        Important: the correspondent mapping update query should be executed first.
        """
        metadata_table = mapping_table.metadata_table
        map_update_name = ensure_unique_name(
            "MAP_UPDATE", [metadata_table.table.name, self.name.name]
        )

        sub_queries = "\nUNION ALL\n".join(dedent(f"""
                SELECT M."{metadata_table.map_doc_id_column.name}", 0, LENGTH(S."{col.name}"), S."{col.name}"
                FROM {metadata_table.table.fully_qualified} S
                JOIN {map_update_name} M ON {metadata_table.map_join_on_clause}
                WHERE M."{metadata_table.map_source_column.name}" = '{col.name}'
                AND S."{col.name}" IS NOT NULL
            """) for col in metadata_table.columns)

        query = dedent(f"""
            INSERT INTO {self.name.fully_qualified}({render_columns(self.document_columns)})
            WITH {map_update_name} AS (
                SELECT M.*
                FROM {mapping_table.name.fully_qualified} M
                LEFT OUTER JOIN {self.name.fully_qualified} D
                ON M."{metadata_table.map_doc_id_column.name}" = D."{self.doc_id_name}"
                WHERE D."{self.doc_id_name}" IS NULL
            )
            {sub_queries}
        """)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.name,
            db_operation_type=DbOperationType.INSERT,
            audit=True,
            parent_log_span=self.log_span,
        )

    def get_source_view_query(self, mapping_table: MappingTable) -> Query:
        """
        Query creating a view that links the document table with source table. This
        allows linking the output of extractors to the source tables via the text span.
        """
        metadata_table = mapping_table.metadata_table
        src_view = get_source_view(self.name, metadata_table.table)
        text_span_columns = ", ".join(
            f"D.{name}" for name in self.text_inputs.text_span.quoted_column_names
        )
        query = dedent(f"""
            CREATE OR REPLACE VIEW {src_view.fully_qualified} AS
            SELECT S.*, {text_span_columns}, M."{SOURCE_COLUMN}"
            FROM {metadata_table.table.fully_qualified} S
            JOIN {mapping_table.name.fully_qualified} M ON {metadata_table.map_join_on_clause}
            JOIN {self.name.fully_qualified} D
            ON M."{metadata_table.map_doc_id_column.name}" = D."{self.doc_id_name}"
        """)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.VIEW,
            db_object_name=src_view,
            db_operation_type=DbOperationType.CREATE_OR_REPLACE,
            audit=True,
            parent_log_span=self.log_span,
        )
