from textwrap import dedent

from exasol.analytics.query_handler.query.interface import Query
from exasol.analytics.query_handler.query.select import (
    LogSpan,
    ModifyQuery,
    SelectQuery,
)
from exasol.analytics.schema import (
    DbObjectType,
    DbOperationType,
    TableLikeName,
)

from exasol.ai.text.impl.stage_graph.query_handlers.source_table.metadata_table import (
    MetadataTable,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.utils import (
    get_map_table,
    render_columns,
    render_create_columns,
)


class MappingTable:
    """
    Class that keeps all information regarding the mapping table that links the source table and the Documents table and
    This class can be used to generate all corresponding queries to this table, such as creation, update and views.
    """

    def __init__(
        self,
        metadata_table: MetadataTable,
        document_table_name: TableLikeName,
        doc_id_name: str,
        log_span: LogSpan,
    ):
        self.log_span = log_span
        self.doc_id_name = doc_id_name
        self.document_table_name = document_table_name
        self.metadata_table = metadata_table

    @property
    def name(self) -> TableLikeName:
        return get_map_table(self.document_table_name, self.metadata_table.table)

    def get_create_mapping_query(self) -> Query:
        """
        Query creating a mapping table for the provided source table,
        if one doesn't exist.
        """
        query = (
            f"CREATE TABLE IF NOT EXISTS {self.name.fully_qualified}"
            f"({render_create_columns(self.metadata_table.map_columns)}, "
            f'DISTRIBUTE BY "{self.metadata_table.map_doc_id_column.name}")'
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.name,
            db_operation_type=DbOperationType.CREATE_IF_NOT_EXISTS,
            audit=True,
            parent_log_span=self.log_span,
        )

    def get_mapping_update_query(self) -> Query:
        """
        Query updating the mapping table with new data found in the provided source
        table.
        During the execution of this query new document IDs are created.
        """
        where_clause = " AND ".join(
            f'M."{key.name}" IS NULL' for key in self.metadata_table.keys
        )
        # The list of the text columns names, to populate the "Source" column with.
        text_list = ", ".join(f"('{col.name}')" for col in self.metadata_table.columns)

        query = dedent(f"""
            INSERT INTO {self.name.fully_qualified}({render_columns(self.metadata_table.map_columns)})
            SELECT {render_columns(self.metadata_table.keys, 'S')}, L.LAST_ID + ROW_NUMBER() OVER (), T.SRC
            FROM {self.metadata_table.table.fully_qualified} S
            LEFT OUTER JOIN {self.name.fully_qualified} M
            ON {self.metadata_table.map_join_on_clause}
            CROSS JOIN (SELECT * FROM (VALUES {text_list})
            AS TEXTS(SRC)) T
            CROSS JOIN (SELECT COALESCE(MAX("{self.doc_id_name}"), 0) AS LAST_ID
            FROM {self.document_table_name.fully_qualified}) L
            WHERE {where_clause}
        """)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.name,
            db_operation_type=DbOperationType.INSERT,
            audit=True,
            parent_log_span=self.log_span,
        )

    def get_mapping_clean_query(self) -> Query:
        """
        Query deleting rows in the mapping tables that do not have correspondent rows
        in the Documents table. This can happen if the text in the source table is NULL.
        It is important to get rid of such rows, otherwise the next update of a mapping
        table can generate a document ID that has already been used, not made it to the
        Document table but is still present in one of the mapping tables.
        """
        query = dedent(f"""
            DELETE FROM {self.name.fully_qualified} 
            WHERE "{self.metadata_table.map_doc_id_column.name}" NOT IN (
                SELECT "{self.doc_id_name}" FROM {self.document_table_name.fully_qualified}
            )
        """)
        # This query would be auditable if there was a DbOperation defined for DELETE.
        return SelectQuery(query)
