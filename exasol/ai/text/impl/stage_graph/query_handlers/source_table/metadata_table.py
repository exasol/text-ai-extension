from dataclasses import dataclass

from exasol.analytics.schema import TableLikeName

from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.utils.pandas_utils import ensure_unique_name
from exasol.ai.text.impl.utils.sql_utils import MAX_NAME_LENGTH


@dataclass
class MetadataTable:
    """
    This is a counterpart of the `TableSource` class defined in the
    source_table_extractor. It contains the collected metadata for a source table.
    While the extractor's `SchemaSource`/`TableSource` classes can be considered a
    request for the metadata, this class holds the result of that request.
    """

    table: TableLikeName
    columns: list[ColumnSpec]
    keys: list[ColumnSpec]

    @property
    def key_names(self) -> list[str]:
        return [key.name for key in self.keys]

    @property
    def map_doc_id_column(self) -> ColumnSpec:
        """
        Document ID column of the mapping table.
        """
        return ColumnSpec(
            name=ensure_unique_name(DOC_ID_COLUMN, self.key_names), sql_type="INTEGER"
        )

    @property
    def map_source_column(self) -> ColumnSpec:
        """
        Source text column name column of the mapping table.
        """
        return ColumnSpec(
            name=ensure_unique_name(SOURCE_COLUMN, self.key_names),
            sql_type=f"VARCHAR({MAX_NAME_LENGTH})",
        )

    @property
    def map_columns(self) -> list[ColumnSpec]:
        """
        List of columns of the mapping table.
        """
        return self.keys + [self.map_doc_id_column, self.map_source_column]

    @property
    def map_join_on_clause(self) -> str:
        """
        The ON condition in the JOIN clause, linking the source and the mapping tables.
        Note, that the aliases for the source and mapping tables are assumed to be
        S and M respectively.
        """
        return " AND ".join(f'S."{key.name}"=M."{key.name}"' for key in self.keys)


DOC_ID_COLUMN = "TEXT_DOC_ID"
SOURCE_COLUMN = "SOURCE"
