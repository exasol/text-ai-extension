from collections import defaultdict
from dataclasses import dataclass
from textwrap import dedent

from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.schema import (
    SchemaName,
    TableLikeName,
    TableLikeNameImpl,
)

from exasol.ai.text.extractors import (
    NameSelector,
    SchemaSource,
    SelectorPredicate,
)
from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.metadata_table import (
    MetadataTable,
)
from exasol.ai.text.impl.utils.sql_utils import create_db_object_name


def get_source_tables(query_result: QueryResult) -> list[MetadataTable]:
    """
    Reads the result of the metadata query and converts it to a list of MetaTable
    objects.
    """

    # Collect the lists of text columns and key columns for each source table
    columns_map = defaultdict(list)
    keys_map = defaultdict(list)
    while True:
        tbl = TableLikeNameImpl(
            query_result.COLUMN_TABLE, SchemaName(query_result.COLUMN_SCHEMA)
        )
        name_and_type = ColumnSpec(
            name=query_result.COLUMN_NAME, sql_type=query_result.COLUMN_TYPE
        )
        if query_result[IS_KEY_COLUMN]:
            keys_map[tbl].append(name_and_type)
        else:
            columns_map[tbl].append(name_and_type)
        if not query_result.next():
            break

    # Convert the results to the list of SourceTable objects
    return [
        MetadataTable(tbl, columns, keys_map[tbl])
        for tbl, columns in columns_map.items()
    ]


def validate_source_tables(tables: list[MetadataTable]) -> None:
    """
    Validates the provided list of source tables. Checks that:
    1. The keys are defined for every table.
    2. The text columns are of type CHAR or VARCHAR.
    """
    if not all(bool(tbl.keys) for tbl in tables):
        tbl_list = ", ".join(
            tbl.table.fully_qualified for tbl in tables if not tbl.keys
        )
        raise RuntimeError(f"No keys found for the following table(s): {tbl_list}.")

    invalid_columns = [
        col
        for tbl in tables
        for col in tbl.columns
        if not col.sql_type.startswith("CHAR")
        and not col.sql_type.startswith("VARCHAR")
    ]
    if invalid_columns:
        col_list = ", ".join(f"{col.name} {col.sql_type}" for col in invalid_columns)
        raise RuntimeError(
            "The source text columns must be either CHAR or VARCHAR. "
            f"The following column(s) have invalid types: {col_list}."
        )


def get_map_table(
    document_tbl: TableLikeName, source_tbl: TableLikeName, *args
) -> TableLikeName:
    """
    Constructs a mapping table name combining the names of the document table,
    the source table and any additional suffixes provided in the variadic list.
    """
    schema_name = (
        [source_tbl.schema_name.name] if source_tbl.schema_name is not None else []
    )
    return TableLikeNameImpl(
        create_db_object_name(document_tbl.name, *schema_name, source_tbl.name, *args),
        document_tbl.schema_name,
    )


def get_source_view(
    document_tbl: TableLikeName, source_tbl: TableLikeName, *args
) -> TableLikeName:
    return get_map_table(document_tbl, source_tbl, "VIEW")


@dataclass
class DbObj:
    """
    Source selector class. This is an adaptation of the extractor's `SchemaSources`
    and `TableSource` classes made for the convenience of recursive processing.
    """

    obj_name: str
    selector: NameSelector
    children: list["DbObj"] | None = None


def name_condition(obj_name: str, selector: NameSelector) -> str:
    """
    Renders the specified name selector as a condition in an SQL WHERE clause.
    The `obj_name` is the name of a schema, table or column the condition is
    applied for.
    """
    if selector.predicate == SelectorPredicate.EXACT:
        return f"{obj_name}='{selector.pattern}'"
    if selector.predicate == SelectorPredicate.LIKE:
        return f"{obj_name} LIKE '{selector.pattern}'"
    return f"{obj_name} REGEXP_LIKE '{selector.pattern}'"


def recursive_condition(db_obj: DbObj) -> str:
    """
    Renders the specified source selector as a logical expression in the WHERE clause
    of the metadata SELECT statement.
    """
    inner_condition = f" AND ({condition(db_obj.children)})" if db_obj.children else ""
    return name_condition(db_obj.obj_name, db_obj.selector) + inner_condition


def condition(db_obj_list: list[DbObj]) -> str:
    """
    Renders the specified list of source selectors as a logical expression in the WHERE
    clause of the metadata SELECT statement.
    """
    return " OR ".join(recursive_condition(db_obj) for db_obj in db_obj_list)


def get_texts(sources: list[SchemaSource]) -> list[DbObj]:
    """
    An adaptor for the source text columns defined in the extractor.
    """
    return [
        DbObj(
            "COLUMN_SCHEMA",
            src.db_schema,
            [
                DbObj(
                    "COLUMN_TABLE",
                    tbl.table,
                    [DbObj("COLUMN_NAME", col) for col in tbl.columns],
                )
                for tbl in src.tables
            ],
        )
        for src in sources
    ]


def get_keys(sources: list[SchemaSource]) -> list[DbObj]:
    """
    An adaptor for the source tables explicit keys defined in the extractor.
    """
    return [
        DbObj(
            "COLUMN_SCHEMA",
            src.db_schema,
            [
                DbObj(
                    "COLUMN_TABLE",
                    tbl.table,
                    [DbObj("COLUMN_NAME", key) for key in tbl.keys],
                )
                for tbl in src.tables
                if tbl.keys
            ],
        )
        for src in sources
        if any(bool(tbl.keys) for tbl in src.tables)
    ]


def get_pk_tables(sources: list[SchemaSource], alias: str) -> list[DbObj]:
    """
    An adaptor for the source tables defined in the extractor where explicit keys
    are not defined, meaning that the primary keys should be used instead.
    Since the metadata SELECT query joins two tables, the caller must provide an
    alias for the SYS.EXA_ALL_CONSTRAINT_COLUMNS table.
    """
    return [
        DbObj(
            f"{alias}.CONSTRAINT_SCHEMA",
            src.db_schema,
            [
                DbObj(f"{alias}.CONSTRAINT_TABLE", tbl.table)
                for tbl in src.tables
                if not tbl.keys
            ],
        )
        for src in sources
        if not all(bool(tbl.keys) for tbl in src.tables)
    ]


def render_columns(columns: list[ColumnSpec], alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return ", ".join(f'{prefix}"{col.name}"' for col in columns)


def render_create_columns(columns: list[ColumnSpec]) -> str:
    return ", ".join(f'"{col.name}" {col.sql_type}' for col in columns)


IS_KEY_COLUMN = "IS_KEY"
METADATA_CAST_SIZE = 10000
METADATA_CAST_TYPE = f"VARCHAR({METADATA_CAST_SIZE})"


def select_metadata_query(sources: list[SchemaSource]) -> str:
    """
    Query collecting metadata for the source tables, columns and keys.

    The query is a union of up to tree sub-queries selecting:
    - information about the source text columns;
    - information about explicitly defined keys of the source tables;
    - information about primary keys of the source tables where the explicit
      keys are not provided.

    We have to cast the output columns to a VARCHAR of defined size to avoid
    potential mismatch between the types of the declared output columns and
    the actual types in the result returned by the metadata query.
    """
    text_columns = get_texts(sources)
    key_columns = get_keys(sources)
    pk_tables = get_pk_tables(sources, "S")
    meta_columns = dedent(f"""
        CAST(C.COLUMN_SCHEMA AS {METADATA_CAST_TYPE}) AS COLUMN_SCHEMA,
        CAST(C.COLUMN_TABLE AS {METADATA_CAST_TYPE}) AS COLUMN_TABLE,
        CAST(C.COLUMN_NAME AS {METADATA_CAST_TYPE}) AS COLUMN_NAME,
        CAST(C.COLUMN_TYPE AS {METADATA_CAST_TYPE}) AS COLUMN_TYPE
    """)

    queries = [dedent(f"""
        SELECT FALSE AS "{IS_KEY_COLUMN}", {meta_columns}
        FROM SYS.EXA_ALL_COLUMNS C
        WHERE {condition(text_columns)}
    """)]
    if key_columns:
        queries.append(dedent(f"""
            SELECT TRUE AS "{IS_KEY_COLUMN}", {meta_columns}
            FROM SYS.EXA_ALL_COLUMNS C
            WHERE {condition(key_columns)}
        """))
    if pk_tables:
        queries.append(dedent(f"""
            SELECT TRUE AS "{IS_KEY_COLUMN}", {meta_columns}
            FROM SYS.EXA_ALL_CONSTRAINT_COLUMNS S JOIN SYS.EXA_ALL_COLUMNS C
            ON S.CONSTRAINT_SCHEMA=C.COLUMN_SCHEMA
            AND S.CONSTRAINT_TABLE=C.COLUMN_TABLE
            AND S.COLUMN_NAME=C.COLUMN_NAME
            WHERE ({condition(pk_tables)}) AND S.CONSTRAINT_TYPE='PRIMARY KEY'
        """))

    return "\nUNION ALL\n".join(queries)
