from unittest import mock

import pytest
from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query.result.python_query_result import (
    PythonQueryResult,
)
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.query_handler.result import Finish
from exasol.analytics.schema import (
    SchemaName,
    TableLikeNameImpl,
)

from exasol.ai.text.extractors.source_table_extractor import (
    NameSelector,
    SchemaSource,
    SelectorPredicate,
    TableSource,
)
from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.stage_graph.input_output.source_table_output import (
    SourceTableOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.metadata_table import (
    DOC_ID_COLUMN,
    SOURCE_COLUMN,
    MetadataTable,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.utils import (
    IS_KEY_COLUMN,
    METADATA_CAST_TYPE,
    get_source_tables,
    validate_source_tables,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table_query_handler import (
    SourceTableQueryHandler,
)
from exasol.ai.text.impl.utils.sql_utils import (
    MAX_NAME_LENGTH,
    integer_column_spec,
)
from test.unit.utils.test_stage_graph_utils import collapse_spaces

_DOCUMENT_TABLE = "documents"
_SCHEMA = "my_output_schema"


@pytest.fixture
def query_handler() -> SourceTableQueryHandler:
    log_span = mock.Mock(spec=LogSpan)
    sources = [
        SchemaSource(
            db_schema=NameSelector(
                pattern="my_schema1", predicate=SelectorPredicate.EXACT
            ),
            tables=[
                TableSource(
                    table=NameSelector(
                        pattern="my_table11", predicate=SelectorPredicate.EXACT
                    ),
                    columns=[
                        NameSelector(
                            pattern="my_column111", predicate=SelectorPredicate.EXACT
                        ),
                        NameSelector(
                            pattern="my_column112", predicate=SelectorPredicate.LIKE
                        ),
                    ],
                    keys=[
                        NameSelector(
                            pattern="my_key111", predicate=SelectorPredicate.EXACT
                        ),
                    ],
                ),
                TableSource(
                    table=NameSelector(
                        pattern="my_table12", predicate=SelectorPredicate.REGEXP_LIKE
                    ),
                    columns=[
                        NameSelector(
                            pattern="my_column121", predicate=SelectorPredicate.EXACT
                        ),
                        NameSelector(
                            pattern="my_column122",
                            predicate=SelectorPredicate.REGEXP_LIKE,
                        ),
                    ],
                    keys=[
                        NameSelector(
                            pattern="my_key121", predicate=SelectorPredicate.LIKE
                        ),
                        NameSelector(
                            pattern="my_key122", predicate=SelectorPredicate.REGEXP_LIKE
                        ),
                    ],
                ),
            ],
        ),
        SchemaSource(
            db_schema=NameSelector(
                pattern="my_schema2", predicate=SelectorPredicate.LIKE
            ),
            tables=[
                TableSource(
                    table=NameSelector(
                        pattern="my_table21", predicate=SelectorPredicate.LIKE
                    ),
                    columns=[
                        NameSelector(
                            pattern="my_column211",
                            predicate=SelectorPredicate.REGEXP_LIKE,
                        )
                    ],
                )
            ],
        ),
    ]
    test_output = TableLikeNameImpl(_DOCUMENT_TABLE, SchemaName(_SCHEMA))
    test_handler_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[SQLStageInputOutput()], result_bucketfs_location=None
    )  # type: ignore
    test_context = mock.Mock(spec=ScopeQueryHandlerContext)
    return SourceTableQueryHandler(
        sources=sources,
        document_table_name=test_output,
        parameter=test_handler_input,
        query_handler_context=test_context,
        log_span=log_span,
    )


@pytest.fixture
def query_result(query_handler) -> QueryResult:
    query_result_data = [
        (False, "my_schema2", "my_table21", "my_column211", "VARCHAR(200000)"),
        (False, "my_schema1", "my_table12", "my_column122", "VARCHAR(200000)"),
        (False, "my_schema1", "my_table11", "my_column112", "VARCHAR(50000)"),
        (True, "my_schema1", "my_table11", "my_key111", "DECIMAL(18,0)"),
        (True, "my_schema1", "my_table12", "my_key121", "CHAR(10)"),
        (True, "my_schema1", "my_table12", "my_key122", "DECIMAL(18,0)"),
        (False, "my_schema1", "my_table12", "my_column121", "VARCHAR(100000)"),
        (True, "my_schema2", "my_table21", "my_key211", "VARCHAR(100)"),
        (False, "my_schema1", "my_table11", "my_column111", "VARCHAR(100000)"),
    ]
    return PythonQueryResult(query_result_data, query_handler.select_metadata_output)


@pytest.fixture
def source_tables() -> list[MetadataTable]:
    return [
        MetadataTable(
            table=TableLikeNameImpl("my_table11", SchemaName("my_schema1")),
            columns=[
                ColumnSpec(name="my_column111", sql_type="VARCHAR(100000)"),
                ColumnSpec(name="my_column112", sql_type="VARCHAR(50000)"),
            ],
            keys=[ColumnSpec(name="my_key111", sql_type="DECIMAL(18,0)")],
        ),
        MetadataTable(
            table=TableLikeNameImpl("my_table12", SchemaName("my_schema1")),
            columns=[
                ColumnSpec(name="my_column121", sql_type="VARCHAR(100000)"),
                ColumnSpec(name="my_column122", sql_type="VARCHAR(200000)"),
            ],
            keys=[
                ColumnSpec(name="my_key121", sql_type="CHAR(10)"),
                ColumnSpec(name="my_key122", sql_type="DECIMAL(18,0)"),
            ],
        ),
        MetadataTable(
            table=TableLikeNameImpl("my_table21", SchemaName("my_schema2")),
            columns=[ColumnSpec(name="my_column211", sql_type="VARCHAR(200000)")],
            keys=[ColumnSpec(name="my_key211", sql_type="VARCHAR(100)")],
        ),
    ]


@pytest.fixture
def expected_select_metadata():
    return collapse_spaces(f"""
        SELECT
            FALSE AS "{IS_KEY_COLUMN}",
            CAST(C.COLUMN_SCHEMA AS {METADATA_CAST_TYPE}) AS COLUMN_SCHEMA,
            CAST(C.COLUMN_TABLE AS {METADATA_CAST_TYPE}) AS COLUMN_TABLE,
            CAST(C.COLUMN_NAME AS {METADATA_CAST_TYPE}) AS COLUMN_NAME,
            CAST(C.COLUMN_TYPE AS {METADATA_CAST_TYPE}) AS COLUMN_TYPE
        FROM SYS.EXA_ALL_COLUMNS C
        WHERE
            COLUMN_SCHEMA='my_schema1' AND (
            COLUMN_TABLE='my_table11' AND (
            COLUMN_NAME='my_column111' OR COLUMN_NAME LIKE 'my_column112') OR
            COLUMN_TABLE REGEXP_LIKE 'my_table12' AND (
            COLUMN_NAME='my_column121' OR COLUMN_NAME REGEXP_LIKE 'my_column122')) OR
            COLUMN_SCHEMA LIKE 'my_schema2' AND (
            COLUMN_TABLE LIKE 'my_table21' AND (
            COLUMN_NAME REGEXP_LIKE 'my_column211'))
        UNION ALL
        SELECT
            TRUE AS "{IS_KEY_COLUMN}",
            CAST(C.COLUMN_SCHEMA AS {METADATA_CAST_TYPE}) AS COLUMN_SCHEMA,
            CAST(C.COLUMN_TABLE AS {METADATA_CAST_TYPE}) AS COLUMN_TABLE,
            CAST(C.COLUMN_NAME AS {METADATA_CAST_TYPE}) AS COLUMN_NAME,
            CAST(C.COLUMN_TYPE AS {METADATA_CAST_TYPE}) AS COLUMN_TYPE
        FROM SYS.EXA_ALL_COLUMNS C
        WHERE
            COLUMN_SCHEMA='my_schema1' AND (
            COLUMN_TABLE='my_table11' AND (
            COLUMN_NAME='my_key111') OR
            COLUMN_TABLE REGEXP_LIKE 'my_table12' AND (
            COLUMN_NAME LIKE 'my_key121' OR COLUMN_NAME REGEXP_LIKE 'my_key122'))
        UNION ALL
        SELECT
            TRUE AS "{IS_KEY_COLUMN}",
            CAST(C.COLUMN_SCHEMA AS {METADATA_CAST_TYPE}) AS COLUMN_SCHEMA,
            CAST(C.COLUMN_TABLE AS {METADATA_CAST_TYPE}) AS COLUMN_TABLE,
            CAST(C.COLUMN_NAME AS {METADATA_CAST_TYPE}) AS COLUMN_NAME,
            CAST(C.COLUMN_TYPE AS {METADATA_CAST_TYPE}) AS COLUMN_TYPE
        FROM SYS.EXA_ALL_CONSTRAINT_COLUMNS S JOIN SYS.EXA_ALL_COLUMNS C
            ON S.CONSTRAINT_SCHEMA=C.COLUMN_SCHEMA
            AND S.CONSTRAINT_TABLE=C.COLUMN_TABLE
            AND S.COLUMN_NAME=C.COLUMN_NAME
        WHERE
        (
            S.CONSTRAINT_SCHEMA LIKE 'my_schema2' AND (
            S.CONSTRAINT_TABLE LIKE 'my_table21')
        ) AND S.CONSTRAINT_TYPE='PRIMARY KEY'
    """)


@pytest.fixture
def expected_document() -> str:
    return f'"{_SCHEMA}"."{_DOCUMENT_TABLE}"'


@pytest.fixture
def expected_mappings() -> list[str]:
    return [
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema1_my_table11"',
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema1_my_table12"',
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema2_my_table21"',
    ]


@pytest.fixture
def expected_views() -> list[str]:
    return [
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema1_my_table11_VIEW"',
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema1_my_table12_VIEW"',
        f'"{_SCHEMA}"."{_DOCUMENT_TABLE}_my_schema2_my_table21_VIEW"',
    ]


@pytest.fixture
def expected_create_document(expected_document) -> str:
    inputs = TextInput()
    doc_id, text_begin, text_end = inputs.text_span.quoted_column_names
    return (
        f"CREATE TABLE IF NOT EXISTS {expected_document}"
        f"({doc_id} INTEGER, {text_begin} INTEGER, "
        f"{text_end} INTEGER, {inputs.text.quoted_name} VARCHAR(2000000), "
        f"DISTRIBUTE BY {doc_id})"
    )


@pytest.fixture
def expected_create_mappings(expected_mappings) -> list[str]:
    return [
        collapse_spaces(
            f"CREATE TABLE IF NOT EXISTS {expected_mappings[0]}"
            '("my_key111" DECIMAL(18,0), '
            f'"{DOC_ID_COLUMN}" INTEGER, "{SOURCE_COLUMN}" VARCHAR({MAX_NAME_LENGTH}), '
            f'DISTRIBUTE BY "{DOC_ID_COLUMN}")'
        ),
        collapse_spaces(
            f"CREATE TABLE IF NOT EXISTS {expected_mappings[1]}"
            '("my_key121" CHAR(10), "my_key122" DECIMAL(18,0), '
            f'"{DOC_ID_COLUMN}" INTEGER, "{SOURCE_COLUMN}" VARCHAR({MAX_NAME_LENGTH}), '
            f'DISTRIBUTE BY "{DOC_ID_COLUMN}")'
        ),
        collapse_spaces(
            f"CREATE TABLE IF NOT EXISTS {expected_mappings[2]}"
            '("my_key211" VARCHAR(100), '
            f'"{DOC_ID_COLUMN}" INTEGER, "{SOURCE_COLUMN}" VARCHAR({MAX_NAME_LENGTH}), '
            f'DISTRIBUTE BY "{DOC_ID_COLUMN}")'
        ),
    ]


@pytest.fixture
def expected_mapping_updates(expected_document, expected_mappings) -> list[str]:
    doc_id, _, _ = TextInput().text_span.quoted_column_names
    return [
        collapse_spaces(f"""
            INSERT INTO {expected_mappings[0]}("my_key111", "{DOC_ID_COLUMN}", "{SOURCE_COLUMN}")
            SELECT S."my_key111", L.LAST_ID + ROW_NUMBER() OVER (), T.SRC
            FROM "my_schema1"."my_table11" S
            LEFT OUTER JOIN {expected_mappings[0]} M ON S."my_key111"=M."my_key111"
            CROSS JOIN (SELECT * FROM (VALUES ('my_column111'), ('my_column112')) AS TEXTS(SRC)) T
            CROSS JOIN (SELECT COALESCE(MAX({doc_id}), 0) AS LAST_ID
            FROM {expected_document}) L WHERE M."my_key111" IS NULL
        """),
        collapse_spaces(f"""
            INSERT INTO {expected_mappings[1]}("my_key121", "my_key122", "{DOC_ID_COLUMN}", "{SOURCE_COLUMN}")
            SELECT S."my_key121", S."my_key122", L.LAST_ID + ROW_NUMBER() OVER (), T.SRC
            FROM "my_schema1"."my_table12" S
            LEFT OUTER JOIN {expected_mappings[1]} M
            ON S."my_key121"=M."my_key121" AND S."my_key122"=M."my_key122"
            CROSS JOIN (SELECT * FROM (VALUES ('my_column121'), ('my_column122')) AS TEXTS(SRC)) T
            CROSS JOIN (SELECT COALESCE(MAX({doc_id}), 0) AS LAST_ID
            FROM {expected_document}) L WHERE M."my_key121" IS NULL AND M."my_key122" IS NULL
        """),
        collapse_spaces(f"""
            INSERT INTO {expected_mappings[2]}("my_key211", "{DOC_ID_COLUMN}", "{SOURCE_COLUMN}")
            SELECT S."my_key211", L.LAST_ID + ROW_NUMBER() OVER (), T.SRC
            FROM "my_schema2"."my_table21" S
            LEFT OUTER JOIN {expected_mappings[2]} M ON S."my_key211"=M."my_key211"
            CROSS JOIN (SELECT * FROM (VALUES ('my_column211')) AS TEXTS(SRC)) T
            CROSS JOIN (SELECT COALESCE(MAX({doc_id}), 0) AS LAST_ID
            FROM {expected_document}) L WHERE M."my_key211" IS NULL
        """),
    ]


@pytest.fixture
def expected_document_updates(expected_document, expected_mappings) -> list[str]:
    inputs = TextInput()
    doc_id, text_begin, text_end = inputs.text_span.quoted_column_names
    text = inputs.text.quoted_name
    return [
        collapse_spaces(f"""
            INSERT INTO {expected_document}({doc_id}, {text_begin}, {text_end}, {text})
            WITH MAP_UPDATE AS (
                SELECT M.*
                FROM {expected_mappings[0]} M
                LEFT OUTER JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
                WHERE D.{doc_id} IS NULL
            )
            SELECT M."{DOC_ID_COLUMN}", 0, LENGTH(S."my_column111"), S."my_column111"
            FROM "my_schema1"."my_table11" S
            JOIN MAP_UPDATE M ON S."my_key111"=M."my_key111"
            WHERE M."{SOURCE_COLUMN}" = 'my_column111'
            AND S."my_column111" IS NOT NULL
            UNION ALL
            SELECT M."{DOC_ID_COLUMN}", 0, LENGTH(S."my_column112"), S."my_column112"
            FROM "my_schema1"."my_table11" S
            JOIN MAP_UPDATE M ON S."my_key111"=M."my_key111"
            WHERE M."{SOURCE_COLUMN}" = 'my_column112'
            AND S."my_column112" IS NOT NULL
        """),
        collapse_spaces(f"""
            INSERT INTO {expected_document}({doc_id}, {text_begin}, {text_end}, {text})
            WITH MAP_UPDATE AS (
                SELECT M.*
                FROM {expected_mappings[1]} M
                LEFT OUTER JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
                WHERE D.{doc_id} IS NULL
            )
            SELECT M."{DOC_ID_COLUMN}", 0, LENGTH(S."my_column121"), S."my_column121"
            FROM "my_schema1"."my_table12" S
            JOIN MAP_UPDATE M ON S."my_key121"=M."my_key121" AND S."my_key122"=M."my_key122"
            WHERE M."{SOURCE_COLUMN}" = 'my_column121'
            AND S."my_column121" IS NOT NULL
            UNION ALL
            SELECT M."{DOC_ID_COLUMN}", 0, LENGTH(S."my_column122"), S."my_column122"
            FROM "my_schema1"."my_table12" S
            JOIN MAP_UPDATE M ON S."my_key121"=M."my_key121" AND S."my_key122"=M."my_key122"
            WHERE M."{SOURCE_COLUMN}" = 'my_column122'
            AND S."my_column122" IS NOT NULL
        """),
        collapse_spaces(f"""
            INSERT INTO {expected_document}({doc_id}, {text_begin}, {text_end}, {text})
            WITH MAP_UPDATE AS (
                SELECT M.*
                FROM {expected_mappings[2]} M
                LEFT OUTER JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
                WHERE D.{doc_id} IS NULL
            )
            SELECT M."{DOC_ID_COLUMN}", 0, LENGTH(S."my_column211"), S."my_column211"
            FROM "my_schema2"."my_table21" S
            JOIN MAP_UPDATE M ON S."my_key211"=M."my_key211"
            WHERE M."{SOURCE_COLUMN}" = 'my_column211'
            AND S."my_column211" IS NOT NULL
        """),
    ]


@pytest.fixture
def expected_mapping_cleanups(expected_document, expected_mappings) -> list[str]:
    doc_id, _, _ = TextInput().text_span.quoted_column_names
    return [collapse_spaces(f"""
            DELETE FROM {expected_map_n}
            WHERE "{DOC_ID_COLUMN}" NOT IN (
                SELECT {doc_id} FROM {expected_document}
            )
        """) for expected_map_n in expected_mappings]


@pytest.fixture
def expected_source_views(
    expected_document, expected_mappings, expected_views
) -> list[str]:
    doc_id, text_begin, text_end = TextInput().text_span.quoted_column_names
    return [
        collapse_spaces(f"""
            CREATE OR REPLACE VIEW {expected_views[0]} AS
            SELECT S.*, D.{doc_id}, D.{text_begin}, D.{text_end}, M."{SOURCE_COLUMN}"
            FROM "my_schema1"."my_table11" S
            JOIN {expected_mappings[0]} M ON S."my_key111"=M."my_key111"
            JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
        """),
        collapse_spaces(f"""
            CREATE OR REPLACE VIEW {expected_views[1]} AS
            SELECT S.*, D.{doc_id}, D.{text_begin}, D.{text_end}, M."{SOURCE_COLUMN}"
            FROM "my_schema1"."my_table12" S
            JOIN {expected_mappings[1]} M
            ON S."my_key121"=M."my_key121" AND S."my_key122"=M."my_key122"
            JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
    """),
        collapse_spaces(f"""
            CREATE OR REPLACE VIEW {expected_views[2]} AS
            SELECT S.*, D.{doc_id}, D.{text_begin}, D.{text_end}, M."{SOURCE_COLUMN}"
            FROM "my_schema2"."my_table21" S
            JOIN {expected_mappings[2]} M ON S."my_key211"=M."my_key211"
            JOIN {expected_document} D ON M."{DOC_ID_COLUMN}" = D.{doc_id}
    """),
    ]


def test_get_source_tables(query_handler, query_result, source_tables):
    sources = get_source_tables(query_result)
    validate_source_tables(sources)

    def t_key(tbl: MetadataTable):
        return tbl.table.fully_qualified

    def c_key(col: ColumnSpec):
        return col.name

    sources = sorted(sources, key=t_key)
    expected_sources = sorted(source_tables, key=t_key)
    assert len(sources) == len(expected_sources)
    assert all(
        (
            t1.table == t2.table
            and sorted(t1.columns, key=c_key) == sorted(t2.columns, key=c_key)
            and sorted(t1.keys, key=c_key) == sorted(t2.keys, key=c_key)
        )
        for t1, t2 in zip(sources, expected_sources)
    )


def test_validate_source_tables_key(query_handler, source_tables):
    query_result_data = [
        (False, "my_schema2", "my_table21", "my_column211", "VARCHAR(200000)"),
        (False, "my_schema1", "my_table12", "my_column122", "VARCHAR(200000)"),
        (False, "my_schema1", "my_table11", "my_column112", "VARCHAR(50000)"),
        (True, "my_schema1", "my_table12", "my_key121", "CHAR(10)"),
        (True, "my_schema1", "my_table12", "my_key122", "DECIMAL(18,0)"),
        (False, "my_schema1", "my_table12", "my_column121", "VARCHAR(100000)"),
        (True, "my_schema2", "my_table21", "my_key211", "VARCHAR(100)"),
        (False, "my_schema1", "my_table11", "my_column111", "VARCHAR(100000)"),
    ]
    query_result = PythonQueryResult(
        query_result_data, query_handler.select_metadata_output
    )

    sources = get_source_tables(query_result)
    with pytest.raises(Exception):
        validate_source_tables(sources)


def test_validate_source_tables_type(query_handler, source_tables):
    query_result_data = [
        (False, "my_schema2", "my_table21", "my_column211", "VARCHAR(200000)"),
        (False, "my_schema1", "my_table12", "my_column122", "DECIMAL(18,0)"),
        (False, "my_schema1", "my_table11", "my_column112", "VARCHAR(50000)"),
        (True, "my_schema1", "my_table11", "my_key111", "DECIMAL(18,0)"),
        (True, "my_schema1", "my_table12", "my_key121", "CHAR(10)"),
        (True, "my_schema1", "my_table12", "my_key122", "DECIMAL(18,0)"),
        (False, "my_schema1", "my_table12", "my_column121", "VARCHAR(100000)"),
        (True, "my_schema2", "my_table21", "my_key211", "VARCHAR(100)"),
        (False, "my_schema1", "my_table11", "my_column111", "DOUBLE"),
    ]
    query_result = PythonQueryResult(
        query_result_data, query_handler.select_metadata_output
    )

    sources = get_source_tables(query_result)
    with pytest.raises(Exception):
        validate_source_tables(sources)


def test_select_metadata_query(query_handler, expected_select_metadata):
    select_metadata = collapse_spaces(query_handler.select_metadata_query)
    assert select_metadata == expected_select_metadata


def test_create_document_query(query_handler, expected_create_document):
    create_document = collapse_spaces(
        query_handler.document_table.create_document_query.query_string
    )
    assert create_document == expected_create_document


def test_get_create_mapping_query(
    query_handler, source_tables, expected_create_mappings
):
    for i, tbl in enumerate(source_tables):
        assert (
            collapse_spaces(
                query_handler.get_mapping_table(tbl)
                .get_create_mapping_query()
                .query_string
            )
            == expected_create_mappings[i]
        )


def test_get_mapping_update_query(
    query_handler, source_tables, expected_mapping_updates
):
    for i, tbl in enumerate(source_tables):
        assert (
            collapse_spaces(
                query_handler.get_mapping_table(tbl)
                .get_mapping_update_query()
                .query_string
            )
            == expected_mapping_updates[i]
        )


def test_get_document_update_query(
    query_handler, source_tables, expected_document_updates
):
    for i, tbl in enumerate(source_tables):
        mapping_table = query_handler.get_mapping_table(tbl)
        assert (
            collapse_spaces(
                query_handler.document_table.get_document_update_query(
                    mapping_table
                ).query_string
            )
            == expected_document_updates[i]
        )


def test_get_mapping_clean_query(
    query_handler, source_tables, expected_mapping_cleanups
):
    for i, tbl in enumerate(source_tables):
        assert (
            collapse_spaces(
                query_handler.get_mapping_table(tbl)
                .get_mapping_clean_query()
                .query_string
            )
            == expected_mapping_cleanups[i]
        )


def test_get_source_view_query(query_handler, source_tables, expected_source_views):
    for i, tbl in enumerate(source_tables):
        mapping_table = query_handler.get_mapping_table(tbl)
        assert (
            collapse_spaces(
                query_handler.document_table.get_source_view_query(
                    mapping_table
                ).query_string
            )
            == expected_source_views[i]
        )


def test_start(query_handler, expected_select_metadata):
    result = query_handler.start()
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    expected_query_list = [
        collapse_spaces(query_handler.start_audit_query.query_string)
    ]
    assert query_list == expected_query_list
    assert collapse_spaces(result.input_query.query_string) == expected_select_metadata


def test_handle_query_result1(
    query_handler,
    query_result,
    expected_create_document,
    expected_create_mappings,
    expected_mapping_updates,
    expected_document_updates,
    expected_mapping_cleanups,
    expected_source_views,
):
    result = query_handler.handle_query_result(query_result)
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    # It is difficult to match the result queries to the expectation since the order
    # of columns and keys in the queries may be different. But we can at least check
    # the number of them.
    expected_length = (
        len(expected_create_mappings)
        + len(expected_mapping_updates)
        + len(expected_document_updates)
        + len(expected_mapping_cleanups)
        + len(expected_source_views)
        + 3
    )
    assert len(query_list) == expected_length
    assert query_list[0] == expected_create_document
    assert query_list[-1] == query_handler.commit_output_query.query_string
    assert query_list[-2] == query_handler.end_audit_query.query_string


def test_handle_query_result2(query_handler, query_result, source_tables):
    query_handler.handle_query_result(query_result)
    query_result = PythonQueryResult([(1.0,)], [integer_column_spec("ONE").to_column()])
    result = query_handler.handle_query_result(query_result)
    assert isinstance(result, Finish)
    assert isinstance(result.result, SourceTableOutput)
    expected_table_list = {tbl.table for tbl in source_tables}
    assert set(result.result.source_mapping.keys()) == expected_table_list
