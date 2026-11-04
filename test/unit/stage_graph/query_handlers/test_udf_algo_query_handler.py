from dataclasses import dataclass
from typing import cast
from unittest import mock

import pytest
from exasol.analytics.audit.columns import BaseAuditColumns
from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query.select import (
    AuditQuery,
    LogSpan,
)
from exasol.analytics.query_handler.result import Finish
from exasol.analytics.schema import (
    SchemaName,
    TableLikeNameImpl,
    UDFNameImpl,
)

from exasol.ai.text.impl.nlp.algorithms.testing.span_size_logger import SpanSizeOutput
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import (
    AlgoNormalisedIOData,
)
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_query_handler import (
    LOOKUP_ID_COLUMN,
    ROWID_COLUMN,
    UDFAlgoQueryHandler,
)
from exasol.ai.text.impl.utils.sql_utils import (
    sql_text_value,
    varchar_column_spec,
)
from test.unit.utils.test_stage_graph_utils import collapse_spaces

_INPUT_TABLE = "input_table"
_OUTPUT_TABLE = "output_table"
_SCHEMA = "my_schema"
_UDF_NAME = "my_udf"
_LANGUAGE_ALIAS = "my_language_alias"


@dataclass(frozen=True)
class _TestSQLStageInput(SQLStageInputOutput):
    pass


def create_test_query_handler(udf_runner: UDFRunner) -> UDFAlgoQueryHandler:
    test_schema = SchemaName(_SCHEMA)
    test_input = TableLikeNameImpl(_INPUT_TABLE, test_schema)
    test_output = TableLikeNameImpl(_OUTPUT_TABLE, test_schema)
    test_handler_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[_TestSQLStageInput()], result_bucketfs_location=None
    )  # type: ignore
    test_context = mock.Mock(spec=ScopeQueryHandlerContext)
    test_context.get_temporary_table_name.return_value = TableLikeNameImpl(
        f"{_OUTPUT_TABLE}_TMP", test_schema
    )
    test_context.get_temporary_udf_name.return_value = UDFNameImpl(
        _UDF_NAME, test_schema
    )
    log_span = mock.Mock(spec=LogSpan)
    return UDFAlgoQueryHandler(
        udf_runner=udf_runner,
        input_table=test_input,
        output_table=test_output,
        language_alias=_LANGUAGE_ALIAS,
        parameter=test_handler_input,
        query_handler_context=test_context,
        parallelism_per_node=2,
        log_span=log_span,
    )


@pytest.fixture
def span_size_runner() -> UDFRunner:
    outputs = SpanSizeOutput(setup=varchar_column_spec("SETUP"))
    return UDFRunner(
        algorithm={"algorithm_name": "span-size-logger", "outputs": outputs},
        batch_size=10,
    )


@pytest.fixture
def span_size_qh(span_size_runner) -> UDFAlgoQueryHandler:
    """
    Returns a query handler for the 'span-size-logger' algorithm. This should be the
    case when no normalisation is required.
    """
    return create_test_query_handler(span_size_runner)


@pytest.fixture
def span_size_expected_create_output() -> str:
    """
    Expected "create_output_query" for the 'span-size-logger' algorithm
    (no normalisation case).
    """
    return collapse_spaces(f"""
        CREATE TABLE IF NOT EXISTS "{_SCHEMA}"."{_OUTPUT_TABLE}"(
            "TEXT_DOC_ID" INTEGER,
            "TEXT_CHAR_BEGIN" INTEGER,
            "TEXT_CHAR_END" INTEGER,
            "SPAN_SIZE" INTEGER,
            "ERROR_MESSAGE" VARCHAR(2000000),
            "SETUP" VARCHAR(2000000)
        )
    """)


@pytest.fixture
def span_size_expected_run_udf(span_size_runner) -> str:
    """
    Expected "run_udf_query" for the 'span-size-logger' algorithm
    (no normalisation case).
    """
    setup = sql_text_value(span_size_runner.algo_setup)
    return collapse_spaces(f"""
        INSERT INTO "{_SCHEMA}"."{_OUTPUT_TABLE}"(
            "TEXT_DOC_ID",
            "TEXT_CHAR_BEGIN",
            "TEXT_CHAR_END",
            "SPAN_SIZE",
            "ERROR_MESSAGE",
            "SETUP"
        )
        WITH T2 AS (
            SELECT
                "TEXT_DOC_ID",
                "TEXT_CHAR_BEGIN",
                "TEXT_CHAR_END"
            FROM "{_SCHEMA}"."{_OUTPUT_TABLE}" T3
            WHERE T3."SETUP"={setup}
        )
        WITH T1 AS (
            SELECT
                ROWID AS "{ROWID_COLUMN}",
                "TEXT_DOC_ID",
                "TEXT_CHAR_BEGIN",
                "TEXT_CHAR_END"
            FROM "{_SCHEMA}"."{_INPUT_TABLE}"
        )
        SELECT "{_SCHEMA}"."{_UDF_NAME}"(
            T1."TEXT_DOC_ID",
            T1."TEXT_CHAR_BEGIN",
            T1."TEXT_CHAR_END"
        )
        FROM T1
        LEFT OUTER JOIN T2 ON
            T1."TEXT_DOC_ID"=T2."TEXT_DOC_ID" AND
            T1."TEXT_CHAR_BEGIN"=T2."TEXT_CHAR_BEGIN" AND
            T1."TEXT_CHAR_END"=T2."TEXT_CHAR_END"
        WHERE
            T2."TEXT_DOC_ID" IS NULL AND
            T2."TEXT_CHAR_BEGIN" IS NULL AND
            T2."TEXT_CHAR_END" IS NULL
        GROUP BY IPROC(), MOD(T1."{ROWID_COLUMN}", 2)
    """)


@pytest.fixture
def span_category_runner() -> UDFRunner:
    return UDFRunner(
        algorithm={"algorithm_name": "span-size-categorizer"}, batch_size=10
    )


@pytest.fixture
def span_category_qh(span_category_runner) -> UDFAlgoQueryHandler:
    """
    Returns a query handler for the 'span-size-categorizer' algorithm.
    This is the case when the output table gets normalised on two output columns -
    "SETUP" and "SIZE_CATEGORY"
    """
    return create_test_query_handler(span_category_runner)


@pytest.fixture
def span_category_expected_create_output() -> str:
    """
    Expected "create_output_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return collapse_spaces(f"""
        CREATE TABLE IF NOT EXISTS "{_SCHEMA}"."{_OUTPUT_TABLE}"(
            "TEXT_DOC_ID" INTEGER,
            "TEXT_CHAR_BEGIN" INTEGER,
            "TEXT_CHAR_END" INTEGER,
            "SIZE_CATEGORY" INTEGER,
            "ERROR_MESSAGE" VARCHAR(2000000),
            "SETUP" INTEGER,
            DISTRIBUTE BY "TEXT_DOC_ID",
            PARTITION BY "TEXT_DOC_ID"
        )
    """)


@pytest.fixture
def span_category_expected_create_tmp_output() -> str:
    """
    Expected "create_tmp_output_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return collapse_spaces(f"""
        CREATE OR REPLACE TABLE "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"(
            "TEXT_DOC_ID" INTEGER,
            "TEXT_CHAR_BEGIN" INTEGER,
            "TEXT_CHAR_END" INTEGER,
            "SIZE_CATEGORY" VARCHAR(2000000),
            "ERROR_MESSAGE" VARCHAR(2000000),
            "SETUP" VARCHAR(2000000),
            DISTRIBUTE BY "TEXT_DOC_ID"
        )
    """)


@pytest.fixture
def span_category_expected_create_lookup_table() -> list[str]:
    """
    Expected "create_lookup_queries" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return [
        collapse_spaces(f"""
            CREATE TABLE IF NOT EXISTS "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SIZE_CATEGORY"(
                "{LOOKUP_ID_COLUMN}" INTEGER IDENTITY PRIMARY KEY,
                "SIZE_CATEGORY" VARCHAR(2000000))
        """),
        collapse_spaces(f"""
            CREATE TABLE IF NOT EXISTS "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SETUP"(
                "{LOOKUP_ID_COLUMN}" INTEGER IDENTITY PRIMARY KEY,
                "SETUP" VARCHAR(2000000))
        """),
    ]


@pytest.fixture
def span_category_expected_create_view() -> str:
    """
    Expected "create_view_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return collapse_spaces(f"""
        CREATE OR REPLACE VIEW "{_SCHEMA}"."{_OUTPUT_TABLE}_VIEW" AS
        SELECT
            T1."TEXT_DOC_ID",
            T1."TEXT_CHAR_BEGIN",
            T1."TEXT_CHAR_END",
            T2."SIZE_CATEGORY",
            T1."ERROR_MESSAGE",
            T3."SETUP"
        FROM "{_SCHEMA}"."{_OUTPUT_TABLE}" T1
        LEFT OUTER JOIN "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SIZE_CATEGORY" T2
        ON T1."SIZE_CATEGORY" = T2."{LOOKUP_ID_COLUMN}"
        LEFT OUTER JOIN "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SETUP" T3
        ON T1."SETUP" = T3."{LOOKUP_ID_COLUMN}"
    """)


@pytest.fixture
def span_category_expected_run_udf(span_category_runner) -> str:
    """
    Expected "run_udf_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    setup = sql_text_value(span_category_runner.algo_setup)
    return collapse_spaces(f"""
        INSERT INTO "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"(
            "TEXT_DOC_ID",
            "TEXT_CHAR_BEGIN",
            "TEXT_CHAR_END",
            "SIZE_CATEGORY",
            "ERROR_MESSAGE",
            "SETUP"
        )
        WITH T2 AS (
            SELECT
                "TEXT_DOC_ID",
                "TEXT_CHAR_BEGIN",
                "TEXT_CHAR_END"
            FROM "{_SCHEMA}"."{_OUTPUT_TABLE}" T3
            JOIN "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SETUP" T4 ON
                T3."SETUP"=T4."{LOOKUP_ID_COLUMN}"
            WHERE T4."SETUP"={setup}
        )
        WITH T1 AS (
            SELECT
                ROWID AS "{ROWID_COLUMN}",
                "TEXT_DOC_ID",
                "TEXT_CHAR_BEGIN",
                "TEXT_CHAR_END"
            FROM "{_SCHEMA}"."{_INPUT_TABLE}"
        )
        SELECT "{_SCHEMA}"."{_UDF_NAME}"(
            T1."TEXT_DOC_ID",
            T1."TEXT_CHAR_BEGIN",
            T1."TEXT_CHAR_END"
        )
        FROM T1
        LEFT OUTER JOIN T2 ON
            T1."TEXT_DOC_ID"=T2."TEXT_DOC_ID" AND
            T1."TEXT_CHAR_BEGIN"=T2."TEXT_CHAR_BEGIN" AND
            T1."TEXT_CHAR_END"=T2."TEXT_CHAR_END"
        WHERE
            T2."TEXT_DOC_ID" IS NULL AND
            T2."TEXT_CHAR_BEGIN" IS NULL AND
            T2."TEXT_CHAR_END" IS NULL
        GROUP BY IPROC(), MOD(T1."{ROWID_COLUMN}", 2)
    """)


@pytest.fixture
def span_category_expected_extract_lookup() -> list[str]:
    """
    Expected "extract_lookup_queries" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return [
        collapse_spaces(f"""
            MERGE INTO "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SIZE_CATEGORY" T
            USING (SELECT DISTINCT "SIZE_CATEGORY"
            FROM "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"
            WHERE "SIZE_CATEGORY" IS NOT NULL) AS S
            ON T."SIZE_CATEGORY" = S."SIZE_CATEGORY"
            WHEN NOT MATCHED THEN INSERT("SIZE_CATEGORY") VALUES(S."SIZE_CATEGORY")
        """),
        collapse_spaces(f"""
            MERGE INTO "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SETUP" T
            USING (SELECT DISTINCT "SETUP"
            FROM "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"
            WHERE "SETUP" IS NOT NULL) AS S
            ON T."SETUP" = S."SETUP"
            WHEN NOT MATCHED THEN INSERT("SETUP") VALUES(S."SETUP")
        """),
    ]


@pytest.fixture
def span_category_expected_normalise() -> str:
    """
    Expected "normalise_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return collapse_spaces(f"""
        INSERT INTO "{_SCHEMA}"."{_OUTPUT_TABLE}"(
            "TEXT_DOC_ID",
            "TEXT_CHAR_BEGIN",
            "TEXT_CHAR_END",
            "SIZE_CATEGORY",
            "ERROR_MESSAGE",
            "SETUP"
        )
        SELECT
            T1."TEXT_DOC_ID",
            T1."TEXT_CHAR_BEGIN",
            T1."TEXT_CHAR_END",
            T2."{LOOKUP_ID_COLUMN}",
            T1."ERROR_MESSAGE",
            T3."{LOOKUP_ID_COLUMN}"
        FROM "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP" T1
        LEFT OUTER JOIN "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SIZE_CATEGORY" T2
        ON T1."SIZE_CATEGORY" = T2."SIZE_CATEGORY"
        LEFT OUTER JOIN "{_SCHEMA}"."{_OUTPUT_TABLE}_LOOKUP_SETUP" T3
        ON T1."SETUP" = T3."SETUP"
    """)


@pytest.fixture
def span_category_error_audit() -> str:
    """
    Expected "error_audit_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return collapse_spaces(f"""
        SELECT DISTINCT "ERROR_MESSAGE"
        AS {BaseAuditColumns.ERROR_MESSAGE.name.quoted_name}
        FROM "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"
    """)


@pytest.fixture
def span_category_expected_drop_tmp_output() -> str:
    """
    Expected "drop_tmp_output_query" for the 'span-size-categorizer' algorithm
    (normalisation case).
    """
    return f'DROP TABLE "{_SCHEMA}"."{_OUTPUT_TABLE}_TMP"'


def test_create_output_non_norm(span_size_qh, span_size_expected_create_output):
    assert (
        collapse_spaces(span_size_qh.create_output_query.query_string)
        == span_size_expected_create_output
    )


def test_run_udf_non_norm(span_size_qh, span_size_expected_run_udf):
    assert (
        collapse_spaces(span_size_qh.run_udf_query.query_string)
        == span_size_expected_run_udf
    )


def test_start_non_norm(
    span_size_qh, span_size_expected_create_output, span_size_expected_run_udf
):
    result = span_size_qh.start()
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    expected_query_list = [
        collapse_spaces(span_size_qh.start_audit_query.query_string),
        span_size_expected_create_output,
        collapse_spaces(span_size_qh.create_udf_query.query_string),
        span_size_expected_run_udf,
        collapse_spaces(span_size_qh.delete_udf_query.query_string),
        collapse_spaces(span_size_qh.end_audit_query.query_string),
        collapse_spaces(span_size_qh.commit_output_query.query_string),
    ]
    assert query_list == expected_query_list


def test_handle_query_result_no_norm(span_size_qh):
    query_result = mock.Mock(spec=QueryResult)
    handler_result = span_size_qh.handle_query_result(query_result)
    assert isinstance(handler_result, Finish)
    assert isinstance(handler_result.result, AlgoNormalisedIOData)
    result_io = cast(AlgoNormalisedIOData, handler_result.result)
    assert result_io.datasource.name == _OUTPUT_TABLE
    assert result_io.dataview is None
    assert result_io.lookup_tables is None


def test_create_output_norm(span_category_qh, span_category_expected_create_output):
    assert (
        collapse_spaces(span_category_qh.create_output_query.query_string)
        == span_category_expected_create_output
    )


def test_create_tmp_output(span_category_qh, span_category_expected_create_tmp_output):
    assert (
        collapse_spaces(span_category_qh.create_tmp_output_query.query_string)
        == span_category_expected_create_tmp_output
    )


def test_create_lookup_table(
    span_category_qh, span_category_expected_create_lookup_table
):
    assert [
        collapse_spaces(qry.query_string)
        for qry in span_category_qh.create_lookup_table_queries
    ] == span_category_expected_create_lookup_table


def test_create_view(span_category_qh, span_category_expected_create_view):
    assert (
        collapse_spaces(span_category_qh.create_view_query.query_string)
        == span_category_expected_create_view
    )


def test_run_udf_norm(span_category_qh, span_category_expected_run_udf):
    assert (
        collapse_spaces(span_category_qh.run_udf_query.query_string)
        == span_category_expected_run_udf
    )


def test_extract_lookup(span_category_qh, span_category_expected_extract_lookup):
    assert [
        collapse_spaces(qry.query_string)
        for qry in span_category_qh.extract_lookup_queries
    ] == span_category_expected_extract_lookup


def test_normalise(span_category_qh, span_category_expected_normalise):
    assert (
        collapse_spaces(span_category_qh.normalise_query.query_string)
        == span_category_expected_normalise
    )


def test_error_audit(span_category_qh, span_category_error_audit):
    audit_query = span_category_qh.error_audit_query
    assert isinstance(audit_query, AuditQuery)
    assert audit_query.query_string == span_category_error_audit
    assert audit_query.select_with_columns.output_columns == [
        BaseAuditColumns.ERROR_MESSAGE
    ]


def test_start_norm(
    span_category_qh,
    span_category_expected_create_output,
    span_category_expected_create_tmp_output,
    span_category_expected_create_lookup_table,
    span_category_expected_create_view,
    span_category_expected_run_udf,
    span_category_expected_extract_lookup,
    span_category_expected_normalise,
    span_category_error_audit,
    span_category_expected_drop_tmp_output,
):
    result = span_category_qh.start()
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    expected_query_list = [
        collapse_spaces(span_category_qh.start_audit_query.query_string),
        span_category_expected_create_output,
        span_category_expected_create_tmp_output,
        *span_category_expected_create_lookup_table,
        span_category_expected_create_view,
        collapse_spaces(span_category_qh.create_udf_query.query_string),
        span_category_expected_run_udf,
        collapse_spaces(span_category_qh.delete_udf_query.query_string),
        *span_category_expected_extract_lookup,
        span_category_expected_normalise,
        span_category_error_audit,
        span_category_expected_drop_tmp_output,
        collapse_spaces(span_category_qh.end_audit_query.query_string),
        collapse_spaces(span_category_qh.commit_output_query.query_string),
    ]
    assert query_list == expected_query_list


def test_handle_query_result_norm(span_category_qh):
    query_result = mock.Mock(spec=QueryResult)
    handler_result = span_category_qh.handle_query_result(query_result)
    assert isinstance(handler_result, Finish)
    assert isinstance(handler_result.result, AlgoNormalisedIOData)
    result_io = cast(AlgoNormalisedIOData, handler_result.result)
    assert result_io.datasource.name == _OUTPUT_TABLE
    assert result_io.dataview.name == f"{_OUTPUT_TABLE}_VIEW"
    assert len(result_io.lookup_tables) == 2
