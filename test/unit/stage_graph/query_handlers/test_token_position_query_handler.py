from dataclasses import dataclass
from typing import cast
from unittest import mock

import pytest
from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.query_handler.result import Finish
from exasol.analytics.schema import (
    SchemaName,
    TableLikeNameImpl,
)

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import TokenizerOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.scripts import TextFunc
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import (
    AlgoIOData,
    AlgoNormalisedIOData,
)
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.token_position_query_handler import (
    HASH_LENGTH,
    TOKEN_ID_COLUMN,
    TOKEN_SEQ_COLUMN,
    TOKEN_SPAN_SUFFIX,
    TokenPositionQueryHandler,
    TokenSpanColumns,
    get_create_function_query,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec
from test.unit.utils.test_stage_graph_utils import collapse_spaces

TOKEN_VIEW_NAME = "TOKEN_VIEW"


@dataclass(frozen=True)
class DummyOutput(TextOutput):
    column_a1: ColumnSpec = varchar_column_spec("A1")
    column_a2: ColumnSpec = varchar_column_spec("A2")
    span_a1: SpanColumns = SpanColumns.create("A1_SPAN")
    span_a2: SpanColumns = SpanColumns.create("A2_SPAN")


@dataclass(frozen=True)
class NoSpanOutput(TextOutput):
    column_d: ColumnSpec = varchar_column_spec("D")


@pytest.fixture
def schema_name_a() -> SchemaName:
    return SchemaName("MY_SCHEMA_A")


@pytest.fixture
def schema_name_t() -> SchemaName:
    return SchemaName("MY_SCHEMA_T")


@pytest.fixture
def dummy_output(schema_name_a) -> AlgoNormalisedIOData:
    return AlgoNormalisedIOData(
        datasource=TableLikeNameImpl("OUTPUT", schema=schema_name_a),
        dataview=TableLikeNameImpl("OUTPUT_VIEW", schema=schema_name_a),
        lookup_tables=None,
        columns=DummyOutput(),
    )


@pytest.fixture
def no_span_output(schema_name_a) -> AlgoNormalisedIOData:
    return AlgoNormalisedIOData(
        datasource=TableLikeNameImpl("OUTPUT", schema=schema_name_a),
        dataview=None,
        lookup_tables=None,
        columns=NoSpanOutput(),
    )


@pytest.fixture
def tokenizer_output(schema_name_t) -> AlgoNormalisedIOData:
    return AlgoNormalisedIOData(
        datasource=TableLikeNameImpl("TOKENIZER", schema=schema_name_t),
        dataview=None,
        lookup_tables=None,
        columns=TokenizerOutput(),
    )


@pytest.fixture
def qh_context() -> ScopeQueryHandlerContext:
    return mock.Mock(spec=ScopeQueryHandlerContext)


@pytest.fixture
def tp_query_handler(
    schema_name_a, dummy_output, tokenizer_output, qh_context
) -> TokenPositionQueryHandler:
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(input_output_list=[dummy_output, tokenizer_output])
        ],
        result_bucketfs_location=None,
    )  # type: ignore
    log_span = mock.Mock(LogSpan)
    return TokenPositionQueryHandler(
        stage_input=stage_input,
        query_handler_context=qh_context,
        output_view=TableLikeNameImpl(TOKEN_VIEW_NAME, schema=schema_name_a),
        log_span=log_span,
    )


@pytest.fixture
def expected_create_view_query(dummy_output, tokenizer_output) -> str:
    hash_format = "x" * HASH_LENGTH
    algo_outputs = cast(DummyOutput, dummy_output.columns)
    tok_outputs = cast(TokenizerOutput, tokenizer_output.columns)
    # text span in the algorithm output
    txt_doc_id, txt_begin, txt_end = algo_outputs.text_span.quoted_column_names
    # character spans in the algorithm output
    algo_doc_id1, algo_begin1, algo_end1 = algo_outputs.span_a1.quoted_column_names
    algo_doc_id2, algo_begin2, algo_end2 = algo_outputs.span_a2.quoted_column_names
    # character span in the tokenizer output
    tok_doc_id, tok_begin, tok_end = tok_outputs.token_span.quoted_column_names
    tok_name1 = f"{algo_outputs.span_a1.name}_{TOKEN_SPAN_SUFFIX}"
    tok_name2 = f"{algo_outputs.span_a2.name}_{TOKEN_SPAN_SUFFIX}"
    # token spans created for the algorithm outputs
    tok_algo_doc_id1, tok_algo_ver1, tok_algo_begin1, tok_algo_end1 = (
        TokenSpanColumns.create(tok_name1).quoted_column_names
    )
    tok_algo_doc_id2, tok_algo_ver2, tok_algo_begin2, tok_algo_end2 = (
        TokenSpanColumns.create(tok_name2).quoted_column_names
    )
    err_msg = algo_outputs.error_message.quoted_name
    setup = algo_outputs.setup.quoted_name

    return collapse_spaces(f"""
        CREATE OR REPLACE VIEW "MY_SCHEMA_A"."{TOKEN_VIEW_NAME}" AS
        WITH T0 AS (
            SELECT
                {txt_doc_id} AS {tok_doc_id},
                {tok_begin},
                {tok_end},
                TO_NUMBER(RIGHT(HASH_TIGER("SETUP"), {HASH_LENGTH}), '{hash_format}') AS "{TOKEN_SEQ_COLUMN}",
                (ROW_NUMBER() OVER (PARTITION BY {txt_doc_id} ORDER BY {tok_begin})) - 1 AS "{TOKEN_ID_COLUMN}"
            FROM "MY_SCHEMA_T"."TOKENIZER"
        )
        SELECT
            T1.{txt_doc_id} AS {txt_doc_id},
            T1.{txt_begin} AS {txt_begin},
            T1.{txt_end} AS {txt_end},
            T1."A1" AS "A1",
            T1."A2" AS "A2",
            T1.{algo_doc_id1} AS {algo_doc_id1},
            T1.{algo_begin1} AS {algo_begin1},
            T1.{algo_end1} AS {algo_end1},
            T1.{algo_doc_id2} AS {algo_doc_id2},
            T1.{algo_begin2} AS {algo_begin2},
            T1.{algo_end2} AS {algo_end2},
            T2.{tok_algo_doc_id1},
            T2.{tok_algo_ver1},
            T2.{tok_algo_begin1},
            T2.{tok_algo_end1},
            T3.{tok_algo_doc_id2},
            T3.{tok_algo_ver2},
            T3.{tok_algo_begin2},
            T3.{tok_algo_end2},
            T1.{err_msg} AS {err_msg},
            T1.{setup} AS {setup}
        FROM
            "MY_SCHEMA_A"."OUTPUT_VIEW" T1
            JOIN (
                SELECT
                    T1.{txt_doc_id} AS {txt_doc_id},
                    FIRST_VALUE(T1.{txt_begin}) AS {txt_begin},
                    FIRST_VALUE(T1.{txt_end}) AS {txt_end},
                    T1.{algo_begin1} AS {algo_begin1},
                    T1.{algo_end1} AS {algo_end1},
                    T1.{txt_doc_id} AS {tok_algo_doc_id1},
                    T0."{TOKEN_SEQ_COLUMN}" AS {tok_algo_ver1},
                    MIN(T0."{TOKEN_ID_COLUMN}") AS {tok_algo_begin1},
                    MAX(T0."{TOKEN_ID_COLUMN}") + 1 AS {tok_algo_end1}
                FROM
                    "MY_SCHEMA_A"."OUTPUT_VIEW" T1
                    LEFT OUTER JOIN T0
                    ON T1.{txt_doc_id} = T0.{tok_doc_id} AND "MY_SCHEMA_A".{TextFunc.char_span_contains}(
                        T1.{txt_doc_id}, T1.{algo_begin1}, T1.{algo_end1},
                        T0.{tok_doc_id}, T0.{tok_begin}, T0.{tok_end})
                GROUP BY T1.{txt_doc_id}, T1.{algo_begin1}, T1.{algo_end1}, T0."{TOKEN_SEQ_COLUMN}"
            ) T2
            ON
                T1.{txt_doc_id}=T2.{txt_doc_id} AND
                T1.{txt_begin}=T2.{txt_begin} AND
                T1.{txt_end}=T2.{txt_end} AND
                T1.{algo_begin1}=T2.{algo_begin1} AND
                T1.{algo_end1}=T2.{algo_end1}
            JOIN (
                SELECT
                    T1.{txt_doc_id} AS {txt_doc_id},
                    FIRST_VALUE(T1.{txt_begin}) AS {txt_begin},
                    FIRST_VALUE(T1.{txt_end}) AS {txt_end},
                    T1.{algo_begin2} AS {algo_begin2},
                    T1.{algo_end2} AS {algo_end2},
                    T1.{txt_doc_id} AS {tok_algo_doc_id2},
                    T0."{TOKEN_SEQ_COLUMN}" AS {tok_algo_ver2},
                    MIN(T0."{TOKEN_ID_COLUMN}") AS {tok_algo_begin2},
                    MAX(T0."{TOKEN_ID_COLUMN}") + 1 AS {tok_algo_end2}
                FROM
                    "MY_SCHEMA_A"."OUTPUT_VIEW" T1
                    LEFT OUTER JOIN T0
                    ON T1.{txt_doc_id} = T0.{tok_doc_id} AND "MY_SCHEMA_A".{TextFunc.char_span_contains}(
                        T1.{txt_doc_id}, T1.{algo_begin2}, T1.{algo_end2},
                        T0.{tok_doc_id}, T0.{tok_begin}, T0.{tok_end})
                GROUP BY T1.{txt_doc_id}, T1.{algo_begin2}, T1.{algo_end2}, T0."{TOKEN_SEQ_COLUMN}"
            ) T3
            ON
                T1.{txt_doc_id}=T3.{txt_doc_id} AND
                T1.{txt_begin}=T3.{txt_begin} AND
                T1.{txt_end}=T3.{txt_end} AND
                T1.{algo_begin2}=T3.{algo_begin2} AND
                T1.{algo_end2}=T3.{algo_end2}
    """)


def test_create_view_query(tp_query_handler, expected_create_view_query):
    create_view_query = collapse_spaces(tp_query_handler.create_view_query.query_string)
    assert create_view_query == expected_create_view_query


def test_get_create_function_query(tp_query_handler, schema_name_a):
    """
    The correctness of the actual query is tested elsewhere. Here we will just make
    sure the function is created in the right schema.
    """
    log_span = mock.Mock(LogSpan)
    query = get_create_function_query(
        tp_query_handler.algo_output, log_span
    ).query_string
    assert schema_name_a.quoted_name in query


def test_no_output(schema_name_a, qh_context, no_span_output, tokenizer_output):
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(input_output_list=[no_span_output, tokenizer_output])
        ],
        result_bucketfs_location=None,
    )  # type: ignore
    log_span = mock.Mock(LogSpan)
    with pytest.raises(ValueError, match=".+output with one or more spans.+"):
        TokenPositionQueryHandler(
            stage_input=stage_input,
            query_handler_context=qh_context,
            output_view=TableLikeNameImpl(TOKEN_VIEW_NAME, schema=schema_name_a),
            log_span=log_span,
        )


def test_no_tokenizer_output(schema_name_a, qh_context, dummy_output):
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[ListInputOutput(input_output_list=[dummy_output])],
        result_bucketfs_location=None,
    )  # type: ignore
    log_span = mock.Mock(LogSpan)
    with pytest.raises(ValueError, match="Exactly one output from a Tokenizer.+"):
        TokenPositionQueryHandler(
            stage_input=stage_input,
            query_handler_context=qh_context,
            output_view=TableLikeNameImpl(TOKEN_VIEW_NAME, schema=schema_name_a),
            log_span=log_span,
        )


def test_multi_tokenizer_output(
    schema_name_a, qh_context, dummy_output, tokenizer_output
):
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[dummy_output, tokenizer_output, tokenizer_output]
            )
        ],
        result_bucketfs_location=None,
    )  # type: ignore
    log_span = mock.Mock(LogSpan)
    with pytest.raises(ValueError, match="Exactly one output from a Tokenizer.+"):
        TokenPositionQueryHandler(
            stage_input=stage_input,
            query_handler_context=qh_context,
            output_view=TableLikeNameImpl(TOKEN_VIEW_NAME, schema=schema_name_a),
            log_span=log_span,
        )


def test_start(tp_query_handler, expected_create_view_query):
    result = tp_query_handler.start()
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    expected_query_list = [
        collapse_spaces(
            get_create_function_query(
                tp_query_handler.algo_output, tp_query_handler.log_span
            ).query_string
        ),
        expected_create_view_query,
    ]
    assert query_list == expected_query_list


def test_handle_query_result(tp_query_handler):
    query_result = mock.Mock(spec=QueryResult)
    handler_result = tp_query_handler.handle_query_result(query_result)
    assert isinstance(handler_result, Finish)
    assert isinstance(handler_result.result, AlgoIOData)
