import re
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
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_envelope import TextEnvelope
from exasol.ai.text.impl.scripts import TextFunc
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.co_occurrence_query_handler import (
    CoOccurrenceQueryHandler,
    DataWithAlias,
    ExprColumn,
    build_expr_columns,
    build_input_pair_columns,
    create_func_in_schema_queries,
    get_prefixed_name,
    get_span_name_counter,
    get_unique_name,
    span_to_str,
)
from exasol.ai.text.impl.utils.sql_utils import (
    boolean_column_spec,
    integer_column_spec,
    varchar_column_spec,
)
from test.unit.utils.test_stage_graph_utils import collapse_spaces
from test.utils.word_span_columns import WordSpanColumns

SCHEMA_NAME = "MY_SCHEMA"


@pytest.fixture
def text_span() -> SpanColumns:
    return SpanColumns.create("TEXT")


@pytest.fixture
def error_message_spec() -> ColumnSpec:
    return varchar_column_spec("ERROR_MESSAGE")


@pytest.fixture
def setup_spec() -> ColumnSpec:
    return varchar_column_spec("SETUP")


@pytest.fixture
def input1(text_span, error_message_spec, setup_spec) -> DataWithAlias:
    return DataWithAlias(
        datasource=TableLikeNameImpl("INP_TABLE1", schema=SchemaName(SCHEMA_NAME)),
        columns=TextEnvelope(
            name="INP1",
            original_type=TextOutput,
            text_span=text_span,
            envelope=[
                varchar_column_spec("COL11"),
                varchar_column_spec("COL12"),
                SpanColumns.create("SPAN11"),
                SpanColumns.create("SPAN12"),
                WordSpanColumns.create("WORD_SPAN11"),
                WordSpanColumns.create("WORD_SPAN12"),
            ],
            error_message=error_message_spec,
            setup=setup_spec,
        ),
        alias="T1",
    )


@pytest.fixture
def input2(text_span, error_message_spec, setup_spec) -> DataWithAlias:
    return DataWithAlias(
        datasource=TableLikeNameImpl("INP_TABLE2", schema=SchemaName(SCHEMA_NAME)),
        columns=TextEnvelope(
            name="INP2",
            original_type=TextOutput,
            text_span=text_span,
            envelope=[
                varchar_column_spec("COL21"),
                SpanColumns.create("SPAN21"),
                SpanColumns.create("SPAN22"),
                SpanColumns.create("SPAN23"),
                WordSpanColumns.create("WORD_SPAN22"),
            ],
            error_message=error_message_spec,
            setup=setup_spec,
        ),
        alias="T2",
    )


@pytest.fixture
def input3(text_span, error_message_spec, setup_spec) -> DataWithAlias:
    return DataWithAlias(
        datasource=TableLikeNameImpl("INP_TABLE3", schema=SchemaName(SCHEMA_NAME)),
        columns=TextEnvelope(
            name="INP3",
            original_type=TextOutput,
            text_span=text_span,
            envelope=[
                varchar_column_spec("COL31"),
                varchar_column_spec("COL32"),
                varchar_column_spec("COL33"),
                SpanColumns.create("SPAN31"),
                WordSpanColumns.create("WORD_SPAN31"),
            ],
            error_message=error_message_spec,
            setup=setup_spec,
        ),
        alias="T3",
    )


@pytest.fixture
def qh_context() -> ScopeQueryHandlerContext:
    return mock.Mock(spec=ScopeQueryHandlerContext)


@pytest.fixture
def coo_query_handler(input1, input2, input3, qh_context) -> CoOccurrenceQueryHandler:
    log_span = mock.Mock(spec=LogSpan)
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[
                    AlgoIOData(datasource=inp.datasource, columns=inp.columns)
                    for inp in [input1, input2, input3]
                ]
            )
        ],
        result_bucketfs_location=None,
    )  # type: ignore
    return CoOccurrenceQueryHandler(
        stage_input=stage_input,
        query_handler_context=qh_context,
        output_view=TableLikeNameImpl("CO_OCCURRENCE"),
        log_span=log_span,
    )


def _span_to_str(is_char: bool, alias: str, span_name: str) -> str:
    if is_char:
        return span_to_str(alias, SpanColumns.create(span_name))
    return span_to_str(alias, WordSpanColumns.create(span_name))


def _distance_x_column(
    is_char: bool, alias1: str, span_name1: str, alias2: str, span_name2: str
) -> ExprColumn:
    span_fanc = (
        TextFunc.char_span_distance if is_char else TextFunc.segment_span_distance
    )
    return ExprColumn(
        column=integer_column_spec(f"DISTANCE_{span_name1}_TO_{span_name2}"),
        expr=(
            f'"{SCHEMA_NAME}".{span_fanc}('
            f"{_span_to_str(is_char, alias1, span_name1)},"
            f"{_span_to_str(is_char, alias2, span_name2)})"
        ),
    )


def _before_x_column(
    is_char: bool, alias1: str, span_name1: str, alias2: str, span_name2: str
) -> ExprColumn:
    span_fanc = TextFunc.char_span_before if is_char else TextFunc.segment_span_before
    return ExprColumn(
        column=boolean_column_spec(f"{span_name1}_BEFORE_{span_name2}"),
        expr=(
            f'"{SCHEMA_NAME}".{span_fanc}('
            f"{_span_to_str(is_char, alias1, span_name1)},"
            f"{_span_to_str(is_char, alias2, span_name2)})"
        ),
    )


def _contains_x_column(
    is_char: bool, alias1: str, span_name1: str, alias2: str, span_name2: str
) -> ExprColumn:
    span_fanc = (
        TextFunc.char_span_contains if is_char else TextFunc.segment_span_contains
    )
    return ExprColumn(
        column=boolean_column_spec(f"{span_name1}_CONTAINS_{span_name2}"),
        expr=(
            f'"{SCHEMA_NAME}".{span_fanc}('
            f"{_span_to_str(is_char, alias1, span_name1)},"
            f"{_span_to_str(is_char, alias2, span_name2)})"
        ),
    )


def _char_span_select_columns(
    alias: str, span_name: str, span_name_out: str = None
) -> list[str]:
    span_name_out = span_name_out or span_name
    return [
        f"{alias}.{col.quoted_name} AS {col_out.quoted_name}"
        for col, col_out in zip(
            SpanColumns.create(span_name).order, SpanColumns.create(span_name_out).order
        )
    ]


def _segment_span_select_columns(alias: str, span_name: str) -> list[str]:
    return [
        f"{alias}.{col.quoted_name} AS {col.quoted_name}"
        for col in WordSpanColumns.create(span_name).order
    ]


def _validate_column_expressions(column_texts: list[str]) -> None:
    """
    The number of expression columns created for each type of span
    and function.
    """
    char_span_funcs = [
        TextFunc.char_span_distance,
        TextFunc.char_span_before,
        TextFunc.char_span_contains,
    ]
    for span_func in char_span_funcs:
        char_func_result = list(filter(lambda text: span_func in text, column_texts))
        assert len(char_func_result) == 22
    segment_span_funcs = [
        TextFunc.segment_span_distance,
        TextFunc.segment_span_before,
        TextFunc.segment_span_contains,
    ]
    for span_func in segment_span_funcs:
        segment_func_result = list(filter(lambda text: span_func in text, column_texts))
        assert len(segment_func_result) == 10


@dataclass
class ExprColumnArgs:
    is_char: bool
    alias1: str
    span_name1: str
    alias2: str
    span_name2: str


def test_get_span_name_counter_no_prefix(input1, input2, input3):
    """
    Validates that the span name is not prefixed with the input name if all span names
    are unique.
    """
    name_counter = get_span_name_counter([input1, input2, input3])
    assert get_unique_name(input1.columns.name, "SPAN11", name_counter) == "SPAN11"


def test_get_span_name_counter_prefix(input1):
    """
    Validates that the span name is prefixed with the input name in case it is present
    in multiple inputs.
    """
    input_columns = cast(TextEnvelope, input1.columns)
    input_copy = DataWithAlias(
        datasource=input1.datasource,
        columns=TextEnvelope(
            name="INP_COPY",
            original_type=TextOutput,
            text_span=input_columns.text_span,
            envelope=input_columns.envelope,
            error_message=input_columns.error_message,
            setup=input_columns.setup,
        ),
        alias="TX",
    )
    name_counter = get_span_name_counter([input1, input_copy])
    assert get_unique_name(
        input1.columns.name, "SPAN11", name_counter
    ) == get_prefixed_name(input1.columns.name, "SPAN11")
    assert get_unique_name("INP_COPY", "SPAN11", name_counter) == get_prefixed_name(
        "INP_COPY", "SPAN11"
    )


def test_get_span_name_counter_error(input1):
    """
    Validates that the error is raised if multiple inputs have the same name and spans
    with identical names.
    """
    name_counter = get_span_name_counter([input1, input1])
    with pytest.raises(ValueError, match=".*unique name.*"):
        get_unique_name(input1.columns.name, "SPAN11", name_counter)


def test_build_input_pair_columns(input1, input2):
    combos = [
        ExprColumnArgs(True, "T1", "SPAN11", "T2", "SPAN21"),
        ExprColumnArgs(True, "T1", "SPAN11", "T2", "SPAN22"),
        ExprColumnArgs(True, "T1", "SPAN11", "T2", "SPAN23"),
        ExprColumnArgs(True, "T1", "SPAN12", "T2", "SPAN21"),
        ExprColumnArgs(True, "T1", "SPAN12", "T2", "SPAN22"),
        ExprColumnArgs(True, "T1", "SPAN12", "T2", "SPAN23"),
        ExprColumnArgs(False, "T1", "WORD_SPAN11", "T2", "WORD_SPAN22"),
        ExprColumnArgs(False, "T1", "WORD_SPAN12", "T2", "WORD_SPAN22"),
    ]
    funcs = [_distance_x_column, _before_x_column, _contains_x_column]
    name_counter = get_span_name_counter([input1, input2])
    expected_result = set(
        [
            func(
                comb.is_char, comb.alias1, comb.span_name1, comb.alias2, comb.span_name2
            )
            for comb in combos
            for func in funcs
        ]
        + [
            func(
                comb.is_char, comb.alias2, comb.span_name2, comb.alias1, comb.span_name1
            )
            for comb in combos
            for func in funcs
        ]
    )
    result = set(build_input_pair_columns(input1, input2, name_counter))
    assert result == expected_result


def test_build_expr_columns(input1, input2, input3):
    result = build_expr_columns([input1, input2, input3])
    _validate_column_expressions([col.expr for col in result])


def test_create_func_in_schema_queries():
    """
    The correctness of the actual queries is tested elsewhere. Here we will just make
    sure that the right functions are created in the right schema.
    """
    log_span = mock.Mock(spec=LogSpan)
    schema = SchemaName(SCHEMA_NAME)
    queries = [
        query.query_string for query in create_func_in_schema_queries(schema, log_span)
    ]
    assert all(schema.quoted_name in query for query in queries)
    joined_queries = ",".join(queries)
    assert all(
        func in joined_queries
        for func in [
            TextFunc.segment_span_distance,
            TextFunc.segment_span_distance,
            TextFunc.char_span_before,
            TextFunc.segment_span_before,
            TextFunc.char_span_contains,
            TextFunc.segment_span_contains,
        ]
    )


def test_insufficient_input(input1, qh_context):
    """
    Validates that an error is raised if the Query Handler gets less than two inputs.
    """
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[
                    AlgoIOData(datasource=input1.datasource, columns=input1.columns)
                ]
            )
        ],
        result_bucketfs_location=None,
    )  # type: ignore
    log_span = mock.Mock(LogSpan)
    with pytest.raises(ValueError, match=r".+requires two or more inputs"):
        CoOccurrenceQueryHandler(
            stage_input=stage_input,
            query_handler_context=qh_context,
            output_view=TableLikeNameImpl("CO_OCCURRENCE"),
            log_span=log_span,
        )


def test_text_span_columns(coo_query_handler, text_span):
    doc_id = text_span.doc_id.quoted_name
    expected_columns = [f"T1.{doc_id} AS {doc_id}"]
    assert coo_query_handler.text_span_columns == expected_columns


def test_algo_columns(coo_query_handler, text_span):
    expected_columns = [
        *_char_span_select_columns("T1", text_span.name, f"INP1_{text_span.name}"),
        'T1."COL11" AS "COL11"',
        'T1."COL12" AS "COL12"',
        *_char_span_select_columns("T1", "SPAN11"),
        *_char_span_select_columns("T1", "SPAN12"),
        *_segment_span_select_columns("T1", "WORD_SPAN11"),
        *_segment_span_select_columns("T1", "WORD_SPAN12"),
        'T1."ERROR_MESSAGE" AS "INP1_ERROR_MESSAGE"',
        'T1."SETUP" AS "INP1_SETUP"',
        *_char_span_select_columns("T2", text_span.name, f"INP2_{text_span.name}"),
        'T2."COL21" AS "COL21"',
        *_char_span_select_columns("T2", "SPAN21"),
        *_char_span_select_columns("T2", "SPAN22"),
        *_char_span_select_columns("T2", "SPAN23"),
        *_segment_span_select_columns("T2", "WORD_SPAN22"),
        'T2."ERROR_MESSAGE" AS "INP2_ERROR_MESSAGE"',
        'T2."SETUP" AS "INP2_SETUP"',
        *_char_span_select_columns("T3", text_span.name, f"INP3_{text_span.name}"),
        'T3."COL31" AS "COL31"',
        'T3."COL32" AS "COL32"',
        'T3."COL33" AS "COL33"',
        *_char_span_select_columns("T3", "SPAN31"),
        *_segment_span_select_columns("T3", "WORD_SPAN31"),
        'T3."ERROR_MESSAGE" AS "INP3_ERROR_MESSAGE"',
        'T3."SETUP" AS "INP3_SETUP"',
    ]
    assert coo_query_handler.algo_columns == expected_columns


def test_expression_columns(coo_query_handler):
    expr_columns = coo_query_handler.expression_columns
    _validate_column_expressions(expr_columns)


def test_from_clause(coo_query_handler, text_span):
    txt_doc_id, txt_begin, txt_end = text_span.quoted_column_names
    expected_from_clause = collapse_spaces(f"""
    FROM
        "{SCHEMA_NAME}"."INP_TABLE1" T1
        JOIN "{SCHEMA_NAME}"."INP_TABLE2" T2 ON
            T1.{txt_doc_id}=T2.{txt_doc_id}
        JOIN "{SCHEMA_NAME}"."INP_TABLE3" T3 ON
            T2.{txt_doc_id}=T3.{txt_doc_id}
    """)
    assert collapse_spaces(coo_query_handler.from_clause) == expected_from_clause


def test_create_view_query(coo_query_handler):
    pattern = r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW.+SELECT.+FROM.+"
    assert (
        re.fullmatch(
            pattern, coo_query_handler.create_view_query.query_string, re.MULTILINE
        )
        is not None
    )


def test_start(coo_query_handler):
    result = coo_query_handler.start()
    query_list = [collapse_spaces(query.query_string) for query in result.query_list]
    assert len(query_list) == 7
    assert query_list[:-1] == [
        collapse_spaces(query.query_string)
        for query in create_func_in_schema_queries(
            SchemaName(SCHEMA_NAME), coo_query_handler.log_span
        )
    ]


def test_handle_query_result(coo_query_handler):
    query_result = mock.Mock(spec=QueryResult)
    handler_result = coo_query_handler.handle_query_result(query_result)
    assert isinstance(handler_result, Finish)
    assert isinstance(handler_result.result, AlgoIOData)
