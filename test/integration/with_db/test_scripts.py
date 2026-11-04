from collections.abc import (
    Callable,
    Generator,
)
from contextlib import contextmanager
from dataclasses import (
    dataclass,
    fields,
)
from functools import partial
from typing import (
    Any,
    cast,
)

import pytest
from exasol.analytics.schema import SchemaName
from pyexasol import ExaConnection

from exasol.ai.text.impl.scripts import (
    Scripts,
    TextFunc,
)


@dataclass
class _RelationTest:
    precedes: bool = False
    meets: bool = False
    overlaps: bool = False
    finished_by: bool = False
    contains: bool = False
    starts: bool = False
    started_by: bool = False
    equals: bool = False
    during: bool = False
    finishes: bool = False
    overlapped_by: bool = False
    met_by: bool = False
    preceded_by: bool = False


@dataclass
class _SpanPair:
    span1: tuple[int, int]
    span2: tuple[int, int]


_relation_examples = {
    "precedes": _SpanPair(span1=(2, 5), span2=(6, 8)),
    "meets": _SpanPair(span1=(2, 5), span2=(5, 8)),
    "overlaps": _SpanPair(span1=(2, 6), span2=(4, 8)),
    "finished_by": _SpanPair(span1=(2, 8), span2=(5, 8)),
    "contains": _SpanPair(span1=(2, 8), span2=(3, 6)),
    "starts": _SpanPair(span1=(4, 6), span2=(4, 8)),
    "equals": _SpanPair(span1=(2, 6), span2=(2, 6)),
    "started_by": _SpanPair(span1=(2, 8), span2=(2, 6)),
    "during": _SpanPair(span1=(3, 6), span2=(2, 8)),
    "finishes": _SpanPair(span1=(4, 6), span2=(2, 6)),
    "overlapped_by": _SpanPair(span1=(4, 8), span2=(2, 6)),
    "met_by": _SpanPair(span1=(5, 8), span2=(2, 5)),
    "preceded_by": _SpanPair(span1=(6, 9), span2=(2, 5)),
}


def _get_sql_query(sql_func_name, *args) -> str:
    """Constructs SQL query calling a function with provided arguments."""
    arg_list = ", ".join(
        f"'{arg}'" if isinstance(arg, str) else str(arg) for arg in args
    )
    return f"SELECT {sql_func_name}({arg_list})"


def _get_id_combos(id_length, odd_num: int, regular_num: int = 1) -> list[list[int]]:
    """
    Produces a collection of different IDs made of the same (regular) number in all
    positions but one, where there is a different (odd) number. The collection consists
    of IDs with the odd number in different positions.
    Example:
    _get_id_combos(3, -1, 1) -> [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    """
    return [
        [regular_num] * i + [odd_num] + [regular_num] * (id_length - i - 1)
        for i in range(id_length)
    ]


def _get_valid_span_id(id_length: int, num: int = 1) -> list[int]:
    return [num] * id_length


def _get_valid_span(id_length, num: int = 1) -> list[int]:
    return _get_valid_span_id(id_length, num) + [10, 20]


def _get_invalid_spans(id_length: int) -> list[list[int]]:
    """Returns all modalities of invalid spans"""
    spans_with_invalid_ids = [
        span_id + [10, 20] for span_id in _get_id_combos(id_length, -1, 1)
    ]
    valid_span_id = _get_valid_span_id(id_length, 1)
    return spans_with_invalid_ids + [
        valid_span_id + [-1, 10],
        valid_span_id + [10, 10],
        valid_span_id + [20, 10],
    ]


def _eval(pyexasol_connection: ExaConnection, sql_func_name: str, *args) -> Any:
    select_query = _get_sql_query(sql_func_name, *args)
    return pyexasol_connection.execute(select_query).fetchval()


def _eval_is_null(
    pyexasol_connection: ExaConnection, sql_func_name: str, *args
) -> bool:
    return _eval(pyexasol_connection, sql_func_name, *args) is None


def _eval_predicate(
    pyexasol_connection: ExaConnection,
    sql_func_name: str,
    span_id1: list[int],
    span_id2: list[int],
    example: _SpanPair,
) -> bool:
    result = _eval(
        pyexasol_connection,
        sql_func_name,
        *span_id1,
        *example.span1,
        *span_id2,
        *example.span2,
    )
    return cast(bool, result)


def _eval_binary_not_null(
    pyexasol_connection: ExaConnection,
    sql_func_name: str,
    span_id1: list[int],
    span_id2: list[int],
    example: _SpanPair,
) -> bool:
    """Evaluates that a function of two spans returns a not-NULL result."""
    return (
        _eval(
            pyexasol_connection,
            sql_func_name,
            *span_id1,
            *example.span1,
            *span_id2,
            *example.span2,
        )
        is not None
    )


def _eval_substring_not_null(
    pyexasol_connection: ExaConnection,
    sql_func_name: str,
    span_id1: int,
    span_id2: int,
    example: _SpanPair,
) -> bool:
    text = "x" * 100
    return (
        _eval(
            pyexasol_connection,
            sql_func_name,
            span_id1,
            *example.span1,
            text,
            span_id2,
            *example.span2,
        )
        is not None
    )


@pytest.fixture(params=[False, True])
def script_schema(request, db_schema_name) -> SchemaName | None:
    if request.param:
        return SchemaName(db_schema_name)
    return None


@contextmanager
def setup_func_validation(
    pyexasol_connection: ExaConnection, func_name: str, script_schema: SchemaName | None
) -> Generator[str, None, None]:
    script_deployer = Scripts(script_schema)
    create_query = getattr(script_deployer, func_name)
    sql_func_name = getattr(TextFunc, func_name)
    try:
        pyexasol_connection.execute(create_query)
        yield sql_func_name
    finally:
        drop_query = f"DROP FUNCTION IF EXISTS {sql_func_name}"
        pyexasol_connection.execute(drop_query)


def validate_span_intervals(
    expected_result: _RelationTest, eval_func: Callable[[_SpanPair], bool]
) -> None:
    """
    Tests a binary operation on two spans against the examples of all relations in the
    Allen's Interval Algebra and validates that the results meet the expectation, e.g.
    for all supported relations the result is True and for all others it's False.
    """
    result_dic: dict[str, bool] = {}
    for relation in fields(expected_result):
        example = _relation_examples[relation.name]
        result = eval_func(example)
        result_dic[relation.name] = result
    final_result = _RelationTest(**result_dic)
    assert final_result == expected_result


@pytest.mark.parametrize(
    ["func_name", "id_length", "expected_result"],
    [
        ("char_span_before", 1, _RelationTest(precedes=True, meets=True)),
        (
            "char_span_contains",
            1,
            _RelationTest(
                contains=True, finished_by=True, started_by=True, equals=True
            ),
        ),
        ("char_span_overlaps", 1, _RelationTest(overlaps=True, overlapped_by=True)),
        ("char_span_equals", 1, _RelationTest(equals=True)),
        ("segment_span_before", 2, _RelationTest(precedes=True, meets=True)),
        (
            "segment_span_contains",
            2,
            _RelationTest(
                contains=True, finished_by=True, started_by=True, equals=True
            ),
        ),
        ("segment_span_overlaps", 2, _RelationTest(overlaps=True, overlapped_by=True)),
        ("segment_span_equals", 2, _RelationTest(equals=True)),
    ],
)
def test_predicates(
    setup_database,
    pyexasol_connection,
    func_name,
    id_length,
    expected_result,
    script_schema,
):
    span_id = _get_valid_span_id(id_length)
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        validate_span_intervals(
            expected_result,
            partial(
                _eval_predicate, pyexasol_connection, sql_func_name, span_id, span_id
            ),
        )


@pytest.mark.parametrize(
    ["func_name", "id_length"],
    [
        ("char_span_before", 1),
        ("char_span_contains", 1),
        ("char_span_overlaps", 1),
        ("char_span_equals", 1),
        ("segment_span_before", 2),
        ("segment_span_contains", 2),
        ("segment_span_overlaps", 2),
        ("segment_span_equals", 2),
    ],
)
def test_predicate_id_mismatch(
    setup_database, pyexasol_connection, func_name, id_length, script_schema
):
    span_id1 = _get_valid_span_id(id_length, 1)
    expected_result = _RelationTest()
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        for span_id2 in _get_id_combos(id_length, 2, 1):
            validate_span_intervals(
                expected_result,
                partial(
                    _eval_predicate,
                    pyexasol_connection,
                    sql_func_name,
                    span_id1,
                    span_id2,
                ),
            )


@pytest.mark.parametrize(
    ["func_name", "id_length"],
    [
        ("char_span_valid", 1),
        ("segment_span_valid", 2),
    ],
)
def test_span_validation(
    setup_database, pyexasol_connection, func_name, id_length, script_schema
):
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        span = _get_valid_span(id_length)
        assert _eval(pyexasol_connection, sql_func_name, *span)
        for span in _get_invalid_spans(id_length):
            assert not _eval(pyexasol_connection, sql_func_name, *span)


@pytest.mark.parametrize(
    ["func_name", "func_args", "expected_result"],
    [
        ("char_span_length", [100, 4, 5], 1),
        ("char_span_length", [100, 4, 10], 6),
        ("char_span_distance", [100, 4, 5, 100, 7, 10], 2),
        ("char_span_distance", [100, 4, 5, 100, 5, 10], 0),
        ("char_span_substring", [100, 4, 15, "take a test", 100, 11, 15], "test"),
        ("char_span_substring", [100, 4, 15, "take a test", 100, 4, 8], "take"),
        ("char_span_substring", [100, 4, 15, "take a test", 100, 7, 12], "e a t"),
        ("char_span_substring", [100, 4, 15, "take", 100, 11, 15], None),
        ("segment_span_length", [100, 1, 4, 5], 1),
        ("segment_span_distance", [100, 1, 4, 5, 100, 1, 7, 10], 2),
        ("segment_span_distance", [100, 1, 4, 5, 100, 1, 5, 10], 0),
    ],
)
def test_scalar_function(
    setup_database,
    pyexasol_connection,
    func_name,
    func_args,
    expected_result,
    script_schema,
):
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        result = _eval(pyexasol_connection, sql_func_name, *func_args)
        assert result == expected_result


@pytest.mark.parametrize(
    ["func_name", "id_length"],
    [
        ("char_span_distance", 1),
        ("segment_span_distance", 2),
    ],
)
def test_span_distance_limits(
    setup_database, pyexasol_connection, func_name, id_length, script_schema
):
    """
    Validates that a `span_distance` function returns NULL if the two spans do not
    satisfy the defined conditions.
    """
    span_id = [1] * id_length
    expected_result = _RelationTest(precedes=True, meets=True)
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        validate_span_intervals(
            expected_result,
            partial(
                _eval_binary_not_null,
                pyexasol_connection,
                sql_func_name,
                span_id,
                span_id,
            ),
        )


@pytest.mark.parametrize(
    ["func_name", "id_length"],
    [
        ("char_span_distance", 1),
        ("segment_span_distance", 2),
    ],
)
def test_binary_id_mismatch(
    setup_database, pyexasol_connection, func_name, id_length, script_schema
):
    """
    Validates that a scalar function of two spans returns NULL if the IDs of the spans
    do not match.
    """
    span_id1 = _get_valid_span_id(id_length, 1)
    expected_result = _RelationTest()
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        for span_id2 in _get_id_combos(id_length, 2, 1):
            validate_span_intervals(
                expected_result,
                partial(
                    _eval_binary_not_null,
                    pyexasol_connection,
                    sql_func_name,
                    span_id1,
                    span_id2,
                ),
            )


def test_span_substring_limits(setup_database, pyexasol_connection, script_schema):
    """
    Validates that a `span_substring` function returns NULL if the two spans do not
    satisfy the defined conditions.
    """
    expected_result = _RelationTest(
        contains=True, finished_by=True, started_by=True, equals=True
    )
    with setup_func_validation(
        pyexasol_connection, "char_span_substring", script_schema
    ) as sql_func_name:
        validate_span_intervals(
            expected_result,
            partial(_eval_substring_not_null, pyexasol_connection, sql_func_name, 1, 1),
        )


def test_span_substring_id_mismatch(setup_database, pyexasol_connection, script_schema):
    """
    Validates that a `span_substring` function returns NULL if the IDs of the two spans
    do not match.
    """
    expected_result = _RelationTest()
    with setup_func_validation(
        pyexasol_connection, "char_span_substring", script_schema
    ) as sql_func_name:
        validate_span_intervals(
            expected_result,
            partial(_eval_substring_not_null, pyexasol_connection, sql_func_name, 1, 2),
        )


@pytest.mark.parametrize(
    ["func_name", "func_args", "expected_result"],
    [
        ("combine_char_span", "100, 4, 15, 100, 6, 20", (100, 4, 20)),
        ("combine_char_span", "100, 4, 15, 101, 6, 20", (None, None, None)),
        (
            "combine_segment_span",
            "100, 40, 150, 1, 4, 15, 100, 60, 200, 1, 6, 20",
            (100, 40, 200, 1, 4, 20),
        ),
        (
            "combine_segment_span",
            "100, 40, 150, 1, 4, 15, 101, 60, 200, 1, 6, 20",
            (None, None, None, None, None, None),
        ),
        (
            "combine_segment_span",
            "100, 40, 150, 1, 4, 15, 100, 60, 200, 2, 6, 20",
            (None, None, None, None, None, None),
        ),
    ],
)
def test_scalar_udf(
    setup_database,
    pyexasol_connection,
    func_name,
    func_args,
    expected_result,
    script_schema,
):
    with setup_func_validation(
        pyexasol_connection, func_name, script_schema
    ) as sql_func_name:
        select_query = f"SELECT {sql_func_name}({func_args})"
        result = pyexasol_connection.execute(select_query).fetchall()
        assert len(result) == 1
        assert result[0] == expected_result
