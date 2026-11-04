from dataclasses import dataclass
from typing import Any

import pytest

from exasol.ai.text.deployment.script_deployer import create_scripts
from exasol.ai.text.impl.scripts import TextFunc


@dataclass
class _FuncTest:
    func_name: str
    func_args: str
    expected_result: tuple[Any, ...]


@pytest.fixture
def func_tests() -> list[_FuncTest]:
    return [
        _FuncTest(TextFunc.char_span_valid, "1, 1, 10", (True,)),
        _FuncTest(TextFunc.char_span_before, "1, 1, 10, 1, 15, 20", (True,)),
        _FuncTest(TextFunc.char_span_contains, "1, 1, 20, 1, 2, 5", (True,)),
        _FuncTest(TextFunc.char_span_overlaps, "1, 1, 10, 1, 4, 15", (True,)),
        _FuncTest(TextFunc.char_span_equals, "1, 1, 10, 1, 1, 10", (True,)),
        _FuncTest(TextFunc.char_span_length, "1, 1, 10", (9,)),
        _FuncTest(TextFunc.char_span_distance, "1, 1, 10, 1, 11, 15", (1,)),
        _FuncTest(TextFunc.char_span_substring, "1, 0, 4, 'abcd', 1, 1, 3", ("bc",)),
        _FuncTest(TextFunc.combine_char_span, "1, 2, 5, 1, 3, 6", (1, 2, 6)),
        _FuncTest(TextFunc.segment_span_valid, "1, 1, 2, 6", (True,)),
        _FuncTest(TextFunc.segment_span_before, "1, 2, 1, 10, 1, 2, 15, 20", (True,)),
        _FuncTest(TextFunc.segment_span_contains, "1, 2, 1, 20, 1, 2, 2, 5", (True,)),
        _FuncTest(TextFunc.segment_span_overlaps, "1, 2, 1, 10, 1, 2, 4, 15", (True,)),
        _FuncTest(TextFunc.segment_span_equals, "1, 2, 1, 10, 1, 2, 1, 10", (True,)),
        _FuncTest(TextFunc.segment_span_length, "1, 2, 1, 10", (9,)),
        _FuncTest(TextFunc.segment_span_distance, "1, 2, 1, 10, 1, 2, 11, 15", (1,)),
        _FuncTest(
            TextFunc.combine_segment_span,
            "1, 2, 5, 2, 15, 18, 1, 3, 6, 2, 13, 17",
            (1, 2, 6, 2, 13, 18),
        ),
    ]


def test_create_scripts(setup_database, pyexasol_connection, func_tests):
    try:
        create_scripts(pyexasol_connection)
        results = [
            pyexasol_connection.execute(
                f"SELECT {tst.func_name}({tst.func_args})"
            ).fetchall()
            for tst in func_tests
        ]
        expected_results = [[tst.expected_result] for tst in func_tests]
        assert results == expected_results
    finally:
        for tst in func_tests:
            drop_query = f"DROP FUNCTION IF EXISTS {tst.func_name}"
            pyexasol_connection.execute(drop_query)
