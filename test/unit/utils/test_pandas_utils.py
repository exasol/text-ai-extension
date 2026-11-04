import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.utils.pandas_utils import (
    adjust_result_span,
    merge_result_with_input,
)


def test_merge_result_with_input():
    input_df = pd.DataFrame({"A": [1, 3, 5], "B": ["aaa", "bbb", "ccc"]})
    result_df = pd.DataFrame(
        {
            "B": ["abc", "def", "ghi", "jkl", "mno"],
            "C": [123, 234, 345, 456, 567],
            "X": [0, 1, 1, 1, 2],
        }
    )
    output_df = merge_result_with_input(
        input_df=input_df, result_df=result_df, merge_on_column="X"
    )
    expected_df = pd.DataFrame(
        {
            "A": [1, 3, 3, 3, 5],
            "B": ["abc", "def", "ghi", "jkl", "mno"],
            "C": [123, 234, 345, 456, 567],
        }
    )
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)


def test_adjust_result_span():
    text_span = SpanColumns.create("text")
    result_span = SpanColumns.create("phrase")
    _, result_begin_column, result_end_column = result_span.column_names
    output_df = pd.DataFrame(
        [(123, 100, 200, 23, 67), (234, 300, 400, 12, 24)],
        columns=[*text_span.column_names, result_begin_column, result_end_column],
    )
    output_df = adjust_result_span(
        output_df=output_df, text_span=text_span, result_span=result_span
    )
    expected_df = pd.DataFrame(
        [(123, 100, 200, 123, 123, 167), (234, 300, 400, 234, 312, 324)],
        columns=[*text_span.column_names, *result_span.column_names],
    )
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)


def test_adjust_result_span_error():
    text_span = SpanColumns.create("text")
    result_span = SpanColumns.create("phrase")
    _, result_begin_column, result_end_column = result_span.column_names
    output_df = pd.DataFrame(
        [(123, 100, 200, 23, 67), (234, 300, 400, 12, 124)],
        columns=[*text_span.column_names, result_begin_column, result_end_column],
    )
    with pytest.raises(Exception):
        adjust_result_span(
            output_df=output_df, text_span=text_span, result_span=result_span
        )
