from inspect import cleandoc

import pandas as pd
from exasol.analytics.sql_executor.testing.mock_result_set import MockResultSet
from exasol.analytics.sql_executor.testing.mock_sql_executor import (
    ExpectedQuery,
    MockSQLExecutor,
)

from exasol.ai.text.impl.db_schema import SpanColumns


def get_input_df(test_cases, text_column: str, text_span: SpanColumns) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                text_case.text_doc_id,
                text_case.text_char_begin,
                text_case.text_char_begin + len(text_case.text),
                text_case.text,
            )
            for text_case in test_cases
        ],
        columns=[*text_span.column_names, text_column],
    )


def common_checks(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    text_column: str,
    error_message_column: str,
    text_span: SpanColumns,
) -> None:
    """
    Checks that all data in the `input_df` is copied to the `output_df`.
    Checks that the error message column is empty.
    """
    merged_df = pd.merge(
        input_df,
        output_df,
        on=text_span.column_names,
        how="left",
        suffixes=("_A", "_B"),
    )
    assert not merged_df[f"{text_column}_B"].isnull().any()
    assert output_df[error_message_column].isnull().all()


def expect_query(template: str, result_set=MockResultSet()):
    """
    Query template may contain placeholders {schema} and {prefix} which
    will be substituted by the values passed as arguments.
    """
    return [template, result_set]


def create_sql_executor(schema: str, prefix: str, *args):
    """
    Each element of args is expected to be a list that will be passed to
    the constructor of :class:`ExpectedQuery`.

    Each list may have 1 or 2 elements. First element is a string containing an
    SQL query.  The query may contain placeholders {schema} and {prefix} which
    will be substituted by the values passed as arguments.

    The second element is optional and may be a :class:`MockResultSet`.

    Examples:

    create_sql_executor("S", "P", ["select 1 from {schema}.{prefix}_table"])
    create_sql_executor("S", "P", ["select 1 from table", result_set])
    create_sql_executor("S", "P", [query_1, result_set_1], [q2, rs2])
    """
    return MockSQLExecutor(
        [
            ExpectedQuery(
                cleandoc(template.format(schema=schema, prefix=prefix)),
                result_set or MockResultSet(),
            )
            for template, result_set in args
        ]
    )
