from typing import Any

import pandas as pd
from pandas.api.types import is_string_dtype
from pyexasol import ExaConnection

from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.utils.sql_utils import sql_text_value


class CommonOutputs:
    text_doc_id = 0
    text_char_begin = 1
    text_char_end = 2
    error_message = -2


def run_udf_with_text_input(
    pyexasol_connection: ExaConnection,
    udf_factory: UDFFactory,
    udf_runner: UDFRunner,
    test_cases,
) -> list[tuple[Any]]:
    """
    Creates and runs a UDF using the provided UDFFactory and UDFRunner. The UDF must
    have a standard input consisting of the text and its span.
    The UDF is fed with the provided data that includes several test cases. Each
    test case should again have a text and its span.

    Returns the UDF output.
    """

    # Create a UDF for the task.
    create_udf_query = udf_factory.create_udf_query(udf_runner)
    pyexasol_connection.execute(create_udf_query)

    # Prepare the UDF input.
    values = ", ".join(
        f"({text_case.text_doc_id}, "
        f"{text_case.text_char_begin}, "
        f"{text_case.text_char_begin + len(text_case.text)}, "
        f"{sql_text_value(text_case.text)})"
        for text_case in test_cases
    )

    # Run the UDF and return the results.
    run_udf_query = (
        f"SELECT {udf_factory.udf_name.fully_qualified}"
        "(t.text_doc_id, t.text_char_begin, t.text_char_end, t.text) "
        f"FROM (VALUES {values} "
        "AS t(text_doc_id, text_char_begin, text_char_end, text))"
    )
    return pyexasol_connection.execute(run_udf_query).fetchall()


def common_checks(keyword_test_cases, output, no_error: bool = True):
    """
    Performs utils checks:
    Validates that the text span is copied from the input to the output.
    Validates that the error message column is empty.
    """
    expected_spans = {
        (
            text_case.text_doc_id,
            text_case.text_char_begin,
            text_case.text_char_begin + len(text_case.text),
        )
        for text_case in keyword_test_cases
    }
    output_spans = {
        (
            row[CommonOutputs.text_doc_id],
            row[CommonOutputs.text_char_begin],
            row[CommonOutputs.text_char_end],
        )
        for row in output
    }
    assert output_spans == expected_spans
    for row in output:
        if no_error:
            assert row[CommonOutputs.error_message] is None, row[
                CommonOutputs.error_message
            ]
        else:
            assert row[CommonOutputs.error_message] is not None


def delete_table(data: AlgoIOData, pyexasol_connection: ExaConnection) -> None:
    query = f"DROP TABLE IF EXISTS {data.datasource.fully_qualified}"
    pyexasol_connection.execute(query)


def create_table(data: AlgoIOData, pyexasol_connection: ExaConnection) -> None:

    columns = ", ".join(
        f"{col.quoted_name} {col.sql_type}" for col in data.columns.order
    )
    query = f"CREATE OR REPLACE TABLE {data.datasource.fully_qualified}({columns})"
    pyexasol_connection.execute(query)


def populate_table(
    data: AlgoIOData, content: pd.DataFrame, pyexasol_connection: ExaConnection
) -> None:

    def cell_value(cell, dtype) -> str:
        if cell is None:
            return "NULL"
        if is_string_dtype(dtype):
            return sql_text_value(cell)
        return str(cell)

    def row_values(row: pd.Series) -> str:
        return ", ".join(
            cell_value(cell, dtype) for cell, dtype in zip(row, content.dtypes)
        )

    values = ", ".join(f"({row_values(row)})" for _, row in content.iterrows())
    query = f"INSERT INTO {data.datasource.fully_qualified} VALUES {values}"
    pyexasol_connection.execute(query)
