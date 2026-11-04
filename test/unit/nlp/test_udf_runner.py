from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column as MockColumn
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.testing import (
    BatchSizeLogger,
    DummyWordCounter,
)
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner

SETUP_COL_ID = -1
ERROR_COL_ID = -2


def create_meta_data(
    word_counter: DummyWordCounter | None = None,
    batch_logger: BatchSizeLogger | None = None,
):
    if word_counter is not None:
        text_column = word_counter.text_column
        error_message_column = word_counter.error_message_column
    elif batch_logger is not None:
        text_column = batch_logger.text_column
        error_message_column = batch_logger.error_message_column
    else:
        text_column = "text"
        error_message_column = "error_message"
    text_span = SpanColumns.create(text_column)
    span_columns = [MockColumn(col, int, "INTEGER") for col in text_span.column_names]
    word_counter_columns = (
        []
        if word_counter is None
        else [MockColumn(word_counter.word_count_column, int, "INTEGER")]
    )
    batch_logger_columns = (
        []
        if batch_logger is None
        else [MockColumn(batch_logger.batch_size_column, int, "INTEGER")]
    )
    return MockMetaData(
        None,
        input_type="SET",
        input_columns=span_columns + [MockColumn(text_column, str, "VARCHAR(2000000)")],
        output_type="EMITS",
        output_columns=span_columns
        + word_counter_columns
        + batch_logger_columns
        + [
            MockColumn(error_message_column, str, "VARCHAR(2000000)"),
            MockColumn("setup", str, "VARCHAR(2000000)"),
        ],
    )


@pytest.fixture
def input_data():
    return [
        (100, 100, 123, "It is greater than God."),
        (200, 100, 120, "Rich people need it."),
        (300, 100, 120, "Poor people have it."),
        (400, 100, 127, "If you eat it you will die."),
        (500, 100, 139, "Can you think what it is?"),
    ]


@pytest.fixture
def output_data():
    # This is the output for the word counter except the error message.
    return [
        (100, 100, 123, 5),
        (200, 100, 120, 4),
        (300, 100, 120, 4),
        (400, 100, 127, 7),
        (500, 100, 139, 6),
    ]


def test_run_word_counter(input_data, output_data, empty_exa):
    """
    Runs a single algorithm - the word counter.
    """

    runner = UDFRunner(algorithm={"algorithm_name": "dummy-word-counter"}, batch_size=3)

    meta_data = create_meta_data(word_counter=runner.algorithm)
    ctx = StandaloneMockContext(input_data, meta_data)
    runner.run(ctx, empty_exa)

    expected_setup = '{"DummyWordCounter": {"algorithm_name": "dummy-word-counter"}}'
    expected_output = [(*text_row, None, expected_setup) for text_row in output_data]
    assert ctx.output == expected_output


def test_run_explicit_setup(input_data, output_data, empty_exa):
    """
    Tests that it's possible to provide the "setup" externally.
    """

    setup = "my_setup"
    runner = UDFRunner(algorithm={"algorithm_name": "dummy-word-counter"}, batch_size=3)
    runner.setup = setup

    meta_data = create_meta_data(word_counter=runner.algorithm)
    ctx = StandaloneMockContext(input_data, meta_data)
    runner.run(ctx, empty_exa)

    expected_output = [(*text_row, None, setup) for text_row in output_data]
    assert ctx.output == expected_output


@pytest.mark.parametrize(
    ["batch_size", "batch_size_output"],
    [(1, [1, 1, 1, 1, 1]), (2, [2, 2, 2, 2, 1]), (3, [3, 3, 3, 2, 2])],
)
def test_batch_size(input_data, output_data, batch_size, batch_size_output, empty_exa):
    """
    Runs an algorithm with different batch sizes.
    """
    runner = UDFRunner(
        algorithm={"algorithm_name": "batch-size-logger"}, batch_size=batch_size
    )

    meta_data = create_meta_data(batch_logger=runner.algorithm)
    ctx = StandaloneMockContext(input_data, meta_data)
    runner.run(ctx, empty_exa)

    expected_setup = '{"BatchSizeLogger": {"algorithm_name": "batch-size-logger"}}'
    expected_output = [
        (*text_row[:-1], b_size, None, expected_setup)
        for text_row, b_size in zip(output_data, batch_size_output)
    ]
    assert ctx.output == expected_output


@patch(
    "exasol.ai.text.impl.nlp.algorithms.testing.dummy_word_counter.DummyWordCounter.run_impl"
)
def test_run_with_error(mock_run_impl, input_data, output_data, empty_exa):

    error_message = "This is a daft algorithm"
    mock_run_impl.side_effect = ValueError(error_message)

    runner = UDFRunner(algorithm={"algorithm_name": "dummy-word-counter"}, batch_size=5)

    meta_data = create_meta_data(word_counter=runner.algorithm)
    ctx = StandaloneMockContext(input_data, meta_data)
    runner.run(ctx, empty_exa)

    # Verify all output columns are empty (ignore error column).
    expected_setup = '{"DummyWordCounter": {"algorithm_name": "dummy-word-counter"}}'
    expected_output_bar_error = [
        (*text_row[:-1], None, expected_setup) for text_row in output_data
    ]
    output_bar_error = [(*row[:ERROR_COL_ID], row[SETUP_COL_ID]) for row in ctx.output]
    assert output_bar_error == expected_output_bar_error

    # Verify error column contains the error message.
    assert all(error_message in row[ERROR_COL_ID] for row in ctx.output)
