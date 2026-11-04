from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.named_entity import HftNamedEntityRecognition
from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityOutput,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec
from test.integration.utils.db_test_utils import (
    common_checks,
    get_input_df,
)

TEXT_COLUMN = "text"
ERROR_MESSAGE_COLUMN = "error_message"
MIN_SCORE = 0.85


@pytest.fixture
def text_span() -> SpanColumns:
    return SpanColumns.create(TEXT_COLUMN)


@pytest.fixture
def named_entity_inputs(text_span) -> TextInput:
    return TextInput(text=varchar_column_spec(TEXT_COLUMN), text_span=text_span)


@pytest.fixture
def named_entity_outputs(text_span) -> NamedEntityOutput:
    return NamedEntityOutput(
        error_message=varchar_column_spec(ERROR_MESSAGE_COLUMN), text_span=text_span
    )


@pytest.fixture
def input_df(named_entity_test_cases, text_span) -> pd.DataFrame:
    return get_input_df(named_entity_test_cases, TEXT_COLUMN, text_span)


@pytest.mark.parametrize("strategy", ["none", "simple", "max"])
def test_named_entity_recognition(
    setup_no_database,
    strategy,
    model_subdir,
    named_entity_model,
    named_entity_model_location,
    named_entity_test_cases,
    input_df,
    empty_exa,
    text_span,
    named_entity_inputs,
    named_entity_outputs,
):
    """
    Tests that the named entity algorithm retrieves all the expected entities.
    Only entities with the score greater or equal to MIN_SCORE are counted.
    """

    algo = HftNamedEntityRecognition(
        model_name=named_entity_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        aggregation_strategy=strategy,
        inputs=named_entity_inputs,
        outputs=named_entity_outputs,
    )
    text_id_column, text_begin_column, _ = text_span.column_names
    _, entity_begin_column, _ = algo.outputs.entity_span.column_names

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = named_entity_model_location
        output_df = algo.run_impl(input_df, empty_exa)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )
    output_df = output_df[output_df[algo.score_column] >= MIN_SCORE]
    output_df = output_df[
        [
            text_id_column,
            algo.entity_type_column,
            algo.entity_column,
            entity_begin_column,
        ]
    ]
    output_rows = {tuple(row) for row in output_df.itertuples(index=False, name=None)}

    expected_rows = {
        (
            text_case.text_doc_id,
            expected_entity.entity_type,
            expected_entity.entity_text,
            expected_entity.position + text_case.text_char_begin,
        )
        for text_case in named_entity_test_cases
        for strategy_entities in text_case.expected_entities
        for expected_entity in strategy_entities.entities
        if strategy in strategy_entities.strategies
    }
    assert expected_rows.issubset(output_rows)


@pytest.mark.parametrize("strategy", ["none", "simple", "max"])
def test_ner_ignore_labels(
    setup_no_database,
    strategy,
    model_subdir,
    named_entity_model,
    named_entity_model_location,
    named_entity_test_cases,
    named_entity_ignore_labels,
    input_df,
    empty_exa,
    text_span,
    named_entity_inputs,
    named_entity_outputs,
):
    """
    Tests that the labels provided in the ignore_labels list are excluded from the
    output.
    """

    ignore_labels = named_entity_ignore_labels[strategy]
    algo = HftNamedEntityRecognition(
        model_name=named_entity_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        aggregation_strategy=strategy,
        ignore_labels=ignore_labels,
        inputs=named_entity_inputs,
        outputs=named_entity_outputs,
    )
    text_id_column, text_begin_column, _ = text_span.column_names
    _, entity_begin_column, _ = algo.outputs.entity_span.column_names

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = named_entity_model_location
        output_df = algo.run_impl(input_df, empty_exa)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )
    # There should be some output for all test cases
    expected_text_ids = {text_case.text_doc_id for text_case in named_entity_test_cases}
    assert set(output_df[text_id_column]) == expected_text_ids
    # But not including the labels designated to be ignored.
    assert not set(output_df[algo.entity_type_column]).intersection(ignore_labels)
