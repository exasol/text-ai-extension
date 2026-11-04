from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.topics import HftTopicClassification
from exasol.ai.text.impl.nlp.algorithms.topics.topic_classification import (
    TopicClassifierOutput,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec
from test.integration.utils.db_test_utils import (
    common_checks,
    get_input_df,
)

TEXT_COLUMN = "text"
ERROR_MESSAGE_COLUMN = "error_message"
UPPER_SCORE_THRESHOLD = 0.5
LOWER_SCORE_THRESHOLD = 0.2


@pytest.fixture
def text_span() -> SpanColumns:
    return SpanColumns.create(TEXT_COLUMN)


@pytest.fixture
def topic_inputs(text_span) -> TextInput:
    return TextInput(text=varchar_column_spec(TEXT_COLUMN), text_span=text_span)


@pytest.fixture
def topic_outputs(text_span) -> TopicClassifierOutput:
    return TopicClassifierOutput(
        error_message=varchar_column_spec(ERROR_MESSAGE_COLUMN), text_span=text_span
    )


@pytest.fixture
def input_df(topic_test_cases, text_span) -> pd.DataFrame:
    return get_input_df(topic_test_cases, TEXT_COLUMN, text_span)


def test_topic_classification(
    setup_no_database,
    model_subdir,
    topic_model,
    topic_model_location,
    topic_test_cases,
    topic_list,
    input_df,
    empty_exa,
    text_span,
    topic_inputs,
    topic_outputs,
):
    """
    Tests that the topic classification algorithm ranks the correct topic as the top
    one and assigns a high score to it. All other topics should get a low score.
    """

    algo = HftTopicClassification(
        model_name=topic_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        topics=topic_list,
        inputs=topic_inputs,
        outputs=topic_outputs,
    )
    text_id_column, _, _ = text_span.column_names

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = topic_model_location
        output_df = algo.run_impl(input_df, empty_exa)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )

    # Check that the correct topic was ranked at the top.
    top_df = output_df[
        (output_df[algo.score_column] >= UPPER_SCORE_THRESHOLD)
        & (output_df[algo.rank_column] == 1)
    ]
    top_df = top_df[[text_id_column, algo.topic_column]]
    top_rows = {tuple(row) for row in top_df.itertuples(index=False, name=None)}
    expected_rows = {
        (text_case.text_doc_id, text_case.topic) for text_case in topic_test_cases
    }
    assert top_rows == expected_rows

    # Check that all other topics have a low score.
    bottom_df = output_df[output_df[algo.rank_column] > 1]
    assert bottom_df[algo.score_column].max() <= LOWER_SCORE_THRESHOLD
