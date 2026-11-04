from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.topics import HftTopicClassification


@pytest.fixture
def topic_classifier() -> HftTopicClassification:
    return HftTopicClassification(
        model_name="my_model",
        bucketfs_conn_name="my_bfs_conn",
        sub_dir="my_model_subdir",
        topics=["physics", "chemistry", "mathematics"],
        setup="my setup",
    )


@patch(
    "exasol.ai.text.impl.nlp.algorithms.topics.hft_topic_classification.HftTopicClassification.run_classifier"
)
def test_run_impl(mock_run_classifier, topic_classifier, empty_exa):

    # Simulate results of run_classifier() for two pieces of text.
    mock_run_classifier.return_value = [
        {"labels": ["mathematics", "physics", "chemistry"], "scores": [0.4, 0.3, 0.1]},
        {"labels": ["chemistry", "physics", "mathematics"], "scores": [0.7, 0.5, 0.2]},
    ]

    text_span_columns = topic_classifier.inputs.text_span.column_names
    input_df = pd.DataFrame(
        [(100, 3, 6, "xxx"), (200, 40, 43, "yyy")],
        columns=topic_classifier.inputs.column_names,
    )

    expected_df = pd.DataFrame(
        [
            (100, 3, 6, "xxx", "mathematics", 0.4, 1, None, "my setup"),
            (100, 3, 6, "xxx", "physics", 0.3, 2, None, "my setup"),
            (100, 3, 6, "xxx", "chemistry", 0.1, 3, None, "my setup"),
            (200, 40, 43, "yyy", "chemistry", 0.7, 1, None, "my setup"),
            (200, 40, 43, "yyy", "physics", 0.5, 2, None, "my setup"),
            (200, 40, 43, "yyy", "mathematics", 0.2, 3, None, "my setup"),
        ],
        columns=[
            *text_span_columns,
            topic_classifier.text_column,
            topic_classifier.topic_column,
            topic_classifier.score_column,
            topic_classifier.rank_column,
            topic_classifier.error_message_column,
            topic_classifier.setup_column,
        ],
    )

    output_df = topic_classifier.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)
