from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.named_entity import HftNamedEntityRecognition


@pytest.fixture
def ner() -> HftNamedEntityRecognition:
    return HftNamedEntityRecognition(
        model_name="my_model",
        bucketfs_conn_name="my_bfs_conn",
        sub_dir="my_model_subdir",
        setup="my setup",
    )


@pytest.mark.parametrize("entity_type_key", ["entity", "entity_group"])
@patch(
    "exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition.HftNamedEntityRecognition.run_classifier"
)
def test_run_impl(mock_run_classifier, entity_type_key, ner, empty_exa):

    # Simulate results of run_classifier() for two pieces of text.
    mock_run_classifier.return_value = [
        [
            {
                "end": 14,
                entity_type_key: "Age",
                "score": 0.998,
                "start": 3,
                "word": "63 - year - old",
            },
            {
                "end": 20,
                entity_type_key: "Sex",
                "score": 0.999,
                "start": 15,
                "word": "woman",
            },
        ],
        [
            {
                "end": 124,
                entity_type_key: "Detailed_description",
                "score": 0.797,
                "start": 105,
                "word": "progressively worse",
            },
            {
                "end": 137,
                entity_type_key: "Sign_symptom",
                "score": 0.999,
                "start": 129,
                "word": "headache",
            },
            {
                "end": 171,
                entity_type_key: "Detailed_description",
                "score": 0.875,
                "start": 156,
                "word": "lightheadedness",
            },
        ],
    ]

    text_span_columns = ner.inputs.text_span.column_names
    entity_span_columns = ner.outputs.entity_span.column_names
    input_df = pd.DataFrame(
        [(100, 3, 1000, "xxx"), (200, 40, 1000, "yyy")], columns=ner.inputs.column_names
    )

    expected_df = pd.DataFrame(
        [
            (
                100,
                3,
                1000,
                "xxx",
                "Age",
                0.998,
                "63 - year - old",
                100,
                6,
                17,
                None,
                "my setup",
            ),
            (100, 3, 1000, "xxx", "Sex", 0.999, "woman", 100, 18, 23, None, "my setup"),
            (
                200,
                40,
                1000,
                "yyy",
                "Detailed_description",
                0.797,
                "progressively worse",
                200,
                145,
                164,
                None,
                "my setup",
            ),
            (
                200,
                40,
                1000,
                "yyy",
                "Sign_symptom",
                0.999,
                "headache",
                200,
                169,
                177,
                None,
                "my setup",
            ),
            (
                200,
                40,
                1000,
                "yyy",
                "Detailed_description",
                0.875,
                "lightheadedness",
                200,
                196,
                211,
                None,
                "my setup",
            ),
        ],
        columns=[
            *text_span_columns,
            ner.text_column,
            ner.entity_type_column,
            ner.score_column,
            ner.entity_column,
            *entity_span_columns,
            ner.error_message_column,
            ner.setup_column,
        ],
    )

    output_df = ner.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)


@pytest.mark.parametrize("entity_type_key", ["entity", "entity_group"])
@patch(
    "exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition.HftNamedEntityRecognition.run_classifier"
)
def test_run_impl_partially_empty(mock_run_classifier, entity_type_key, ner, empty_exa):

    # Simulate results of run_classifier() for two pieces of text.
    mock_run_classifier.return_value = [
        [],
        [
            {
                "end": 14,
                entity_type_key: "Age",
                "score": 0.998,
                "start": 3,
                "word": "63 - year - old",
            },
            {
                "end": 20,
                entity_type_key: "Sex",
                "score": 0.999,
                "start": 15,
                "word": "woman",
            },
        ],
        [],
    ]

    text_span_columns = ner.inputs.text_span.column_names
    entity_span_columns = ner.outputs.entity_span.column_names
    input_df = pd.DataFrame(
        [(300, 199, 9000, "zzz"), (100, 3, 1000, "xxx"), (200, 40, 1000, "yyy")],
        columns=[*text_span_columns, ner.text_column],
    )

    expected_df = pd.DataFrame(
        [
            (
                300,
                199,
                9000,
                "zzz",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "my setup",
            ),
            (
                100,
                3,
                1000,
                "xxx",
                "Age",
                0.998,
                "63 - year - old",
                100,
                6,
                17,
                None,
                "my setup",
            ),
            (100, 3, 1000, "xxx", "Sex", 0.999, "woman", 100, 18, 23, None, "my setup"),
            (
                200,
                40,
                1000,
                "yyy",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "my setup",
            ),
        ],
        columns=[
            *text_span_columns,
            ner.text_column,
            ner.entity_type_column,
            ner.score_column,
            ner.entity_column,
            *entity_span_columns,
            ner.error_message_column,
            ner.setup_column,
        ],
    )

    output_df = ner.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition.HftNamedEntityRecognition.run_classifier"
)
def test_run_impl_error(mock_run_classifier, ner, empty_exa):

    # Simulate results of run_classifier() for two pieces of text.
    mock_run_classifier.return_value = [
        [
            {"end": 14, "score": 0.998, "start": 3, "word": "63 - year - old"},
            {"end": 20, "score": 0.999, "start": 15, "word": "woman"},
        ]
    ]

    text_span_columns = ner.inputs.text_span.column_names
    input_df = pd.DataFrame(
        [(100, 3, 1000, "xxx")], columns=[*text_span_columns, ner.text_column]
    )
    with pytest.raises(KeyError):
        ner.run_impl(input_df, empty_exa)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition.HftNamedEntityRecognition.run_classifier"
)
def test_run_impl_empty(mock_run_classifier, ner, empty_exa):

    # Simulate results of run_classifier() for two pieces of text.
    mock_run_classifier.return_value = [[], []]

    text_span_columns = ner.inputs.text_span.column_names
    entity_span_columns = ner.outputs.entity_span.column_names
    input_df = pd.DataFrame(
        [(100, 3, 1000, "xxx"), (200, 40, 1000, "yyy")],
        columns=[*text_span_columns, ner.text_column],
    )

    expected_df = ner.set_empty(input_df)
    output_df = ner.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)
