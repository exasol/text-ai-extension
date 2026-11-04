from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.keywords import KeywordExtractor


@pytest.fixture
def extractor():
    return KeywordExtractor(setup="my setup")


@pytest.fixture
def input_df(extractor) -> pd.DataFrame:
    return pd.DataFrame(
        [(123, 100, 118, "aaa"), (456, 200, 212, "bbb"), (789, 300, 313, "ccc")],
        columns=extractor.inputs.column_names,
    )


@patch(
    "exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor.KeywordExtractor.extract_keywords"
)
def test_run_impl(mock_extract_keywords, extractor, input_df, empty_exa):

    mock_extract_keywords.side_effect = [
        [
            [("kw11", 0.3), ("kw12", 0.4)],
            [("kw21", 0.6)],
            [("kw31", 0.5), ("kw32", 0.8)],
        ]
    ]

    output_df = extractor.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)

    expected_output_df = pd.DataFrame(
        [
            (123, 100, 118, "aaa", "kw11", 0.3, None, "my setup"),
            (123, 100, 118, "aaa", "kw12", 0.4, None, "my setup"),
            (456, 200, 212, "bbb", "kw21", 0.6, None, "my setup"),
            (789, 300, 313, "ccc", "kw31", 0.5, None, "my setup"),
            (789, 300, 313, "ccc", "kw32", 0.8, None, "my setup"),
        ],
        columns=[
            *input_df.columns,
            extractor.keyword_column,
            extractor.score_column,
            extractor.error_message_column,
            extractor.setup_column,
        ],
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor.KeywordExtractor.extract_keywords"
)
def test_run_impl_empty(mock_extract_keywords, extractor, input_df, empty_exa):

    mock_extract_keywords.side_effect = [[[], [], []]]

    output_df = extractor.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)

    expected_output_df = extractor.set_empty(input_df)
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor.KeywordExtractor.extract_keywords"
)
def test_run_impl_partially_empty(
    mock_extract_keywords, extractor, input_df, empty_exa
):

    mock_extract_keywords.side_effect = [
        [[("kw11", 0.3), ("kw12", 0.4)], [], [("kw31", 0.5), ("kw32", 0.8)]]
    ]

    output_df = extractor.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)

    expected_output_df = pd.DataFrame(
        [
            (123, 100, 118, "aaa", "kw11", 0.3, None, "my setup"),
            (123, 100, 118, "aaa", "kw12", 0.4, None, "my setup"),
            (456, 200, 212, "bbb", None, None, None, "my setup"),
            (789, 300, 313, "ccc", "kw31", 0.5, None, "my setup"),
            (789, 300, 313, "ccc", "kw32", 0.8, None, "my setup"),
        ],
        columns=[
            *input_df.columns,
            extractor.keyword_column,
            extractor.score_column,
            extractor.error_message_column,
            extractor.setup_column,
        ],
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@pytest.mark.parametrize("error_message", [None, "message"])
def test_set_empty(extractor, input_df, error_message):

    output_df = extractor.set_empty(input_df, error_message=error_message)

    # Check the default values of the created columns.
    assert output_df[extractor.keyword_column].isnull().all()
    assert output_df[extractor.score_column].isnull().all()
    if error_message:
        assert (output_df[extractor.error_message_column] == error_message).all()
    else:
        assert output_df[extractor.error_message_column].isnull().all()
