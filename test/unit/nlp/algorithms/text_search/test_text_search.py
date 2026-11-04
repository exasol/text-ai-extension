from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.text_search.text_search import (
    PhraseLocations,
    TextSearch,
)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.text_search.text_search.TextSearch.locate_phrases"
)
def test_run_impl(mock_locate_phrases, empty_exa):
    # Simulate results of locate_phrases(). This function is called once for each
    # distinct document span in the input. See the function's docstring for details.
    text1, text2, text3 = "abcdefg", "noooo", "xxx-yyy"
    mock_search_results = {
        text1: [("dog", 3, 6), ("cat", 20, 23), ("dog", 30, 33), ("cat", 50, 53)],
        text2: [],
        text3: [
            ("pig", 50, 53),
            ("horse", 3, 8),
            ("cow", 32, 35),
            ("pig", 46, 49),
            ("horse", 39, 44),
        ],
    }

    def mock_phrase_locations(text: str, phrases: Iterable[str]) -> PhraseLocations:
        phrase_map = {phrase: phrase_id for phrase_id, phrase in enumerate(phrases)}
        result = mock_search_results[text]
        return [(phrase_map[res[0]], res[1], res[2]) for res in result]

    mock_locate_phrases.side_effect = mock_phrase_locations

    searcher = TextSearch(setup="my setup")
    phrase_input_column = searcher.phrase_input_column

    text_span_columns = searcher.inputs.text_span.column_names
    phrase_output_column = searcher.phrase_output_column
    phrase_span_columns = searcher.outputs.phrase_span.column_names

    # Here we will also check that existing output columns are ignored
    # by adding one of the columns of the phrase span to the input_df.
    input_df = pd.DataFrame(
        [
            (100, 3, 1000, text1, "dog", 20.0, 0, None),
            (100, 3, 1000, text1, "cat", 4.5, 666, "oops"),
            (300, 5, 500, text2, "pigeon", 3.3, 0, None),
            (300, 5, 500, text2, "chicken", 2.3, 0, None),
            (200, 40, 1000, text3, "horse", 420.0, 0, None),
            (200, 40, 1000, text3, "pig", 250, 0, None),
            (200, 40, 1000, text3, "cow", 510.0, 0, None),
            (200, 40, 1000, text3, "unicorn", 400.0, 0, None),
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            phrase_input_column,
            "weight",
            phrase_span_columns[1],
            searcher.error_message_column,
        ],
    )

    # The expected locations of the phrases are provided by the mock. They do not
    # correspond to the locations of the phrases in the text (of which there are None).
    expected_output_df = pd.DataFrame(
        [
            (100, 3, 1000, text1, "cat", 4.5, 100, 23, 26, None, "my setup"),
            (100, 3, 1000, text1, "cat", 4.5, 100, 53, 56, None, "my setup"),
            (100, 3, 1000, text1, "dog", 20.0, 100, 6, 9, None, "my setup"),
            (100, 3, 1000, text1, "dog", 20.0, 100, 33, 36, None, "my setup"),
            (200, 40, 1000, text3, "cow", 510.0, 200, 72, 75, None, "my setup"),
            (200, 40, 1000, text3, "horse", 420.0, 200, 43, 48, None, "my setup"),
            (200, 40, 1000, text3, "horse", 420.0, 200, 79, 84, None, "my setup"),
            (200, 40, 1000, text3, "pig", 250, 200, 86, 89, None, "my setup"),
            (200, 40, 1000, text3, "pig", 250, 200, 90, 93, None, "my setup"),
            (
                200,
                40,
                1000,
                text3,
                "unicorn",
                400.0,
                None,
                None,
                None,
                None,
                "my setup",
            ),
            (300, 5, 500, text2, "chicken", 2.3, None, None, None, None, "my setup"),
            (300, 5, 500, text2, "pigeon", 3.3, None, None, None, None, "my setup"),
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            phrase_output_column,
            "weight",
            *phrase_span_columns,
            searcher.error_message_column,
            searcher.setup_column,
        ],
    )

    output_df = searcher.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    output_df.sort_values(
        by=[*text_span_columns, phrase_output_column, *phrase_span_columns],
        inplace=True,
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.text_search.text_search.TextSearch.locate_phrases"
)
def test_run_impl_empty_result(mock_locate_phrases, empty_exa):
    """
    This test validates the case when none of the phrases is found in all documents.
    """
    text1, text2, text3 = "abcdefg", "noooo", "xxx-yyy"
    mock_locate_phrases.side_effect = [[], []]

    searcher = TextSearch(setup="my setup")
    phrase_input_column = searcher.phrase_input_column

    text_span_columns = searcher.inputs.text_span.column_names
    phrase_output_column = searcher.phrase_output_column
    phrase_span_columns = searcher.outputs.phrase_span.column_names

    input_df = pd.DataFrame(
        [
            (100, 3, 1000, text1, "dog"),
            (100, 3, 1000, text1, "cat"),
            (200, 40, 1000, text3, "horse"),
            (200, 40, 1000, text3, "pig"),
            (200, 40, 1000, text3, "cow"),
        ],
        columns=[*text_span_columns, searcher.text_column, phrase_input_column],
    )

    expected_output_df = pd.DataFrame(
        [
            (100, 3, 1000, text1, "cat", None, None, None, None, "my setup"),
            (100, 3, 1000, text1, "dog", None, None, None, None, "my setup"),
            (200, 40, 1000, text3, "cow", None, None, None, None, "my setup"),
            (200, 40, 1000, text3, "horse", None, None, None, None, "my setup"),
            (200, 40, 1000, text3, "pig", None, None, None, None, "my setup"),
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            phrase_output_column,
            *phrase_span_columns,
            searcher.error_message_column,
            searcher.setup_column,
        ],
    )
    for col in phrase_span_columns:
        expected_output_df[col] = expected_output_df[col].astype(float)

    output_df = searcher.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    output_df.sort_values(
        by=[*text_span_columns, phrase_output_column], inplace=True, ignore_index=True
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@patch(
    "exasol.ai.text.impl.nlp.algorithms.text_search.text_search.TextSearch.locate_phrases"
)
def test_run_impl_empty_phrase(mock_locate_phrases, empty_exa):
    """
    This test validates the case of an empty searched phrase in the input.
    """
    searcher = TextSearch(setup="my setup")
    phrase_input_column = searcher.phrase_input_column

    text_span_columns = searcher.inputs.text_span.column_names
    phrase_output_column = searcher.phrase_output_column
    phrase_span_columns = searcher.outputs.phrase_span.column_names

    input_df = pd.DataFrame(
        [
            (100, 10, 30, "abc", ""),
            (100, 30, 40, "abc", None),
        ],
        columns=[*text_span_columns, searcher.text_column, phrase_input_column],
    )

    expected_output_df = pd.DataFrame(
        [
            (100, 10, 30, "abc", "", None, None, None, None, "my setup"),
            (100, 30, 40, "abc", None, None, None, None, None, "my setup"),
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            phrase_output_column,
            *phrase_span_columns,
            searcher.error_message_column,
            searcher.setup_column,
        ],
    )
    for col in phrase_span_columns:
        expected_output_df[col] = expected_output_df[col].astype(float)

    output_df = searcher.run_impl(input_df, empty_exa)
    mock_locate_phrases.assert_not_called()
    output_df.reset_index(drop=True, inplace=True)
    output_df.sort_values(by=text_span_columns, inplace=True, ignore_index=True)
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


@pytest.mark.parametrize("error_message", [None, "message"])
def test_set_empty(error_message):

    searcher = TextSearch()

    text_span_columns = searcher.inputs.text_span.column_names
    text_id_column, _, _ = text_span_columns
    phrase_id_column, phrase_begin_column, phrase_end_column = (
        searcher.outputs.phrase_span.column_names
    )

    input_df = pd.DataFrame(
        [(100, 3, 1000, "abcdefg", "x"), (100, 3, 1000, "abcdefg", "y")],
        columns=[
            *text_span_columns,
            searcher.text_column,
            searcher.phrase_input_column,
        ],
    )

    output_df = searcher.set_empty(input_df, error_message=error_message)

    # Check the expected values of the created columns
    assert output_df[phrase_id_column].isnull().all()
    assert output_df[phrase_begin_column].isnull().all()
    assert output_df[phrase_end_column].isnull().all()
    if error_message:
        assert (output_df[searcher.error_message_column] == error_message).all()
    else:
        assert output_df[searcher.error_message_column].isnull().all()
