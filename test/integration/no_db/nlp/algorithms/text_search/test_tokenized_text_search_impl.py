import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)


@pytest.mark.parametrize("case_no", [0, 1])
def test_locate_phrases(
    setup_no_database, spacy_pipeline, text_search_test_cases, case_no
):
    test_case = text_search_test_cases[case_no]
    searcher = TokenizedTextSearch()
    result = searcher.locate_phrases(
        test_case.text, [phrase.phrase for phrase in test_case.phrases_occurrences]
    )
    expected_result = {
        (phrase_seq, loc, loc + len(phrase.phrase))
        for phrase_seq, phrase in enumerate(test_case.phrases_occurrences)
        for loc in phrase.occurrences
    }
    assert set(result) == expected_result


@pytest.mark.parametrize("case_no", [0, 1])
def test_locate_phrase_with_newlines(
    setup_no_database, spacy_pipeline, text_search_test_cases, case_no
):
    """
    In this test we do not ignore the spaces, hence expect to find less occurrences
    of the test phrase due to the presence of the newline in some of them.
    """
    test_case = text_search_test_cases[case_no]
    text = test_case.text
    searcher = TokenizedTextSearch(ignore_spaces=False)
    result = searcher.locate_phrases(
        test_case.text, [phrase.phrase for phrase in test_case.phrases_occurrences]
    )
    expected_result = {
        (phrase_seq, loc, loc + len(phrase.phrase))
        for phrase_seq, phrase in enumerate(test_case.phrases_occurrences)
        for loc in phrase.occurrences
        if "\n" not in text[loc : loc + len(phrase.phrase)]
    }
    assert set(result) == expected_result


def test_locate_phrases_fuzzy(
    setup_no_database, spacy_pipeline, text_search_test_cases
):
    """
    In this test we will adulterate the input text and check if the fuzzy text search
    still finds all the occurrences. We will change one character in the middle of the
    longest word at the first occurrence of the first search phrase.
    """
    test_case = text_search_test_cases[0]
    text = test_case.text
    phrases_occurrences = test_case.phrases_occurrences[0]
    phrase = phrases_occurrences.phrase

    # Choose the word to modify.
    longest_word = sorted(phrase.split(), key=lambda word: len(word))[-1]
    # Choose the character in the centre of the word to modify.
    centre = phrase.find(longest_word) + (len(longest_word) // 2)
    hit_position = phrases_occurrences.occurrences[0] + centre
    # Get a new character
    old_char = text[hit_position]
    new_char = "a" if old_char.lower() != "a" else "e"
    # Make the modification in the text.
    text = text[:hit_position] + new_char + text[hit_position + 1 :]

    searcher = TokenizedTextSearch(fuzziness=2)
    result = searcher.locate_phrases(
        text, [phrase.phrase for phrase in test_case.phrases_occurrences]
    )
    expected_result = {
        (phrase_seq, loc, loc + len(phrase.phrase))
        for phrase_seq, phrase in enumerate(test_case.phrases_occurrences)
        for loc in phrase.occurrences
    }
    assert set(result) == expected_result


def test_locate_phrases_with_space(setup_no_database, spacy_pipeline):
    """
    Test that a phrase corrupted with spaces can still be found if we ignore spaces.
    """
    text = "some text"
    searcher = TokenizedTextSearch()
    result = searcher.locate_phrases(text, ["some     text"])
    expected_result = [(0, 0, 9)]
    assert result == expected_result


def test_locate_phrases_empty(setup_no_database, spacy_pipeline):
    """
    Test of an edge case when the phrase to search contains nothing but spaces.
    """
    text = "some text"
    searcher = TokenizedTextSearch()
    with pytest.raises(ValueError):
        searcher.locate_phrases(text, ["some", "     ", "text"])


def test_run_impl(setup_no_database, spacy_pipeline, text_search_test_cases, empty_exa):

    searcher = TokenizedTextSearch(setup="my setup")
    text_span_columns = searcher.inputs.text_span.column_names

    input_df = pd.DataFrame(
        [
            (
                test_case.text_doc_id,
                test_case.text_char_begin,
                test_case.text_char_begin + len(test_case.text),
                test_case.text,
                phrase.phrase,
            )
            for test_case in text_search_test_cases
            for phrase in test_case.phrases_occurrences
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            searcher.phrase_input_column,
        ],
    )

    output_df = searcher.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)

    expected_output_df = pd.DataFrame(
        [
            (
                test_case.text_doc_id,
                test_case.text_char_begin,
                test_case.text_char_begin + len(test_case.text),
                test_case.text,
                phrase.phrase,
                test_case.text_doc_id,
                loc + test_case.text_char_begin,
                loc + len(phrase.phrase) + test_case.text_char_begin,
                None,
                "my setup",
            )
            for test_case in text_search_test_cases
            for phrase in test_case.phrases_occurrences
            for loc in phrase.occurrences
        ],
        columns=[
            *text_span_columns,
            searcher.text_column,
            searcher.phrase_output_column,
            *searcher.outputs.phrase_span.column_names,
            searcher.error_message_column,
            searcher.setup_column,
        ],
    )
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)
