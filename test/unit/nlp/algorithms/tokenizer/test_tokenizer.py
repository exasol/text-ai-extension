import dataclasses
from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import (
    Token,
    TokenContent,
)
from exasol.ai.text.impl.utils.char_span import CharSpan
from test.utils.dataframe_matcher import DataFrameMatcher

pytest_plugins = ["test.utils.tokenizer_test_cases"]


@patch("test.utils.tokenizer_test_cases.TestTokenizer.tokenize_texts")
def test_run_impl(
    mock_tokenize_texts,
    empty_exa,
    tokenizer,
    expected_tokenizer_output_dataframe,
    tokenizer_input_dataframe,
    tokenizer_test_cases,
):
    mock_tokenize_texts.side_effect = [
        [
            [
                Token(
                    span=CharSpan(
                        doc_id=-1,
                        char_begin=token.span.char_begin - test_case.text_char_begin,
                        char_end=token.span.char_end - test_case.text_char_begin,
                    ),
                    content=token.content,
                )
                for token in test_case.tokens
            ]
            for test_case in tokenizer_test_cases
        ]
    ]
    actual_output_dataframe = tokenizer.run_impl(tokenizer_input_dataframe, empty_exa)
    matcher = DataFrameMatcher(
        expected_tokenizer_output_dataframe, actual_output_dataframe
    )
    pd.options.display.max_columns = 100
    assert matcher, matcher.diff()
    assert mock_tokenize_texts


@patch("test.utils.tokenizer_test_cases.TestTokenizer.tokenize_texts")
def test_run_impl_empty(mock_tokenize_texts, empty_exa, tokenizer):
    input_df = pd.DataFrame(
        [(100, 10, 20, "a"), (200, 20, 30, "b")], columns=tokenizer.inputs.column_names
    )

    mock_tokenize_texts.side_effect = [[[], []]]
    expected_output_df = tokenizer.set_empty(input_df)
    output_df = tokenizer.run_impl(input_df, empty_exa)
    matcher = DataFrameMatcher(expected_output_df, output_df)
    pd.options.display.max_columns = 100
    assert matcher, matcher.diff()
    assert mock_tokenize_texts


@pytest.mark.parametrize("error_message", [None, "message"])
def test_set_empty(
    error_message, tokenizer, tokenizer_input_dataframe, tokenizer_test_cases
):
    tokenizer_input_dataframe = tokenizer_input_dataframe.drop(
        columns=tokenizer.text_column
    )
    output_df = tokenizer.set_empty(
        tokenizer_input_dataframe, error_message=error_message
    )
    output_df.reset_index(drop=True, inplace=True)

    expected_output_df = pd.DataFrame(
        [
            [
                test_case.text_doc_id,
                test_case.text_char_begin,
                test_case.text_char_begin + len(test_case.text),
                None,
                None,
                None,
            ]
            + list(dataclasses.astuple(TokenContent(token=None)))
            + [error_message, tokenizer.algo_setup]
            for test_case in tokenizer_test_cases
        ],
        columns=tokenizer.outputs.column_names,
    )

    pd.options.display.max_columns = 100
    matcher = DataFrameMatcher(expected_output_df, output_df)
    assert matcher, matcher.diff()
