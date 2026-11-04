import pandas as pd

from exasol.ai.text.impl.nlp.algorithms.tokenizer.spacy_tokenizer import SpacyTokenizer
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import Token
from exasol.ai.text.impl.utils.char_span import CharSpan
from test.utils.dataframe_matcher import DataFrameMatcher


def test_tokenize(
    setup_no_database, spacy_pipeline, tokenizer_input_dataframe, tokenizer_test_cases
):
    tokenizer = SpacyTokenizer()
    actual_tokens = tokenizer.tokenize_texts(
        tokenizer_input_dataframe[tokenizer.text_column]
    )
    expected_tokens = [
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
    assert actual_tokens == expected_tokens


def test_run_impl_success(
    setup_no_database,
    spacy_pipeline,
    tokenizer_input_dataframe,
    expected_tokenizer_output_dataframe,
    tokenizer_test_cases,
    empty_exa,
):
    tokenizer = SpacyTokenizer(setup="my setup")
    expected_tokenizer_output_dataframe[tokenizer.setup_column] = "my setup"
    actual_output_dataframe = tokenizer.run_impl(tokenizer_input_dataframe, empty_exa)
    actual_output_dataframe.reset_index(drop=True, inplace=True)

    pd.options.display.max_columns = 100
    matcher = DataFrameMatcher(
        expected_tokenizer_output_dataframe, actual_output_dataframe
    )
    assert matcher, matcher.diff()
