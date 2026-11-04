import dataclasses
from dataclasses import dataclass
from textwrap import dedent

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.tokenizer.spacy_tokenizer import SpacyTokenizer
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import (
    Token,
    TokenContent,
    Tokenizer,
)
from exasol.ai.text.impl.utils.char_span import CharSpan
from test.utils.test_case_utils import TextTestCase


class TestTokenizer(Tokenizer):

    def tokenize_texts(self, texts):
        pass


@pytest.fixture
def tokenizer() -> Tokenizer:
    return TestTokenizer()


@dataclass
class TokenTestCase(TextTestCase):
    tokens: list[Token]


@pytest.fixture
def tokenizer_test_cases() -> list[TokenTestCase]:
    return [
        TokenTestCase(
            text_doc_id=1234,
            text_char_begin=200,
            text=dedent("""
                This is a test.
            """).strip(),
            tokens=[
                Token(
                    span=CharSpan(doc_id=1234, char_begin=200, char_end=204),
                    content=TokenContent(
                        token="This",
                        lemma="this",
                        pos="DET",
                        tag="DT",
                        shape="Xxxx",
                        is_upper=False,
                        is_lower=False,
                        is_title=True,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=1234, char_begin=205, char_end=207),
                    content=TokenContent(
                        token="is",
                        lemma="be",
                        pos="AUX",
                        tag="VBZ",
                        shape="xx",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=1234, char_begin=208, char_end=209),
                    content=TokenContent(
                        token="a",
                        lemma="a",
                        pos="DET",
                        tag="DT",
                        shape="x",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=1234, char_begin=210, char_end=214),
                    content=TokenContent(
                        token="test",
                        lemma="test",
                        pos="NOUN",
                        tag="NN",
                        shape="xxxx",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=False,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=1234, char_begin=214, char_end=215),
                    content=TokenContent(
                        token=".",
                        lemma=".",
                        pos="PUNCT",
                        tag=".",
                        shape=".",
                        is_upper=False,
                        is_lower=False,
                        is_title=False,
                        is_punct=True,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=False,
                        is_stop=False,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
            ],
        ),
        TokenTestCase(
            text_doc_id=5678,
            text_char_begin=500,
            text=dedent("""
                    This was another test.
                """).strip(),
            tokens=[
                Token(
                    span=CharSpan(doc_id=5678, char_begin=500, char_end=504),
                    content=TokenContent(
                        token="This",
                        lemma="this",
                        pos="DET",
                        tag="DT",
                        shape="Xxxx",
                        is_upper=False,
                        is_lower=False,
                        is_title=True,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=5678, char_begin=505, char_end=508),
                    content=TokenContent(
                        token="was",
                        lemma="be",
                        pos="AUX",
                        tag="VBD",
                        shape="xxx",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=5678, char_begin=509, char_end=516),
                    content=TokenContent(
                        token="another",
                        lemma="another",
                        pos="DET",
                        tag="DT",
                        shape="xxxx",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=True,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=5678, char_begin=517, char_end=521),
                    content=TokenContent(
                        token="test",
                        lemma="test",
                        pos="NOUN",
                        tag="NN",
                        shape="xxxx",
                        is_upper=False,
                        is_lower=True,
                        is_title=False,
                        is_punct=False,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=True,
                        is_stop=False,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
                Token(
                    span=CharSpan(doc_id=5678, char_begin=521, char_end=522),
                    content=TokenContent(
                        token=".",
                        lemma=".",
                        pos="PUNCT",
                        tag=".",
                        shape=".",
                        is_upper=False,
                        is_lower=False,
                        is_title=False,
                        is_punct=True,
                        is_right_punct=False,
                        is_left_punct=False,
                        is_ascii=True,
                        is_digit=False,
                        is_quote=False,
                        is_bracket=False,
                        is_currency=False,
                        is_alpha=False,
                        is_stop=False,
                        like_url=False,
                        like_num=False,
                        like_email=False,
                    ),
                ),
            ],
        ),
    ]


@pytest.fixture
def tokenizer_input_dataframe(tokenizer, tokenizer_test_cases):
    text_span_columns = tokenizer.inputs.text_span.column_names
    input_df = pd.DataFrame(
        [
            (
                test_case.text_doc_id,
                test_case.text_char_begin,
                test_case.text_char_begin + len(test_case.text),
                test_case.text,
            )
            for test_case in tokenizer_test_cases
        ],
        columns=[*text_span_columns, tokenizer.text_column],
    )
    return input_df


@pytest.fixture
def expected_tokenizer_output_dataframe(
    tokenizer: SpacyTokenizer, tokenizer_test_cases: list[TokenTestCase]
):
    columns = [tokenizer.text_column] + [
        column.name for column in tokenizer.outputs.order
    ]
    print(columns)
    expected_output_df = pd.DataFrame(
        [
            [
                test_case.text,
                test_case.text_doc_id,
                test_case.text_char_begin,
                test_case.text_char_begin + len(test_case.text),
                test_case.text_doc_id,
                token.span.char_begin,
                token.span.char_end,
            ]
            + list(dataclasses.astuple(token.content))
            + [None, tokenizer.algo_setup]
            for test_case in tokenizer_test_cases
            for token in test_case.tokens
        ],
        columns=columns,
    )
    return expected_output_df
