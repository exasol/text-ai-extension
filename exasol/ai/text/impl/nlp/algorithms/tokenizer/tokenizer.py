import dataclasses
from abc import abstractmethod
from dataclasses import (
    dataclass,
    fields,
)

import pandas as pd

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.char_span import CharSpan
from exasol.ai.text.impl.utils.pandas_utils import (
    adjust_result_span,
    generate_random_name,
    merge_result_with_input,
)
from exasol.ai.text.impl.utils.sql_utils import (
    boolean_column_spec,
    varchar_column_spec,
)

_default_token = varchar_column_spec("TOKEN")


@dataclass(frozen=True)
class TokenizerOutput(TextOutput):
    name: str = "TOKEN"

    """
    The output columns of a tokenizer algorithm.
    """

    token_span: SpanColumns = SpanColumns.create(_default_token.name)
    """
    The character span of the token. The span is relative to the source document, not
    to the text chunk the was tokenized.
    """

    token: ColumnSpec = _default_token
    """
    The text identified by the token span, or NULL in case of an
    error.
    """

    lemma: ColumnSpec = varchar_column_spec("LEMMA", flags=IOColumnFlags.Normalisation)
    """
    The lemmatized text identified by the token span, or NULL in case of an
    error or if the tokenizer doesn't support this feature.
    """
    pos: ColumnSpec = varchar_column_spec("POS", flags=IOColumnFlags.Normalisation)
    """
    The coarse-grained part-of-speach from https://universaldependencies.org/u/pos/ of the text identified
    by the token span, or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    tag: ColumnSpec = varchar_column_spec("TAG", flags=IOColumnFlags.Normalisation)
    """
    The fine-grained part-of-speach from https://universaldependencies.org/u/pos/ of the text identified
    by the token span, or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    shape: ColumnSpec = varchar_column_spec("SHAPE")
    """
    The shape of the text identified by the token span, or NULL in case of an
    error or if the tokenizer doesn't support this feature.
    """

    is_alpha: ColumnSpec = boolean_column_spec("IS_ALPHA")
    """
    Indicates if the text identified by the token span consists out of alphabetic characters,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_stop: ColumnSpec = boolean_column_spec("IS_STOP")
    """
    Indicates if the text identified by the token span is a stopword,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_ascii: ColumnSpec = boolean_column_spec("IS_ASCII")
    """
    Indicates if the text identified by the token span consists out of ASCII characters,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_digit: ColumnSpec = boolean_column_spec("IS_DIGIT")
    """
    Indicates if the text identified by the token span consists out of digits,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_lower: ColumnSpec = boolean_column_spec("IS_LOWER")
    """
    Indicates if the text identified by the token span is lowercase,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_upper: ColumnSpec = boolean_column_spec("IS_UPPER")
    """
    Indicates if the text identified by the token span is uppercase,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_title: ColumnSpec = boolean_column_spec("IS_TITLE")
    """
    Indicates if the text identified by the token span is titlecase,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_punct: ColumnSpec = boolean_column_spec("IS_PUNCT")
    """
    Indicates if the text identified by the token span is punctuation,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_left_punct: ColumnSpec = boolean_column_spec("IS_LEFT_PUNCT")
    """
    Indicates if the text identified by the token span is left punctuation, e.g. "(",
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_right_punct: ColumnSpec = boolean_column_spec("IS_RIGHT_PUNCT")
    """
    Indicates if the text identified by the token span is right punctuation, e.g. ")",
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_bracket: ColumnSpec = boolean_column_spec("IS_BRACKET")
    """
    Indicates if the text identified by the token span is a bracket,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_quote: ColumnSpec = boolean_column_spec("IS_QUOTE")
    """
    Indicates if the text identified by the token span is a quotation mark,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    is_currency: ColumnSpec = boolean_column_spec("IS_CURRENCY")
    """
    Indicates if the text identified by the token span is a currency symbol,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    like_url: ColumnSpec = boolean_column_spec("LIKE_URL")
    """
    Indicates if the text identified by the token span resemble a URL,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    like_num: ColumnSpec = boolean_column_spec("LIKE_NUM")
    """
    Indicates if the text identified by the token span represent a number, e.g. “10.9”, “10”, “ten”, etc.
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """

    like_email: ColumnSpec = boolean_column_spec("LIKE_EMAIL")
    """
    Indicates if the text identified by the token span resemble an email address,
    or NULL in case of an error or if the tokenizer doesn't support this feature.
    """


@dataclass(frozen=True)
class TokenContent:
    """
    Data class that defines the content the can be extracted for a text token by a tokenizer.
    """

    token: str
    lemma: str | None = None
    pos: str | None = None
    tag: str | None = None
    shape: str | None = None
    is_alpha: bool | None = None
    is_stop: bool | None = None
    is_ascii: bool | None = None
    is_digit: bool | None = None
    is_lower: bool | None = None
    is_upper: bool | None = None
    is_title: bool | None = None
    is_punct: bool | None = None
    is_left_punct: bool | None = None
    is_right_punct: bool | None = None
    is_bracket: bool | None = None
    is_quote: bool | None = None
    is_currency: bool | None = None
    like_url: bool | None = None
    like_num: bool | None = None
    like_email: bool | None = None


@dataclass(frozen=True)
class Token:
    """
    Data class that represents a token extracted by a  tokenizer.
    """

    span: CharSpan
    content: TokenContent


class Tokenizer(UDFTextBase):
    """
    The base class for tokenizer algorithms.
    """

    outputs: TokenizerOutput = TokenizerOutput()

    @property
    def token_column(self) -> str:
        return self.outputs.token.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        tokenized_texts = self.tokenize_texts(batch_df[self.text_column])
        text_seq_column = generate_random_name(
            [*batch_df.columns, *self.token_column_names, self.error_message_column]
        )
        token_df = self._tokenized_text_to_dataframe(tokenized_texts, text_seq_column)
        output_df = merge_result_with_input(
            input_df=batch_df, result_df=token_df, merge_on_column=text_seq_column
        )
        output_df[self.error_message_column] = None
        output_df = adjust_result_span(
            output_df=output_df,
            text_span=self.inputs.text_span,
            result_span=self.outputs.token_span,
        )
        return super().run_impl(output_df, exa)

    def _tokenized_text_to_dataframe(
        self, tokenized_texts: list[list[Token]], text_seq_column: str
    ) -> pd.DataFrame:
        token_df = pd.DataFrame(
            [
                [index]
                + list(dataclasses.astuple(token.span))
                + list(dataclasses.astuple(token.content))
                for index, tokens in enumerate(tokenized_texts)
                for token in tokens
            ],
            columns=[text_seq_column] + self.token_column_names,
        )
        return token_df

    @abstractmethod
    def tokenize_texts(self, texts: pd.Series) -> list[list[Token]]:
        """
        Args:
            texts: A series of text

        Returns:
            For each string in the input a list of tokens in this string with a char span
            with relative positions to the beginning string and arbitrary doc id.
            For example, if the string is "An test", then the char span for the first token would be,
            CharSpan(doc_id=-1, begin=0, begin=2).
        """
        pass

    @property
    def token_column_names(self) -> list[str]:
        token_content_names = [field.name.upper() for field in fields(TokenContent)]
        return self.outputs.token_span.column_names + token_content_names
