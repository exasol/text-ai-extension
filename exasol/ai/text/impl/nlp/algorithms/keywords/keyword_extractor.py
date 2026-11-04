from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.pandas_utils import (
    generate_random_name,
    merge_result_with_input,
)
from exasol.ai.text.impl.utils.sql_utils import (
    double_column_spec,
    varchar_column_spec,
)

_default_keyword = varchar_column_spec("KEYWORD", flags=IOColumnFlags.Normalisation)


@dataclass(frozen=True)
class KeywordOutput(TextOutput):

    name: str = "KEYWORD"

    keyword: ColumnSpec = _default_keyword
    """
    The keyword/key-phrase or NULL in case of an error.
    """

    keyword_score: ColumnSpec = double_column_spec(f"{_default_keyword.name}_SCORE")
    """
    The keyword score, higher is better, or NULL in case of an error.
    """


KeywordExtractorOutput = list[list[tuple[str, float]]]


class KeywordExtractor(UDFTextBase):
    """
    The base class for keyword extraction algorithms.

    A keyword extraction algorithm makes no attempt to create a coherent list of top n
    keywords in the scope of the entire document. Instead, it outputs the top n keywords
    individually for each of the provided pieces of text. Therefore, if the keywords get
    aggregated for a document the number of unique keywords can exceed the specified
    limit n.
    """

    outputs: KeywordOutput = KeywordOutput()

    @property
    def keyword_column(self) -> str:
        return self.outputs.keyword.name

    @property
    def score_column(self) -> str:
        return self.outputs.keyword_score.name

    def _build_keyword_dataframe(
        self, batch_keywords: KeywordExtractorOutput, text_seq_column: str
    ) -> pd.DataFrame:
        """
        Packs the keywords returned by the extract_keywords function into a DataFrame
        with three columns: text sequential number, keyword text, keyword score.

        Parameters:
            batch_keywords:
                List of keywords. For the information on the data structure see the
                docstring of the `extract_keywords` function.
            text_seq_column:
                The name to give the column with the text sequential number.
        """
        return pd.DataFrame(
            [
                (text_seq, *keyword)
                for text_seq, keywords in enumerate(batch_keywords)
                for keyword in keywords
            ],
            columns=[text_seq_column, self.keyword_column, self.score_column],
        )

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:

        # Extract keywords for the whole batch.
        batch_keywords = self.extract_keywords(exa, batch_df[self.text_column])

        # Create temporarily a column to be used for joining keywords with the input.
        added_columns = [
            self.keyword_column,
            self.score_column,
            self.error_message_column,
        ]
        text_seq_column = generate_random_name([*batch_df.columns, *added_columns])

        # Convert the result to a DataFrame and merge it with the input.
        keyword_df = self._build_keyword_dataframe(batch_keywords, text_seq_column)
        output_df = merge_result_with_input(
            input_df=batch_df, result_df=keyword_df, merge_on_column=text_seq_column
        )
        output_df[self.error_message_column] = None
        return super().run_impl(output_df, exa)

    def extract_keywords(self, exa, texts: Iterable[str]) -> KeywordExtractorOutput:
        """
        For each of the provided text chunks returns a list of 2-element tuples -
        the keyword and its score (0..1). The keywords shall be returned in a list of
        the same size as in list of input texts. Each element of the list is itself a
        list of keywords for the correspondent input text.

        This method should be implemented by the subclasses.
        """
        return [[] for _ in texts]
