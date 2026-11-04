from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextInput,
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.pandas_utils import (
    adjust_result_span,
    generate_random_name,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec

_default_phrase = varchar_column_spec("PHRASE")


@dataclass(frozen=True)
class TextSearchInput(TextInput):

    phrase: ColumnSpec = _default_phrase
    """
    The phrase to search for.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [*self.text_span.order, self.text, self.phrase]


@dataclass(frozen=True)
class TextSearchOutput(TextOutput):

    phrase: ColumnSpec = _default_phrase
    """
    The searched phrase.
    """

    phrase_span: SpanColumns = SpanColumns.create(_default_phrase.name)
    """
    The character span of the phrase occurrence. The span is relative to the source
    document, not to the text chunk the phrase was searched in.
    """


PhraseLocations = list[tuple[int, int, int]]


class TextSearch(UDFTextBase):
    name: str = "TEXT_SEARCH"

    """
    The base class for text search algorithms.
    """
    inputs: TextSearchInput = TextSearchInput()
    outputs: TextSearchOutput = TextSearchOutput()

    @property
    def phrase_input_column(self) -> str:
        return self.inputs.phrase.name

    @property
    def phrase_output_column(self) -> str:
        return self.outputs.phrase.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:

        text_span_columns = self.inputs.text_span.column_names
        text_id_column, text_begin_column, _ = text_span_columns
        phrase_span_columns = self.outputs.phrase_span.column_names
        phrase_id_column, phrase_begin_column, phrase_end_column = phrase_span_columns

        # Make sure the input columns do not include those to be created.
        added_columns = [*phrase_span_columns, self.error_message_column]
        batch_df = batch_df[
            [col for col in batch_df.columns if col not in added_columns]
        ]

        # Get a unique column name for the sequential number of the searched phrase.
        # This column is needed for merging the search results into the `batch_df`.
        phrase_seq_column = generate_random_name([*batch_df.columns, *added_columns])

        def group_func(group: pd.DataFrame) -> pd.DataFrame:
            """
            Builds a DataFrame of phrase locations for a single document span.
            Each row of the data frame ``group`` contains the same document span and
            a phrase to be located within this span.
            For each location of the phrase the result will include a copy of the input
            columns and the added columns (the phrase span in the document), as per the
            algorith specification.
            """

            # The text should be the same in all rows of the group
            text = group.iloc[0][self.text_column]
            phrases = group[self.phrase_input_column]
            phrases = phrases[phrases.astype(bool)]
            if phrases.empty:
                locations: PhraseLocations = []
            else:
                locations = self.locate_phrases(text, phrases)
            location_df = pd.DataFrame(
                locations,
                columns=[phrase_seq_column, phrase_begin_column, phrase_end_column],
                dtype=int,
            )
            # Merge locations with the group rows
            group_copy = group.copy()
            group_copy[phrase_seq_column] = range(len(group))
            return pd.merge(group_copy, location_df, on=phrase_seq_column, how="left")

        output_df = batch_df.groupby(list(text_span_columns)).apply(group_func)
        output_df[self.error_message_column] = None
        output_df.drop(phrase_seq_column, axis=1, inplace=True)
        output_df = adjust_result_span(
            output_df=output_df,
            text_span=self.inputs.text_span,
            result_span=self.outputs.phrase_span,
        )
        return super().run_impl(output_df, exa)

    def locate_phrases(self, text: str, phrases: Iterable[str]) -> PhraseLocations:
        """
        Executes the text search. This method should be implemented in a subclass.

        Parameters:
            text:
                Text to search the phrase in.
            phrases:
                A list or any other Iterable of the phrases to search for.

        Returns a list of findings, each described by a tuple:
            - Sequential number of the phrase in the input.
            - Starting index of the phrase occurrence in the text.
            - Ending index of the phrase occurrence in the text.
        """
        return []
