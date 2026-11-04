from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import Field

from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.sql_utils import integer_column_spec


@dataclass(frozen=True)
class WordCounterOutput(TextOutput):
    word_count: ColumnSpec = integer_column_spec("WORD_COUNT")
    """
    The number of words in the text span.
    """


class DummyWordCounter(UDFTextBase):
    """
    This is a dummy algorithm for testing.
    It counts the number of words in a text in the simplest possible way.

    algorithm discriminator: dummy-word-counter
    """

    outputs: WordCounterOutput = WordCounterOutput()

    algorithm_name: Literal["dummy-word-counter"] = Field(
        "dummy-word-counter", repr=False
    )

    @property
    def word_count_column(self) -> str:
        return self.outputs.word_count.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        batch_df[self.word_count_column] = (
            batch_df[self.text_column].str.split().str.len()
        )
        batch_df[self.error_message_column] = None
        return super().run_impl(batch_df, exa)
