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
class BatchSizeOutput(TextOutput):
    batch_size: ColumnSpec = integer_column_spec("BATCH_SIZE")
    """
    The number of rows in the input batch.
    """


class BatchSizeLogger(UDFTextBase):
    """
    This algorithm outputs the batch size.

    algorithm discriminator: batch-size-logger
    """

    outputs: BatchSizeOutput = BatchSizeOutput()

    algorithm_name: Literal["batch-size-logger"] = Field(
        "batch-size-logger", repr=False
    )

    @property
    def batch_size_column(self) -> str:
        return self.outputs.batch_size.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        batch_df[self.batch_size_column] = len(batch_df)
        batch_df[self.error_message_column] = None
        return super().run_impl(batch_df, exa)
