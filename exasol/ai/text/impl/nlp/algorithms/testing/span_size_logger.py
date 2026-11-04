from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import Field

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import (
    AlgoInput,
    AlgoOutput,
    UDFAlgoBase,
)
from exasol.ai.text.impl.utils.pandas_utils import copy_span_columns
from exasol.ai.text.impl.utils.sql_utils import integer_column_spec

_default_text_span = SpanColumns.create("TEXT", is_key=True)


@dataclass(frozen=True)
class SpanSizeInput(AlgoInput):
    text_span: SpanColumns = _default_text_span
    """
    Input text span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return self.text_span.order


@dataclass(frozen=True)
class SpanSizeOutput(AlgoOutput):
    text_span: SpanColumns = _default_text_span
    """
    A copy of the input text span.
    """

    span_size: ColumnSpec = integer_column_spec("SPAN_SIZE")
    """
    The size of the text span provided in the input.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [*self.text_span.order, self.span_size, self.error_message, self.setup]


class SpanSizeLogger(UDFAlgoBase):
    """
    This algorithm outputs the size of the input text span. The algorithm inherits
    from the UDFAlgoBase rather than UDFTextBase, therefore doesn't include the text
    in the input and the error_message in the output. It sets none of the IOColumnFlags
    to any of the columns.

    algorithm discriminator: span-size-logger
    """

    inputs: SpanSizeInput = SpanSizeInput()
    outputs: SpanSizeOutput = SpanSizeOutput()

    algorithm_name: Literal["span-size-logger"] = Field("span-size-logger", repr=False)

    @property
    def span_prefix(self) -> str:
        # To be removed
        return "TEXT"

    @property
    def span_size_column(self) -> str:
        return self.outputs.span_size.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        _, begin_column, end_column = self.inputs.text_span.column_names
        batch_df[self.span_size_column] = batch_df[end_column] - batch_df[begin_column]
        batch_df[self.error_message_column] = None
        batch_df = copy_span_columns(
            batch_df, self.inputs.text_span, self.outputs.text_span
        )
        return super().run_impl(batch_df, exa)

    def set_empty(
        self, output_df: pd.DataFrame, error_message: str | None = None
    ) -> pd.DataFrame:

        output_df = copy_span_columns(
            output_df, self.inputs.text_span, self.outputs.text_span
        )
        output_df[self.span_size_column] = None
        return super().set_empty(output_df, error_message)
