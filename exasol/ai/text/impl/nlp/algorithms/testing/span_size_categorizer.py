from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
)

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import (
    AlgoInput,
    AlgoOutput,
    UDFAlgoBase,
)
from exasol.ai.text.impl.utils.pandas_utils import copy_span_columns
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec

_default_text_span = SpanColumns.create(
    "TEXT", is_key=True, flags=IOColumnFlags.Distribution | IOColumnFlags.Partitioning
)


class SizeCategory(BaseModel):
    min_size: int
    category: str | None


@dataclass(frozen=True)
class SizeCategoryInput(AlgoInput):
    text_span: SpanColumns = _default_text_span
    """
    Input text span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return self.text_span.order


@dataclass(frozen=True)
class SizeCategoryOutput(AlgoOutput):
    text_span: SpanColumns = _default_text_span
    """
    A copy of the input text span.
    """

    size_category: ColumnSpec = varchar_column_spec(
        "SIZE_CATEGORY", flags=IOColumnFlags.Normalisation
    )
    """
    The category of the text span size. The output data table should be
    normalised over this column.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [
            *self.text_span.order,
            self.size_category,
            self.error_message,
            self.setup,
        ]


class SpanSizeCategorizer(UDFAlgoBase):
    """
    This algorithm outputs the size of the input text span in a form of a category,
    e.g. small, medium, large. The algorithm inherits from the UDFAlgoBase rather than
    UDFTextBase, therefore doesn't include the text in the input and the error_message
    in the output.

    algorithm discriminator: span-size-categorizer

    Parameters:
        categories:
            The list of defined size categories - (minimum size, category name). It
            would make sense to set the minimum size for the first category to zero.
    """

    inputs: SizeCategoryInput = SizeCategoryInput()
    outputs: SizeCategoryOutput = SizeCategoryOutput()

    algorithm_name: Literal["span-size-categorizer"] = Field(
        "span-size-categorizer", repr=False
    )
    categories: list[SizeCategory] = [
        SizeCategory(min_size=0, category="small"),
        SizeCategory(min_size=100, category="medium"),
        SizeCategory(min_size=500, category="large"),
    ]

    @property
    def span_prefix(self) -> str:
        # To be removed
        return "TEXT"

    @property
    def size_category_column(self) -> str:
        return self.outputs.size_category.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        _, begin_column, end_column = self.inputs.text_span.column_names
        span_size = batch_df[end_column] - batch_df[begin_column]
        for cat in sorted(self.categories, key=lambda c: c.min_size):
            batch_df.loc[span_size >= cat.min_size, self.size_category_column] = (
                cat.category
            )
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
        output_df[self.size_category_column] = None
        return super().set_empty(output_df, error_message)
