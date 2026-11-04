from dataclasses import (
    dataclass,
    fields,
)

import pandas as pd

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    OrderedColumns,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import (
    AlgoInput,
    AlgoOutput,
    UDFAlgoBase,
)
from exasol.ai.text.impl.utils.pandas_utils import copy_span_columns
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec

_default_text = varchar_column_spec("TEXT")
_default_text_span = SpanColumns.create(
    _default_text.name,
    is_key=True,
    flags=IOColumnFlags.Distribution | IOColumnFlags.Partitioning,
)


@dataclass(frozen=True)
class TextInput(AlgoInput):
    """
    Base input columns for an NLP algorithm. This should work for most text algorithms.
    """

    name: str = "TEXT"

    text_span: SpanColumns = _default_text_span
    """
    Input text span.
    """

    text: ColumnSpec = _default_text
    """
    Input text identified by the span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [*self.text_span.order, self.text]


@dataclass(frozen=True)
class TextOutput(AlgoOutput):
    """
    Base output columns of an NLP algorithm. The output will contain a reference
    (a span) of the text the algorithm was processing, the actual output of the
    algorithm (to be defined by derived classes) and the error message.

    By default, the columns are ordered as following: The text span goes first,
    then all columns specific to an algorithm in the order they are defined in a
    dataclass, and finally the error message.
    """

    name: str = "TEXT"

    text_span: SpanColumns = _default_text_span
    """
    A copy of the input text span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        base_fields = set(fields(TextOutput))

        def defined_columns(dc) -> list[ColumnSpec]:
            columns: list[ColumnSpec] = []
            for fld in fields(dc):
                if fld not in base_fields:
                    obj = getattr(dc, fld.name)
                    if isinstance(obj, OrderedColumns):
                        columns.extend(obj.order)
                    elif isinstance(obj, ColumnSpec):
                        columns.append(obj)
            return columns

        return [
            *self.text_span.order,
            *defined_columns(self),
            self.error_message,
            self.setup,
        ]


# noinspection PyAbstractClass
class UDFTextBase(UDFAlgoBase):
    """
    The base class for a UDF enabled NLP algorithm.
    """

    inputs: TextInput = TextInput()
    outputs: TextOutput = TextOutput()

    @property
    def text_column(self) -> str:
        return self.inputs.text.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        # The output of an NLP algorithm should contain a copy of the input text span.
        # In theory, the column names of the two spans can be different, hence we should
        # copy or rename them explicitly.
        batch_df = copy_span_columns(
            batch_df, self.inputs.text_span, self.outputs.text_span
        )
        return super().run_impl(batch_df, exa)

    def set_empty(
        self, output_df: pd.DataFrame, error_message: str | None = None
    ) -> pd.DataFrame:
        # The default implementation of this function sets to None all fields except
        # the text span copied from the input.
        output_df = copy_span_columns(
            output_df, self.inputs.text_span, self.outputs.text_span
        )
        nullable_columns = [
            name
            for name in self.outputs.column_names
            if name not in self.outputs.text_span.column_names
        ]
        output_df[nullable_columns] = None
        return super().set_empty(output_df, error_message)
