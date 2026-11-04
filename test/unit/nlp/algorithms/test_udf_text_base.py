from dataclasses import dataclass

import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextInput,
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.sql_utils import (
    integer_column_spec,
    varchar_column_spec,
)

_default_phrase = varchar_column_spec("PHRASE")


def _get_input_df(inputs: TextInput) -> pd.DataFrame:
    return pd.DataFrame(
        [(123, 100, 118, "aaa"), (456, 200, 212, "bbb"), (789, 300, 313, "ccc")],
        columns=[*inputs.text_span.column_names, inputs.text.name],
    )


@dataclass(frozen=True)
class SampleOutput(TextOutput):
    varchar_out: ColumnSpec = varchar_column_spec("V_OUT")
    integer_out: ColumnSpec = integer_column_spec("I_OUT")


class SampleExtractor(UDFTextBase):
    outputs: SampleOutput = SampleOutput()

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        batch_df[self.outputs.varchar_out.name] = "X"
        batch_df[self.outputs.integer_out.name] = 5
        batch_df[self.outputs.error_message.name] = None
        return super().run_impl(batch_df, exa)


def test_output():
    algo = SampleExtractor()
    assert algo.outputs.order == [
        *algo.outputs.text_span.order,
        algo.outputs.varchar_out,
        algo.outputs.integer_out,
        algo.outputs.error_message,
        algo.outputs.setup,
    ]


@pytest.mark.parametrize(
    ["in_span_name", "out_span_name"],
    [("text", "text"), ("text_in", "text_out")],
    ids=["same span name", "different span name"],
)
def test_run_impl(empty_exa, in_span_name, out_span_name):
    algo = SampleExtractor(
        inputs=TextInput(text_span=SpanColumns.create(in_span_name)),
        outputs=SampleOutput(text_span=SpanColumns.create(out_span_name)),
    )
    input_df = _get_input_df(algo.inputs)
    output_df = algo.run_impl(input_df, empty_exa)
    expected_df = input_df.copy()
    expected_df[algo.outputs.text_span.column_names] = input_df[
        algo.inputs.text_span.column_names
    ]
    expected_df[algo.outputs.varchar_out.name] = "X"
    expected_df[algo.outputs.integer_out.name] = 5
    expected_df[algo.error_message_column] = None
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    ["in_span_name", "out_span_name"],
    [("text", "text"), ("text_in", "text_out")],
    ids=["same span name", "different span name"],
)
def test_set_empty(empty_exa, in_span_name, out_span_name):
    algo = SampleExtractor(
        inputs=TextInput(text_span=SpanColumns.create(in_span_name)),
        outputs=SampleOutput(text_span=SpanColumns.create(out_span_name)),
    )
    input_df = _get_input_df(algo.inputs)
    output_df = algo.set_empty(input_df, "the error")
    expected_df = input_df.copy()
    expected_df[algo.outputs.text_span.column_names] = input_df[
        algo.inputs.text_span.column_names
    ]
    expected_df[algo.outputs.varchar_out.name] = None
    expected_df[algo.outputs.integer_out.name] = None
    expected_df[algo.error_message_column] = "the error"
    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)
