from dataclasses import dataclass

import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import ColumnSpec
from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import (
    AlgoInput,
    AlgoOutput,
    UDFAlgoBase,
)
from exasol.ai.text.impl.nlp.algorithms.udf_algo_pipeline import (
    UDFAlgoPipeline,
    map_fields,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec


@dataclass(frozen=True)
class SampleInputAlgo1(AlgoInput):
    item1: ColumnSpec = varchar_column_spec("C1")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1]


@dataclass(frozen=True)
class SampleOutputAlgo1(AlgoOutput):
    item1: ColumnSpec = varchar_column_spec("C2")
    item2: ColumnSpec = varchar_column_spec("C3")
    item3: ColumnSpec = varchar_column_spec("C4")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1, self.item2, self.item3, self.error_message, self.setup]


@dataclass(frozen=True)
class SampleInputAlgo2(AlgoInput):
    item1: ColumnSpec = varchar_column_spec("C0")
    item2: ColumnSpec = varchar_column_spec("C2")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1, self.item2]


@dataclass(frozen=True)
class SampleOutputAlgo2(AlgoOutput):
    item1: ColumnSpec = varchar_column_spec("C5")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1, self.error_message, self.setup]


@dataclass(frozen=True)
class SampleInputPipeline(AlgoInput):
    item1: ColumnSpec = varchar_column_spec("C0")
    item2: ColumnSpec = varchar_column_spec("C1")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1, self.item2]


@dataclass(frozen=True)
class SampleOutputPipeline(AlgoOutput):
    item1: ColumnSpec = varchar_column_spec("C3")
    item2: ColumnSpec = varchar_column_spec("C5")

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.item1, self.item2, self.error_message, self.setup]


class SampleAlgo1(UDFAlgoBase):
    inputs: SampleInputAlgo1 = SampleInputAlgo1()
    outputs: SampleOutputAlgo1 = SampleOutputAlgo1()

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        return batch_df


class SampleAlgo2(UDFAlgoBase):
    inputs: SampleInputAlgo2 = SampleInputAlgo2()
    outputs: SampleOutputAlgo2 = SampleOutputAlgo2()

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        return batch_df


class SamplePipelineAlgo(UDFAlgoPipeline):
    inputs: SampleInputPipeline = SampleInputPipeline()
    outputs: SampleOutputPipeline = SampleOutputPipeline()


class CompleteAlgoPipeline(SamplePipelineAlgo):
    algorithms: tuple[SampleAlgo1, SampleAlgo2]


class MissingAlgo1Pipeline(SamplePipelineAlgo):
    algorithms: tuple[SampleAlgo2]


class MissingAlgo2Pipeline(SamplePipelineAlgo):
    algorithms: tuple[SampleAlgo1]


def test_map_fields():
    data = SampleOutputAlgo1()
    alt_item1 = varchar_column_spec("xxx")
    alt_item3 = varchar_column_spec("zzz")
    expected_data = SampleOutputAlgo1(item1=alt_item1, item3=alt_item3)
    output_data = map_fields(data, {"item1": alt_item1, "item3": alt_item3})
    assert output_data == expected_data


def test_pipeline():
    """
    This test tests the column compatibility validation. The check is made that all
    algorithms get input columns listed in their input specifications, and the
    output of the last algorithm has all the columns listed in the pipline output
    specification. In this test the validation should succeed.
    """
    CompleteAlgoPipeline(algorithms=(SampleAlgo1(), SampleAlgo2()))
    assert True


@pytest.mark.parametrize(
    ["pipeline_type", "algo_type"],
    [(MissingAlgo1Pipeline, SampleAlgo2), (MissingAlgo2Pipeline, SampleAlgo1)],
    ids=["missing algo input", "missing pipeline output"],
)
def test_pipeline_error(pipeline_type, algo_type):
    """
    The first test tests that the column compatibility validation fails if an algorithm
    doesn't get some input columns listed in its input specification. The second test
    tests that the validation fails if the output of the last algorithm doesn't have
    some columns listed in the pipeline's output specification.
    """
    with pytest.raises(ValueError):
        pipeline_type(algorithms=(algo_type(),))
