from dataclasses import dataclass
from typing import cast
from unittest import mock

import pytest
from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.result import Finish
from exasol.analytics.schema import (
    SchemaName,
    TableLikeName,
    TableLikeNameImpl,
)

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.splitter_query_handler import (
    SplitterQueryHandler,
    copy_algo_io_data,
)


@dataclass(frozen=True)
class DummyIOData(AlgoIOData):
    dummy_field: str


@pytest.fixture
def datasource() -> TableLikeName:
    return TableLikeNameImpl("my_source", schema=SchemaName("my_schema"))


@pytest.fixture
def qh_context() -> ScopeQueryHandlerContext:
    return mock.Mock(spec=ScopeQueryHandlerContext)


def create_sql_stage_inputs(
    datasource: TableLikeName, input_is_list: bool, num_inputs: int = 1
) -> list[SQLStageInputOutput]:
    stage_input = [
        AlgoIOData(datasource=datasource, columns=TextOutput())
        for _ in range(num_inputs)
    ]
    if input_is_list:
        return [ListInputOutput(input_output_list=stage_input)]
    return stage_input


def test_copy_algo_io_data(datasource):
    text_span = SpanColumns.create("my_span")
    dummy = DummyIOData(
        datasource=datasource,
        columns=TextOutput(name="old_name", text_span=text_span),
        dummy_field="abc",
    )
    dummy_copy = copy_algo_io_data(dummy, new_name="new_name")
    assert isinstance(dummy_copy, DummyIOData)
    assert dummy_copy.datasource == datasource
    assert dummy_copy.dummy_field == "abc"
    assert isinstance(dummy_copy.columns, TextOutput)
    assert dummy_copy.columns.name == "new_name"
    assert dummy_copy.columns.text_span == text_span


@pytest.mark.parametrize("input_is_list", [True, False])
def test_splitter_query_handler(datasource, qh_context, input_is_list):
    new_names = {"alpha", "beta", "gamma"}
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=create_sql_stage_inputs(datasource, input_is_list),
        result_bucketfs_location=None,  # type: ignore
    )
    query_handler = SplitterQueryHandler(
        stage_input=stage_input, context=qh_context, new_names=new_names
    )
    result = query_handler.start()
    assert isinstance(result, Finish)
    assert isinstance(result.result, ListInputOutput)
    assert all(isinstance(item, AlgoIOData) for item in result.result.input_output_list)
    names = {
        cast(AlgoIOData, item).columns.name for item in result.result.input_output_list
    }
    assert names == new_names


@pytest.mark.parametrize("input_is_list", [True, False])
def test_splitter_multi_input_error(datasource, qh_context, input_is_list):
    new_names = {"alpha", "beta", "gamma"}
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=create_sql_stage_inputs(
            datasource, input_is_list, num_inputs=2
        ),
        result_bucketfs_location=None,  # type: ignore
    )
    with pytest.raises(ValueError, match=".+expects single input.*"):
        SplitterQueryHandler(
            stage_input=stage_input, context=qh_context, new_names=new_names
        )


def test_splitter_wrong_input_error(datasource, qh_context):
    new_names = {"alpha", "beta", "gamma"}
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[SQLStageInputOutput()],
        result_bucketfs_location=None,  # type: ignore
    )
    with pytest.raises(ValueError, match=".+expects input object.+"):
        SplitterQueryHandler(
            stage_input=stage_input, context=qh_context, new_names=new_names
        )
