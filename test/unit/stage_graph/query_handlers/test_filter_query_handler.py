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

from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.filter_query_handler import (
    FilterQueryHandler,
)


@dataclass(frozen=True)
class FooIO(SQLStageInputOutput):
    pass


@dataclass(frozen=True)
class BarIO(SQLStageInputOutput):
    pass


@pytest.fixture
def datasource() -> TableLikeName:
    return TableLikeNameImpl("my_source", schema=SchemaName("my_schema"))


@pytest.fixture
def qh_context() -> ScopeQueryHandlerContext:
    return mock.Mock(spec=ScopeQueryHandlerContext)


def test_filter_query_handler(datasource, qh_context):
    input_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    allowed_names = ["beta", "gamma", "epsilon"]
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[
                    AlgoIOData(datasource=datasource, columns=TextOutput(name=name))
                    for name in input_names
                ]
            )
        ],
        result_bucketfs_location=None,  # type: ignore
    )
    query_handler = FilterQueryHandler(
        stage_input=stage_input, context=qh_context, allowed_names=set(allowed_names)
    )
    result = query_handler.start()
    assert isinstance(result, Finish)
    assert isinstance(result.result, ListInputOutput)
    assert all(isinstance(item, AlgoIOData) for item in result.result.input_output_list)
    names = [
        cast(AlgoIOData, item).columns.name for item in result.result.input_output_list
    ]
    assert names == allowed_names


def test_filter_wrong_input_error(datasource, qh_context):
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(input_output_list=[FooIO(), FooIO(), BarIO()])
        ],
        result_bucketfs_location=None,  # type: ignore
    )
    with pytest.raises(ValueError, match=".+input objects.+"):
        FilterQueryHandler(
            stage_input=stage_input, context=qh_context, allowed_names={"anything"}
        )


def test_filter_empty_output(datasource, qh_context):
    input_names = ["alpha", "beta", "gamma"]
    allowed_names = ["delta", "epsilon"]
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[
                    AlgoIOData(datasource=datasource, columns=TextOutput(name=name))
                    for name in input_names
                ]
            )
        ],
        result_bucketfs_location=None,  # type: ignore
    )
    with pytest.raises(ValueError, match="No outputs.+"):
        FilterQueryHandler(
            stage_input=stage_input,
            context=qh_context,
            allowed_names=set(allowed_names),
        )
