from unittest.mock import Mock

import pytest
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.result import Finish

from exasol.ai.text.impl.stage_graph.query_handlers.common import IllegalMethodCallError
from exasol.ai.text.impl.stage_graph.query_handlers.list_collector_query_handler import (
    ListCollectorQueryHandler,
    ListCollectorSqlStage,
    ListInputOutput,
)


def test_empty_input():
    input = Mock(sql_stage_inputs=[])
    context = Mock()
    with pytest.raises(AssertionError):
        ListCollectorQueryHandler(input, context)


@pytest.mark.parametrize("len", [1, 2])
def test_create_query_handler(len):
    input = Mock(sql_stage_inputs=["input"])
    context = Mock()
    actual = ListCollectorSqlStage().create_query_handler(input, context)
    assert isinstance(actual, ListCollectorQueryHandler)
    assert actual._parameter == input
    assert actual._query_handler_context == context


def query_handler(sql_stage_inputs=["input"], context=Mock()):
    input = Mock(sql_stage_inputs=sql_stage_inputs)
    context = Mock()
    return ListCollectorQueryHandler(input, context)


def test_query_handler_start():
    sql_stage_inputs = ["input"]
    testee = query_handler(sql_stage_inputs)
    assert testee.start() == Finish(ListInputOutput(sql_stage_inputs))


def test_query_handler_handle_query_result():
    with pytest.raises(IllegalMethodCallError):
        query_handler().handle_query_result(Mock())


def test_list_in_input():
    stage_inputs = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            SQLStageInputOutput(),
            ListInputOutput(
                input_output_list=[SQLStageInputOutput() for _ in range(3)]
            ),
            SQLStageInputOutput(),
        ],
        result_bucketfs_location=None,  # type: ignore
    )
    context = Mock()
    qh = ListCollectorQueryHandler(stage_inputs, context)
    result = qh.start().result
    assert isinstance(result, ListInputOutput)
    assert len(result.input_output_list) == 5
