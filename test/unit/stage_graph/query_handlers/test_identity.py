from unittest.mock import Mock

import pytest
from exasol.analytics.query_handler.result import Finish

from exasol.ai.text.impl.stage_graph.query_handlers.common import IllegalMethodCallError
from exasol.ai.text.impl.stage_graph.query_handlers.identity_query_handler import (
    IdentityQueryHandler,
    IdentitySqlStage,
)


@pytest.mark.parametrize("sql_stage_inputs", ([], ["one", "two"]))
def test_illegal_number_of_stage_inputs(sql_stage_inputs):
    "IdentitySqlStage expects sql_stage_inputs lave length 1"
    input = Mock(sql_stage_inputs=sql_stage_inputs)
    context = Mock()
    with pytest.raises(AssertionError):
        IdentityQueryHandler(input, context)


def test_create_query_handler():
    input = Mock(sql_stage_inputs=["input"])
    context = Mock()
    actual = IdentitySqlStage().create_query_handler(input, context)
    assert isinstance(actual, IdentityQueryHandler)
    assert actual._parameter == input
    assert actual._query_handler_context == context


@pytest.fixture
def query_handler():
    input = Mock(sql_stage_inputs=["input"])
    context = Mock()
    return IdentityQueryHandler(input, context)


def test_query_handler_start(query_handler):
    assert query_handler.start() == Finish("input")


def test_query_handler_handle_query_result(query_handler):
    with pytest.raises(IllegalMethodCallError):
        query_handler.handle_query_result(Mock())
