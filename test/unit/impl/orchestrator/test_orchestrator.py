from typing import Any
from unittest.mock import (
    Mock,
    create_autospec,
    patch,
)

import pytest
from exasol.analytics.query_handler.query_handler import QueryHandler
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)

from exasol.ai.text.impl.orchestrator.extraction_query_handler import (
    ExtractionQueryHandler,
)
from exasol.ai.text.impl.orchestrator.orchestrator_query_handler import (
    OrchestratorQueryHandler,
)


def mock_query_handler(continue_count: int, result: Any) -> QueryHandler:
    mock = create_autospec(QueryHandler)
    if continue_count < 1:
        mock.start.return_value = Finish(result)
        return mock
    mock.start.return_value = Mock(Continue)
    mock.handle_query_result.side_effect = [
        Mock(Continue) for _ in range(0, continue_count - 1)
    ] + [Finish(result)]
    return mock


MODULE = "exasol.ai.text.impl.orchestrator.orchestrator_query_handler"


@pytest.mark.parametrize("continue_counts", range(0, 3))
def test_orchestrator_calls_extraction(continue_counts):
    """
    Verify the actual extraction query handler has been called.

    Parameter `continue_counts` specifies the number of times the query
    handlers for the extraction phase, should be simulated to return
    Continue. Each continue indicates the need to call handle_query_result()
    once again. If continue_count for a query handler is 0, then method
    handle_query_result() is expected not to be called at all.
    """
    calls_to_handle_query_result = continue_counts
    extraction_result = "perfect extraction"
    extraction = mock_query_handler(
        continue_count=continue_counts,
        result=extraction_result,
    )
    testee = OrchestratorQueryHandler(parameter=Mock(), context=Mock())
    with patch.object(ExtractionQueryHandler, attribute="create") as mock:
        mock.return_value = extraction
        action = testee.start()
        for i in range(0, calls_to_handle_query_result):
            assert isinstance(action, Continue)
            action = testee.handle_query_result(query_result=Mock())
        assert isinstance(action, Finish)
        assert action.result == extraction_result

    assert extraction.start.call_count == 1
    assert extraction.handle_query_result.call_count == continue_counts
