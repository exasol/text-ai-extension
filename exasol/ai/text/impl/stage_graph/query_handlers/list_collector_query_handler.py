from itertools import chain

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.result import Finish

from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.common import (
    ContinueOrFinish,
    IllegalMethodCallError,
)


class ListCollectorQueryHandler(SQLStageQueryHandler):
    """
    Query handler for :class:``ListCollectorSqlStage`` expecting at least
    one input.
    Creates a plain list of `SQLStageInputOutput` objects found in the Query Handler
    inputs and stores it in the `ListInputOutput`. If some of these objects are
    themselves `ListInputOutput` their internal lists get flattened into a single
    list.

    For a use case please see :class:`BranchGraphBuilder`.
    """

    def __init__(
        self, parameter: SQLStageQueryHandlerInput, context: ScopeQueryHandlerContext
    ):
        super().__init__(parameter, context)
        self._parameter = parameter
        assert len(self._parameter.sql_stage_inputs) >= 1

    def start(self) -> ContinueOrFinish:
        stage_inputs = list(
            chain(
                *[
                    inp.input_output_list if isinstance(inp, ListInputOutput) else [inp]
                    for inp in self._parameter.sql_stage_inputs
                ]
            )
        )
        output = ListInputOutput(stage_inputs)
        return Finish(output)

    def handle_query_result(self, query_result: QueryResult) -> ContinueOrFinish:
        raise IllegalMethodCallError(
            "Method ListCollectorQueryHandler.handle_query_result()"
            " is not intended to be called."
        )


class ListCollectorSqlStage(SQLStage):
    """
    A ListCollectorSqlStage serves as end node for an SQLStageGraph
    representing a branch structure.  The created query handler outputs a list
    of all its inputs.

    For a usecase please see :class:`BranchGraphBuilder`.
    """

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        qh_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return ListCollectorQueryHandler(stage_input, qh_context)
