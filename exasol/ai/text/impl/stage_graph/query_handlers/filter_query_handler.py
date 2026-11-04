from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.result import Finish

from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.common import (
    ContinueOrFinish,
    IllegalMethodCallError,
)
from exasol.ai.text.impl.utils.stage_graph_utils import get_input_list


def _validate_stage_input(input_list: list[SQLStageInputOutput]):
    if not all(isinstance(inp, AlgoIOData) for inp in input_list):
        invalid_types = {
            type(inp).__name__ for inp in input_list if not isinstance(inp, AlgoIOData)
        }
        invalid_type_list = ", ".join(invalid_types)
        raise ValueError(
            "The Filter Query Handler expects all input objects to be "
            f"{AlgoIOData.__name__} or a type derived from it. Got the "
            f"invalid type(s) {invalid_type_list}."
        )


class FilterQueryHandler(SQLStageQueryHandler):
    """
    Input filter Query handler.

    Copies objects from the input to the output if their names are in the configured
    list of allowed names.
    The inputs are expected to be outputs from algorithms, with objects being AlgoIOData
    or of a type derived from it. The objects must be enclosed in the ListInputOutput.

    An exception is raised if the filtered list gets empty. Should an exception be
    raised if there is no input found for one or more allowed names is an open question.
    Currently, we don't do this extra check.
    """

    def __init__(
        self,
        stage_input: SQLStageQueryHandlerInput,
        context: ScopeQueryHandlerContext,
        allowed_names: set[str],
    ):
        input_list = get_input_list(stage_input)
        _validate_stage_input(input_list)
        super().__init__(stage_input, context)

        self._outputs = list(
            filter(
                lambda inp: isinstance(inp, AlgoIOData)
                and inp.columns.name in allowed_names,
                input_list,
            )
        )
        if not self._outputs:
            raise ValueError(f"No outputs with the names {allowed_names}.")

    def start(self) -> ContinueOrFinish:
        output = ListInputOutput(self._outputs)
        return Finish(output)

    def handle_query_result(self, query_result: QueryResult) -> ContinueOrFinish:
        raise IllegalMethodCallError(
            "Method FilterQueryHandler.handle_query_result()"
            " is not intended to be called."
        )


class FilterSqlStage(SQLStage):

    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        qh_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return FilterQueryHandler(stage_input, qh_context, self.allowed_names)
