from dataclasses import fields
from typing import cast

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


def copy_algo_io_data(data: AlgoIOData, new_name: str) -> AlgoIOData:
    """
    Create a copy of the provided AlgoIOData object with the new name of its
    `columns`. The type of the object and all the fields apart from the name
    are preserved.
    """
    data_dict = {
        fld.name: getattr(data, fld.name)
        for fld in fields(data)
        if fld.name != "columns"
    }
    column_dict = {
        fld.name: getattr(data.columns, fld.name)
        for fld in fields(data.columns)
        if fld.name != "name"
    }
    return type(data)(
        columns=type(data.columns)(name=new_name, **column_dict), **data_dict
    )


def _validate_stage_input(input_list: list[SQLStageInputOutput]):
    if len(input_list) != 1:
        raise ValueError(
            "The Splitter Query Handler expects single input. "
            f"Got {len(input_list)}."
        )
    input_obj = input_list[0]
    if not isinstance(input_obj, AlgoIOData):
        raise ValueError(
            "The Splitter Query Handler expects input object to be "
            f"{AlgoIOData.__name__} or a type derived from it. "
            f"Got the type {type(input_obj).__name__}."
        )


class SplitterQueryHandler(SQLStageQueryHandler):
    """
    Input splitter Query handler.

    Creates multiple copies of the object in the single input, assigning different
    names to them.
    The input is expected to be an output from a single algorithm, with the object
    being AlgoIOData or of a type derived from it. The object can be enclosed in the
    ListInputOutput.
    """

    def __init__(
        self,
        stage_input: SQLStageQueryHandlerInput,
        context: ScopeQueryHandlerContext,
        new_names: set[str],
    ):
        if len(stage_input.sql_stage_inputs) == 1 and isinstance(
            stage_input.sql_stage_inputs[0], ListInputOutput
        ):
            input_list = cast(
                ListInputOutput, stage_input.sql_stage_inputs[0]
            ).input_output_list
        else:
            input_list = stage_input.sql_stage_inputs
        _validate_stage_input(input_list)
        super().__init__(stage_input, context)

        input_obj = cast(AlgoIOData, input_list[0])
        self._outputs = [
            copy_algo_io_data(input_obj, new_name) for new_name in new_names
        ]

    def start(self) -> ContinueOrFinish:
        output = ListInputOutput(self._outputs)  # type: ignore
        return Finish(output)

    def handle_query_result(self, query_result: QueryResult) -> ContinueOrFinish:
        raise IllegalMethodCallError(
            "Method SplitterQueryHandler.handle_query_result()"
            " is not intended to be called."
        )


class SplitterSqlStage(SQLStage):

    def __init__(self, new_names: set[str]) -> None:
        self.new_names = new_names

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        qh_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        return SplitterQueryHandler(stage_input, qh_context, self.new_names)
