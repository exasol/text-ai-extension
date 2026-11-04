from typing import cast

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.schema import TableLikeName

from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_query_handler import (
    UDFAlgoQueryHandler,
)


class UDFAlgoSQLStage(SQLStage):
    """
    The factory of an algorithm running query handler.
    """

    def __init__(
        self,
        udf_runner: UDFRunner,
        output_table: TableLikeName,
        language_alias: str,
        parallelism_per_node: int,
        log_span: LogSpan,
    ) -> None:
        self.log_span = log_span
        self.udf_runner = udf_runner
        self.output_table = output_table
        self.language_alias = language_alias
        self.parallelism_per_node = parallelism_per_node

    def create_query_handler(
        self,
        stage_input: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
    ) -> SQLStageQueryHandler:
        """
        Verifies that the provided input is of a suitable type and
        create an instance of the query handler.
        For information on the required input see the docstring of the
        UDFAlgoQueryHandler class.
        """
        # Type check the input parameter.
        if (num_inputs := len(stage_input.sql_stage_inputs)) != 1:
            raise RuntimeError(
                "An algorithm query handler expects exactly one input, "
                f"but received {num_inputs}."
            )
        if not isinstance(stage_input.sql_stage_inputs[0], AlgoIOData):
            raise RuntimeError(
                f"An algorithm query handler expects {AlgoIOData.__name__} "
                f"as an input but got {type(stage_input.sql_stage_inputs[0]).__name__}."
            )
        data_in = cast(AlgoIOData, stage_input.sql_stage_inputs[0])
        expected_input_type = type(self.udf_runner.algorithm.inputs)
        if not isinstance(data_in.columns, expected_input_type):
            raise RuntimeError(
                "The algorithm query handler expects the inputs to be "
                f"{expected_input_type.__name__}. Got {type(data_in.columns).__name__}."
            )

        return UDFAlgoQueryHandler(
            udf_runner=self.udf_runner,
            input_table=data_in.primary_view,
            output_table=self.output_table,
            language_alias=self.language_alias,
            parameter=stage_input,
            query_handler_context=query_handler_context,
            parallelism_per_node=self.parallelism_per_node,
            log_span=self.log_span,
        )
