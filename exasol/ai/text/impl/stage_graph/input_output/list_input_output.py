from dataclasses import dataclass

from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)


@dataclass(frozen=True)
class ListInputOutput(SQLStageInputOutput):
    input_output_list: list[SQLStageInputOutput]
