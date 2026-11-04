from __future__ import annotations

from typing import cast

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.execution.input import (
    SQLStageGraphExecutionInput,
)
from exasol.analytics.query_handler.graph.stage.sql.execution.query_handler import (
    SQLStageGraphExecutionQueryHandler,
)
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extraction import AbstractExtraction
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.default_graph_builder import (
    default_graph_builder,
)
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)


class EmptySQLStageInput(SQLStageInputOutput):
    pass


def _stage_graph(extraction: AbstractExtraction, language_alias: str) -> SQLStageGraph:
    """
    Convert the user-defined extraction into a stage graph by using the
    `default_graph_builder()`.
    """
    graph_builder = default_graph_builder()
    context = ExtractorContext(
        language_alias=language_alias,
        output=extraction.output,
        defaults=extraction.defaults,
    )
    return graph_builder.build(
        extractor=cast(Extractor, extraction.extractor),
        context=context,
        downstream_builder=graph_builder,
    )


class ExtractionQueryHandler(SQLStageGraphExecutionQueryHandler):
    """
    Main phase for a top-level extraction.
    """

    def __init__(
        self, parameter: SQLStageGraphExecutionInput, context: ScopeQueryHandlerContext
    ):
        super().__init__(parameter, context)

    @classmethod
    def create(
        cls,
        extraction: AbstractExtraction,
        context: ScopeQueryHandlerContext,
        language_alias: str,
    ) -> ExtractionQueryHandler:
        bfs_location = context.get_temporary_bucketfs_location().bucketfs_location()
        stage_graph = _stage_graph(extraction, language_alias)
        parameter = SQLStageGraphExecutionInput(
            input=EmptySQLStageInput(),
            result_bucketfs_location=bfs_location,
            sql_stage_graph=stage_graph,
        )
        return cls(parameter=parameter, context=context)
