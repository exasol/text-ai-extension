from typing import cast

from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.schema import (
    SchemaName,
    TableLikeName,
    TableLikeNameImpl,
)

from exasol.ai.text.extractors.base_extractor import OutputExtractor
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)

Edge = tuple[SQLStage, SQLStage]


def all_edges(graphs: list[SQLStageGraph]) -> set[Edge]:
    """
    Returns a set of all edges from all graphs passed as argument.
    """
    return {edge for g in graphs for edge in g.edges()}


def linear_interlinks(graphs: list[SQLStageGraph]) -> set[Edge]:
    """
    Links a chain of execution graphs. Returns a set of edges each running
    from the end node of each subgraph to the start node of its successor.

    For graphs [ g1, g2, g3 ] returns { e(g1 -> g2), e(g2 -> g3) }.
    """

    def edge(i: int, j: int) -> Edge:
        """Return the edge running from node i to j."""
        a = graphs[i].end_node
        b = graphs[j].start_node
        return a, b

    return {edge(i, i + 1) for i in range(len(graphs) - 1)}


def get_output_table(selector: Extractor, context: ExtractorContext) -> TableLikeName:
    """
    Returns the extractor's output table or view.

    The logic of this function is subject to change. For the moment, the name of
    the table or view is determined by the properties of the provided extractor. The
    extractor may contain the output db name provided by the user. If the name is not
    set explicitly, the extractor's name should be used instead.

    Unless the name is supplied explicitly, a prefix may be added to it. The prefix
    can be optionally defined in the context.
    """
    explicit_name = False
    extractor = cast(Extractor, selector)
    db_output: str | None = None
    if isinstance(extractor, OutputExtractor):
        db_output = extractor.db_output
        explicit_name = bool(db_output)
    table_name = db_output or extractor.name

    if (not explicit_name) and context.output.table_prefix:
        table_name = f"{context.output.table_prefix}_{table_name}"
    return TableLikeNameImpl(table_name, SchemaName(context.output.db_schema))


def get_input_list(stage_input: SQLStageQueryHandlerInput) -> list[SQLStageInputOutput]:
    """
    Validates that the provided graph stage input has exactly on input and its type
    is `ListInputOutput`.
    Returns the list of objects enclosed in the `ListInputOutput`
    """
    if len(stage_input.sql_stage_inputs) != 1 or (
        not isinstance(stage_input.sql_stage_inputs[0], ListInputOutput)
    ):
        input_types = [type(inp).__name__ for inp in stage_input.sql_stage_inputs]
        raise ValueError(
            f"Invalid stage graph input. Expected "
            f"[{ListInputOutput.__name__}]. Got {input_types}."
        )
    return cast(ListInputOutput, stage_input.sql_stage_inputs[0]).input_output_list
