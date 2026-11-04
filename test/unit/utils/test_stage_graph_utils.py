import re
from typing import cast

import pytest
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage import SQLStage
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)

from exasol.ai.text.extractors import (
    KeywordExtractor,
    PatternRankKeywordSettings,
    SourceTableExtractor,
)
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.utils.stage_graph_utils import (
    all_edges,
    get_input_list,
    get_output_table,
    linear_interlinks,
)
from test.unit.sql_stage_mock import (
    SqlStageMock,
    sql_stage_graph,
)
from test.utils.helpers import source_tables


def test_all_edges():
    def graph(extra_nodes=0) -> SQLStageGraph:
        """
        Return an SQLStageGraph, with start node and end node plus the
        specified number of additional nodes and related edges connecting the
        additional nodes to the graph's start and end node.
        """
        start: SQLStage = SqlStageMock(1)
        end: SQLStage = SqlStageMock(2)
        edges = {(start, end)}
        for i in range(0, extra_nodes):
            node = cast(SQLStage, (SqlStageMock(3 + i)))
            edges.update({(start, node), (node, end)})
        return SQLStageGraph(start, end, edges)

    g1 = graph()
    g2 = graph(1)
    g3 = graph(2)

    expected = g1.edges() | g2.edges() | g3.edges()
    assert expected == all_edges([g1, g2, g3])


def test_linear_interlinks():
    g1 = sql_stage_graph(1, 2)
    g2 = sql_stage_graph(3, 4)
    g3 = sql_stage_graph(5, 6)
    expected = {
        (g1.end_node, g2.start_node),
        (g2.end_node, g3.start_node),
    }
    assert expected == linear_interlinks([g1, g2, g3])


def collapse_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # Remove leading and trailing spaces in brackets.
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def test_get_extractor_output_table(extractor_context):
    extractor = SourceTableExtractor(sources=source_tables())
    output_table = get_output_table(extractor, extractor_context)
    assert output_table.name.endswith("SOURCE_TABLE")
    assert output_table.name.startswith(extractor_context.output.table_prefix)
    assert output_table.schema_name.name == extractor_context.output.db_schema


def test_get_extractor_explicit_output_table(extractor_context):
    extractor = KeywordExtractor(
        keyword_settings=PatternRankKeywordSettings(), db_output="MY_TABLE"
    )
    output_table = get_output_table(extractor, extractor_context)
    assert output_table.name == "MY_TABLE"


def test_get_input_list():
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=[
            ListInputOutput(
                input_output_list=[SQLStageInputOutput(), SQLStageInputOutput()]
            )
        ],
        result_bucketfs_location=None,  # type: ignore
    )
    input_list = get_input_list(stage_input)
    assert len(input_list) == 2


@pytest.mark.parametrize(
    "inputs",
    [
        [SQLStageInputOutput()],
        [
            ListInputOutput(input_output_list=[SQLStageInputOutput()]),
            ListInputOutput(input_output_list=[SQLStageInputOutput()]),
        ],
    ],
    ids=["not-a-list", "multiple-lists"],
)
def test_invalid_input_list(inputs):
    stage_input = SQLStageQueryHandlerInput(
        sql_stage_inputs=inputs, result_bucketfs_location=None  # type: ignore
    )
    with pytest.raises(ValueError, match="Invalid stage graph input.+"):
        get_input_list(stage_input)
