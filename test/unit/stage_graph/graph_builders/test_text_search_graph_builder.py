from typing import cast

from exasol.ai.text.extractors import (
    TextSearchExtractor,
    TokenizedTextSearchSettings,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)
from exasol.ai.text.impl.stage_graph.graph_builders import TextSearchGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_text_search_graph_builder(extractor_context):

    extractor = TextSearchExtractor(
        text_search_settings=TokenizedTextSearchSettings(
            fuzziness=3, ignore_spaces=True
        ),
        name="MyTextSearch",
    )

    builder = TextSearchGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    assert isinstance(stage.udf_runner.algorithm, TokenizedTextSearch)
    algorithm = cast(TokenizedTextSearch, stage.udf_runner.algorithm)
    settings = extractor.text_search_settings
    assert algorithm.fuzziness == settings.fuzziness
    assert algorithm.ignore_spaces == settings.ignore_spaces
    assert algorithm.outputs.name == "MyTextSearch"
