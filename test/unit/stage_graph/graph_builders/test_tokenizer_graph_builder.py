from typing import cast

from exasol.ai.text.extractors import (
    SpacyTokenizerSettings,
    TokenizerExtractor,
)
from exasol.ai.text.impl.nlp.algorithms.tokenizer.spacy_tokenizer import SpacyTokenizer
from exasol.ai.text.impl.stage_graph.graph_builders import TokenizerGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_tokenizer_graph_builder(extractor_context):

    extractor = TokenizerExtractor(
        tokenizer_settings=SpacyTokenizerSettings(model_name="my_pipeline"),
        name="MyTokenizer",
    )

    builder = TokenizerGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    assert isinstance(stage.udf_runner.algorithm, SpacyTokenizer)
    assert stage.udf_runner.algorithm.outputs.name == "MyTokenizer"
