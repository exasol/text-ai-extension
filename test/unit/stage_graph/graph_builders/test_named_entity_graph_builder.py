from typing import cast

from exasol.ai.text.extractors import (
    HftNamedEntitySettings,
    NamedEntityExtractor,
)
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition import (
    HftNamedEntityRecognition,
)
from exasol.ai.text.impl.stage_graph.graph_builders import NamedEntityGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_named_entity_graph_builder(extractor_context):
    extractor = NamedEntityExtractor(
        named_entity_settings=HftNamedEntitySettings(
            model=HuggingFaceModel(
                name="my/awesome_model",
                repository=BucketFSRepository(
                    connection_name="my_bucketfs_conn", sub_dir="my_model_dir"
                ),
            ),
            ignore_labels=["A", "B", "C"],
            aggregation_strategy="first",
        ),
        name="MyNamedEntity",
    )

    builder = NamedEntityGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    assert isinstance(stage.udf_runner.algorithm, HftNamedEntityRecognition)
    algorithm = cast(HftNamedEntityRecognition, stage.udf_runner.algorithm)
    settings = extractor.named_entity_settings
    assert algorithm.model_name == settings.model.name
    assert algorithm.bucketfs_conn_name == settings.model.repository.connection_name
    assert algorithm.sub_dir == settings.model.repository.sub_dir
    assert algorithm.ignore_labels == settings.ignore_labels
    assert algorithm.aggregation_strategy == settings.aggregation_strategy
    assert algorithm.outputs.name == "MyNamedEntity"
