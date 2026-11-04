from typing import cast

from exasol.ai.text.extractors import (
    HftTopicSettings,
    TopicClassifierExtractor,
)
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.nlp.algorithms.topics.hft_topic_classification import (
    HftTopicClassification,
)
from exasol.ai.text.impl.stage_graph.graph_builders import TopicClassifierGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_topic_classifier_graph_builder(extractor_context):
    extractor = TopicClassifierExtractor(
        topic_settings=HftTopicSettings(
            ranking_model=HuggingFaceModel(
                name="my/awesome_model",
                repository=BucketFSRepository(
                    connection_name="my_bucketfs_conn", sub_dir="my_model_dir"
                ),
            ),
            topics=["A", "C", "B"],
            hypothesis_template="{xyz}",
            multi_label=True,
        ),
        name="MyTopicClassifier",
    )

    builder = TopicClassifierGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    assert isinstance(stage.udf_runner.algorithm, HftTopicClassification)
    algorithm = cast(HftTopicClassification, stage.udf_runner.algorithm)
    settings = extractor.topic_settings
    assert algorithm.model_name == settings.ranking_model.name
    assert (
        algorithm.bucketfs_conn_name
        == settings.ranking_model.repository.connection_name
    )
    assert algorithm.sub_dir == settings.ranking_model.repository.sub_dir
    assert algorithm.topics == ["A", "B", "C"]
    assert algorithm.hypothesis_template == settings.hypothesis_template
    assert algorithm.multi_label == settings.multi_label
    assert algorithm.outputs.name == "MyTopicClassifier"
