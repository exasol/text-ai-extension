from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.topic_classifier_extractor import TopicBase
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.topics.hft_topic_classification import (
    HftTopicClassification as HftTopicAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.topics.topic_classification import (
    TopicClassification as TopicAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.topics.topic_classification import (
    TopicClassifierOutput,
)
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.nlp_graph_builder import (
    NLPGraphBuilder,
    get_bucketfs_connection_name,
    get_bucketfs_subdir,
)


def algorithm_from_extractor(
    extractor: NLPExtractor, context: ExtractorContext
) -> TopicAlgorithm:
    if isinstance(extractor, TopicBase):
        # At the moment there is only one topic classification algorithm
        # in use. Once there is a choice of such algorithms this code should
        # switch on the discriminator of the `topic_settings`.
        ti_settings = cast(TopicBase, extractor).topic_settings
        return HftTopicAlgorithm(
            model_name=ti_settings.ranking_model.name,
            bucketfs_conn_name=get_bucketfs_connection_name(
                context, ti_settings.ranking_model
            ),
            sub_dir=get_bucketfs_subdir(context, ti_settings.ranking_model),
            topics=sorted(ti_settings.topics),
            hypothesis_template=ti_settings.hypothesis_template,
            multi_label=ti_settings.multi_label,
            outputs=TopicClassifierOutput(name=extractor.name),
        )
    raise TypeError(
        f"The extractor of type {type(extractor)} does not "
        f"inherit from {TopicBase.__name__}."
    )


class TopicClassifierGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for topic extractor.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        return cast(AlgorithmModel, algorithm_from_extractor(extractor, context))
