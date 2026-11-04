from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.named_entity_extractor import NamedEntityBase
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition import (
    HftNamedEntityRecognition as HftNamedEntityAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityOutput,
)
from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityRecognition as NamedEntityAlgorithm,
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
) -> NamedEntityAlgorithm:
    if isinstance(extractor, NamedEntityBase):
        # At the moment there is only one named entity recognition algorithm
        # in use. Once there is a choice of such algorithms this code should
        # switch on the discriminator of the `named_entity_settings`.
        ne_settings = cast(NamedEntityBase, extractor).named_entity_settings
        return HftNamedEntityAlgorithm(
            model_name=ne_settings.model.name,
            bucketfs_conn_name=get_bucketfs_connection_name(context, ne_settings.model),
            sub_dir=get_bucketfs_subdir(context, ne_settings.model),
            ignore_labels=ne_settings.ignore_labels,
            aggregation_strategy=ne_settings.aggregation_strategy,
            outputs=NamedEntityOutput(name=extractor.name),
        )
    raise TypeError(
        f"The extractor of type {type(extractor)} does not "
        f"inherit from {NamedEntityBase.__name__}."
    )


class NamedEntityGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for a named entity extractor.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        return cast(AlgorithmModel, algorithm_from_extractor(extractor, context))
