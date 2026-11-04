from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.keyword_extractor import KeywordBase
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import (
    KeywordExtractor as KeywordAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import (
    KeywordOutput,
)
from exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor import (
    PatternRankKeywordExtractor as PatternRankKeywordAlgorithm,
)
from exasol.ai.text.impl.stage_graph.graph_builders import ExtractorContext
from exasol.ai.text.impl.stage_graph.graph_builders.nlp_graph_builder import (
    NLPGraphBuilder,
    get_bucketfs_connection_name,
    get_bucketfs_subdir,
)


def algorithm_from_extractor(
    extractor: NLPExtractor, context: ExtractorContext
) -> KeywordAlgorithm:
    if isinstance(extractor, KeywordBase):
        # At the moment there is only one keyword extraction algorithm
        # in use. Once there is a choice of such algorithms this code should
        # switch on the discriminator of the `keyword_settings`.
        kw_settings = cast(KeywordBase, extractor).keyword_settings
        return PatternRankKeywordAlgorithm(
            model_name=kw_settings.ranking_model.name,
            bucketfs_conn_name=get_bucketfs_connection_name(
                context, kw_settings.ranking_model
            ),
            sub_dir=get_bucketfs_subdir(context, kw_settings.ranking_model),
            vec_kwargs={
                "max_df": kw_settings.max_df,
                "min_df": kw_settings.min_df,
                "pos_pattern": kw_settings.pos_pattern,
            },
            kbx_kwargs={
                "top_n": kw_settings.top_n,
                "use_maxsum": kw_settings.use_maxsum,
                "use_mmr": kw_settings.use_mmr,
                "diversity": kw_settings.diversity,
                "nr_candidates": kw_settings.nr_candidates,
            },
            outputs=KeywordOutput(name=extractor.name),
        )
    raise TypeError(
        f"The extractor of type {type(extractor)} does not "
        f"inherit from {KeywordBase.__name__}."
    )


class KeywordGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for a keyword extractor.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        return cast(AlgorithmModel, algorithm_from_extractor(extractor, context))
