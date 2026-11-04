from typing import cast

from exasol.ai.text.extractors import (
    KeywordExtractor,
    PatternRankKeywordSettings,
)
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor import (
    PatternRankKeywordExtractor as PatternRankKeywordAlgorithm,
)
from exasol.ai.text.impl.stage_graph.graph_builders import KeywordGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_keyword_graph_builder(extractor_context):
    extractor = KeywordExtractor(
        keyword_settings=PatternRankKeywordSettings(
            ranking_model=HuggingFaceModel(
                name="my/awesome_model",
                repository=BucketFSRepository(
                    connection_name="my_bucketfs_conn", sub_dir="my_model_dir"
                ),
            ),
            max_df=99,
            min_df=3,
            top_n=7,
            use_maxsum=True,
            use_mmr=True,
            diversity=0.75,
            nr_candidates=18,
        ),
        name="MyKeyword",
    )

    builder = KeywordGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    assert isinstance(stage.udf_runner.algorithm, PatternRankKeywordAlgorithm)
    algorithm = cast(PatternRankKeywordAlgorithm, stage.udf_runner.algorithm)
    settings = extractor.keyword_settings
    assert algorithm.model_name == settings.ranking_model.name
    assert (
        algorithm.bucketfs_conn_name
        == settings.ranking_model.repository.connection_name
    )
    assert algorithm.sub_dir == settings.ranking_model.repository.sub_dir
    assert algorithm.vec_kwargs == {
        "max_df": settings.max_df,
        "min_df": settings.min_df,
        "pos_pattern": settings.pos_pattern,
    }
    assert algorithm.kbx_kwargs == {
        "top_n": settings.top_n,
        "use_maxsum": settings.use_maxsum,
        "use_mmr": settings.use_mmr,
        "diversity": settings.diversity,
        "nr_candidates": settings.nr_candidates,
    }
    assert algorithm.outputs.name == "MyKeyword"
