from typing import cast

from exasol.ai.text.extractors import (
    KeywordSearchExtractor,
    PatternRankKeywordSettings,
    TokenizedTextSearchSettings,
)
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor import (
    PatternRankKeywordExtractor as PatternRankKeywordAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)
from exasol.ai.text.impl.stage_graph.graph_builders import KeywordSearchGraphBuilder
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)


def test_keyword_search_graph_builder(extractor_context):
    extractor = KeywordSearchExtractor(
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
        text_search_settings=TokenizedTextSearchSettings(
            fuzziness=3, ignore_spaces=True
        ),
        name="MyKeywordSearch",
    )

    builder = KeywordSearchGraphBuilder()
    graph = builder.build(extractor, context=extractor_context)

    assert isinstance(graph.start_node, UDFAlgoSQLStage)
    stage = cast(UDFAlgoSQLStage, graph.start_node)
    algorithms = stage.udf_runner.algorithm.algorithms
    assert isinstance(algorithms[0], PatternRankKeywordAlgorithm)
    assert isinstance(algorithms[1], TokenizedTextSearch)
    keyword_algo = cast(PatternRankKeywordAlgorithm, algorithms[0])
    keyword_settings = extractor.keyword_settings
    assert keyword_algo.model_name == keyword_settings.ranking_model.name
    assert (
        keyword_algo.bucketfs_conn_name
        == keyword_settings.ranking_model.repository.connection_name
    )
    assert keyword_algo.sub_dir == keyword_settings.ranking_model.repository.sub_dir
    assert keyword_algo.vec_kwargs == {
        "max_df": keyword_settings.max_df,
        "min_df": keyword_settings.min_df,
        "pos_pattern": keyword_settings.pos_pattern,
    }
    assert keyword_algo.kbx_kwargs == {
        "top_n": keyword_settings.top_n,
        "use_maxsum": keyword_settings.use_maxsum,
        "use_mmr": keyword_settings.use_mmr,
        "diversity": keyword_settings.diversity,
        "nr_candidates": keyword_settings.nr_candidates,
    }
    search_algo = cast(TokenizedTextSearch, algorithms[1])
    search_settings = extractor.text_search_settings
    assert search_algo.fuzziness == search_settings.fuzziness
    assert search_algo.ignore_spaces == search_settings.ignore_spaces
    assert stage.udf_runner.algorithm.outputs.name == "MyKeywordSearch"
