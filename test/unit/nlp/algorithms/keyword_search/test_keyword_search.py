import json
from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearch,
)
from exasol.ai.text.impl.nlp.algorithms.keywords import PatternRankKeywordExtractor
from exasol.ai.text.impl.nlp.algorithms.text_search import TokenizedTextSearch


@pytest.fixture
def algorithm() -> KeywordSearch:
    kw = PatternRankKeywordExtractor(
        model_name="my model", bucketfs_conn_name="", sub_dir="models"
    )
    ts = TokenizedTextSearch()
    return KeywordSearch(algorithms=(kw, ts), setup="my setup")


@pytest.fixture
def input_df(algorithm) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (123, 100, 136, "aaa kw11 aa kw12 aa kw11 aa kw12 aaa"),
            (456, 200, 226, "bbb kw21 b kw21 b kw21 bbb"),
            (789, 300, 331, "kw32 ccc kw31 ccc kw31 ccc kw32"),
        ],
        columns=algorithm.inputs.column_names,
    )


@patch(
    "exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor.PatternRankKeywordExtractor.extract_keywords"
)
@patch(
    "exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search.TokenizedTextSearch.locate_phrases"
)
def test_run_impl(
    mock_locate_phrases, mock_extract_keywords, input_df, algorithm, empty_exa
):
    mock_extract_keywords.side_effect = [
        [
            [("kw11", 0.3), ("kw12", 0.4)],
            [("kw21", 0.6)],
            [("kw31", 0.5), ("kw32", 0.8)],
        ]
    ]
    mock_locate_phrases.side_effect = [
        [(0, 4, 8), (1, 12, 16), (0, 20, 24), (1, 28, 32)],
        [(0, 4, 8), (0, 11, 15), (0, 18, 22)],
        [(1, 0, 4), (0, 9, 13), (0, 18, 22), (1, 27, 31)],
    ]

    expected_output_df = pd.DataFrame(
        [
            (
                123,
                100,
                136,
                "aaa kw11 aa kw12 aa kw11 aa kw12 aaa",
                "kw11",
                0.3,
                123,
                104,
                108,
                None,
                "my setup",
            ),
            (
                123,
                100,
                136,
                "aaa kw11 aa kw12 aa kw11 aa kw12 aaa",
                "kw11",
                0.3,
                123,
                120,
                124,
                None,
                "my setup",
            ),
            (
                123,
                100,
                136,
                "aaa kw11 aa kw12 aa kw11 aa kw12 aaa",
                "kw12",
                0.4,
                123,
                112,
                116,
                None,
                "my setup",
            ),
            (
                123,
                100,
                136,
                "aaa kw11 aa kw12 aa kw11 aa kw12 aaa",
                "kw12",
                0.4,
                123,
                128,
                132,
                None,
                "my setup",
            ),
            (
                456,
                200,
                226,
                "bbb kw21 b kw21 b kw21 bbb",
                "kw21",
                0.6,
                456,
                204,
                208,
                None,
                "my setup",
            ),
            (
                456,
                200,
                226,
                "bbb kw21 b kw21 b kw21 bbb",
                "kw21",
                0.6,
                456,
                211,
                215,
                None,
                "my setup",
            ),
            (
                456,
                200,
                226,
                "bbb kw21 b kw21 b kw21 bbb",
                "kw21",
                0.6,
                456,
                218,
                222,
                None,
                "my setup",
            ),
            (
                789,
                300,
                331,
                "kw32 ccc kw31 ccc kw31 ccc kw32",
                "kw31",
                0.5,
                789,
                309,
                313,
                None,
                "my setup",
            ),
            (
                789,
                300,
                331,
                "kw32 ccc kw31 ccc kw31 ccc kw32",
                "kw31",
                0.5,
                789,
                318,
                322,
                None,
                "my setup",
            ),
            (
                789,
                300,
                331,
                "kw32 ccc kw31 ccc kw31 ccc kw32",
                "kw32",
                0.8,
                789,
                300,
                304,
                None,
                "my setup",
            ),
            (
                789,
                300,
                331,
                "kw32 ccc kw31 ccc kw31 ccc kw32",
                "kw32",
                0.8,
                789,
                327,
                331,
                None,
                "my setup",
            ),
        ],
        columns=[
            *input_df.columns,
            algorithm.outputs.keyword.name,
            algorithm.outputs.keyword_score.name,
            *algorithm.outputs.keyword_span.column_names,
            algorithm.outputs.error_message.name,
            algorithm.outputs.setup.name,
        ],
    )

    output_df = algorithm.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


def test_set_empty(input_df, algorithm):
    expected_output_df = pd.DataFrame(
        [
            (
                123,
                100,
                136,
                "aaa kw11 aa kw12 aa kw11 aa kw12 aaa",
                None,
                None,
                None,
                None,
                None,
                "the error",
                "my setup",
            ),
            (
                456,
                200,
                226,
                "bbb kw21 b kw21 b kw21 bbb",
                None,
                None,
                None,
                None,
                None,
                "the error",
                "my setup",
            ),
            (
                789,
                300,
                331,
                "kw32 ccc kw31 ccc kw31 ccc kw32",
                None,
                None,
                None,
                None,
                None,
                "the error",
                "my setup",
            ),
        ],
        columns=[
            *input_df.columns,
            algorithm.outputs.keyword.name,
            algorithm.outputs.keyword_score.name,
            *algorithm.outputs.keyword_span.column_names,
            algorithm.outputs.error_message.name,
            algorithm.outputs.setup.name,
        ],
    )

    output_df = algorithm.set_empty(input_df, "the error")
    pd.testing.assert_frame_equal(output_df, expected_output_df, check_like=True)


def test_algo_setup(algorithm):
    algorithm.setup = None
    algo_setup = json.loads(algorithm.algo_setup)
    expected_setup = {
        "KeywordSearch": [
            {"PatternRankKeywordExtractor": {"model_name": "my model"}},
            {"TokenizedTextSearch": {}},
        ]
    }
    assert algo_setup == expected_setup
