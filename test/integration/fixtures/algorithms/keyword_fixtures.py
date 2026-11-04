# pylint: disable=line-too-long
from collections.abc import Generator
from dataclasses import dataclass
from textwrap import dedent

import exasol.bucketfs as bfs
import pytest
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from transformers import AutoModel

from test.integration.utils.model_utils import (
    download_model_locally,
    download_model_to_bucketfs,
)
from test.integration.utils.text_test_case import TextTestCase


@pytest.fixture(scope="session")
def feature_extraction_model():
    return "Lihuchen/pearl_small"


@pytest.fixture(scope="session")
def feature_extraction_model_revision():
    return "581e568640860124bfa5df6f6a2a25d3d9e57645"


@pytest.fixture(scope="session")
def feature_extraction_model_location(
    tmp_path_factory,
    feature_extraction_model,
    feature_extraction_model_revision,
    model_subdir,
) -> bfs.path.PathLike:
    model_specification = ModelSpecification(
        feature_extraction_model, "feature-extraction"
    )
    return download_model_locally(
        model_specification,
        feature_extraction_model_revision,
        tmp_path_factory,
        model_subdir,
        AutoModel,
    )


@pytest.fixture(scope="session")
def feature_extraction_model_download(
    feature_extraction_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    with download_model_to_bucketfs(
        feature_extraction_model,
        model_subdir,
        "feature-extraction",
        backend_aware_bucketfs_params,
        AutoModel,
    ):
        yield


@dataclass
class KeywordsTestCase(TextTestCase):
    keywords: list[str]


@pytest.fixture
def keyword_test_cases() -> list[KeywordsTestCase]:
    return [
        KeywordsTestCase(
            text_doc_id=1234,
            text_char_begin=200,
            text=dedent("""
                If there's racing on today then we'll have expert tips from our
                resident racing tipsters, sharing the Nap of the Day, Lucky 15
                selections and more. We always try to post our racing preview the night
                before, to give you the best chance to get on our tips at the best
                prices, but sometimes our tipsters prefer to wait until the day of the
                race to see if there's any non-runners or changes to the going which
                may affect their predictions.
                All our horse racing tips are free. No sign-up is required to access
                our free racing tips and comments from members of our community, who
                often post their own selections and predictions every day.
            """),
            keywords=["racing", "tips"],
        ),
        KeywordsTestCase(
            text_doc_id=5678,
            text_char_begin=400,
            text=dedent("""
                Knowledge is power, and greyhound betting is no exception. It is
                crucial for beginners to invest time in understanding the basics of
                greyhound racing before they start with betting.
                Greyhound races are categorized based on their distance, age, and
                quality of the dogs. All these factors can have a significant impact
                on the outcome of a race. Therefore, understanding these areas will
                help you make more informed bets.
                Understanding how odds work in greyhound racing is crucial to make
                informed bets. The odds represent the probability of a certain dog
                winning the race. Therefore, a dog with lower odds is seen as more
                likely to win than a dog with higher odds.
                Yet, this doesn’t mean you should always bet on the favorite.
                Experienced bettors know that sometimes dogs with higher odds can
                surprise. Consequently, interpreting the odds in context to the race
                conditions and the dog’s previous performances can lead to successful
                betting decisions.
            """),
            keywords=["racing", "greyhound"],
        ),
    ]
