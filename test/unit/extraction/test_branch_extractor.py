import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors import BranchExtractor
from test.utils.helpers import (
    keyword_extractor,
    keyword_search_extractor,
    named_entity_extractor,
    source_table_extractor,
    topic_classifier_extractor,
)


def test_empty_branch():
    with pytest.raises(ValidationError):
        BranchExtractor()


def test_empty_paths():
    with pytest.raises(ValidationError):
        BranchExtractor(paths=[])


@pytest.mark.parametrize(
    "paths",
    [
        [source_table_extractor()],
        [named_entity_extractor()],
        [keyword_search_extractor()],
        [keyword_extractor()],
        [topic_classifier_extractor()],
        [named_entity_extractor()] * 2,
        [source_table_extractor()] * 2,
        [keyword_extractor(), topic_classifier_extractor()],
    ],
    ids=[
        "SourceTableExtractor",
        "NamedEntityExtractor",
        "KeywordSearchExtractor",
        "KeywordExtractor",
        "TopicClassifierExtractor",
        "2xNamedEntityExtractor",
        "2xSourceTableExtractor",
        "KeywordExtractor, TopicClassifierExtractor",
    ],
)
def test_non_empty_branch(paths):
    branch = BranchExtractor(paths=paths)
    assert branch.paths == paths
