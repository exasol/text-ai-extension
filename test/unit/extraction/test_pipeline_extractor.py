import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors import PipelineExtractor
from test.utils.helpers import (
    keyword_extractor,
    keyword_search_extractor,
    named_entity_extractor,
    source_table_extractor,
    topic_classifier_extractor,
)


def test_empty_pipeline():
    with pytest.raises(ValidationError):
        PipelineExtractor()


def test_empty_steps():
    with pytest.raises(ValidationError):
        PipelineExtractor(steps=[])


@pytest.mark.parametrize(
    "steps",
    [
        [source_table_extractor()],
        [named_entity_extractor()],
        [keyword_search_extractor()],
        [keyword_extractor()],
        [topic_classifier_extractor()],
        [named_entity_extractor()] * 2,
        [source_table_extractor(), topic_classifier_extractor()],
    ],
    ids=[
        "SourceTableExtractor",
        "NamedEntityExtractor",
        "KeywordSearchExtractor",
        "KeywordExtractor",
        "TopicClassifierExtractor",
        "2xNamedEntityExtractor",
        "SourceTableExtractor, TopicClassifierExtractor",
    ],
)
def test_non_empty_pipeline(steps):
    pipeline = PipelineExtractor(steps=steps)
    assert pipeline.steps == steps
