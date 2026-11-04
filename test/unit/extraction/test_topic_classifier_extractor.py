import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import topic_classifier_extractor


def test_topic_classifier_extractor_no_settings():
    with pytest.raises(ValidationError):
        topic_classifier_extractor(topic_settings=None)


def test_topic_classifier_extractor_success():
    extractor = topic_classifier_extractor()
    assert isinstance(extractor, BaseExtractor)
    assert isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "TopicClassifierExtractor"
