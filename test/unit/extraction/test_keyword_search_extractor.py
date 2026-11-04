import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import keyword_search_extractor


def test_keyword_search_extractor_no_keyword_settings():
    with pytest.raises(ValidationError):
        keyword_search_extractor(keyword_settings=None)


def test_keyword_search_extractor_no_text_search_settings():
    with pytest.raises(ValidationError):
        keyword_search_extractor(text_search_settings=None)


def test_keyword_search_extractor_success():
    extractor = keyword_search_extractor()
    assert isinstance(extractor, BaseExtractor)
    assert isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "KeywordSearchExtractor"
