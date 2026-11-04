import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import keyword_extractor


def test_keyword_extractor_no_settings():
    with pytest.raises(ValidationError):
        keyword_extractor(keyword_settings=None)


def test_keyword_extractor_success():
    extractor = keyword_extractor()
    assert isinstance(extractor, BaseExtractor)
    assert isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "KeywordExtractor"
