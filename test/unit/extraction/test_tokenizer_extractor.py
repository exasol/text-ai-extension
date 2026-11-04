import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import tokenizer_extractor


def test_tokenizer_extractor_no_settings():
    with pytest.raises(ValidationError):
        tokenizer_extractor(settings=None)


def test_tokenizer_extractor_success():
    extractor = tokenizer_extractor()
    assert isinstance(extractor, BaseExtractor)
    assert isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "TokenizerExtractor"
