import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import named_entity_extractor


def test_named_entity_extractor_no_settings():
    with pytest.raises(ValidationError):
        named_entity_extractor(named_entity_settings=None)


def test_named_entity_extractor_success():
    extractor = named_entity_extractor()
    assert isinstance(extractor, BaseExtractor)
    assert isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "NamedEntityExtractor"
