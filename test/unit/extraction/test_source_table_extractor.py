import pytest
from pydantic import ValidationError

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NLPExtractor,
)
from test.utils.helpers import (
    source_table_extractor,
    source_tables,
)


def test_source_table_extractor_no_sources():
    with pytest.raises(ValidationError):
        source_table_extractor(sources=None)


def test_source_table_extractor_empty_sources():
    with pytest.raises(ValidationError):
        source_table_extractor(sources=[])


@pytest.mark.parametrize("source_count", [1, 2, 3])
def test_source_table_extractor_success(source_count):
    sources = source_tables() * source_count
    extractor = source_table_extractor(sources=sources)
    assert isinstance(extractor, BaseExtractor)
    assert not isinstance(extractor, NLPExtractor)
    assert extractor.extractor_type == "SourceTableExtractor"
