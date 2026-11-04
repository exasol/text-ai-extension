import pytest

from exasol.ai.text.extraction.abstract_extraction import Output


@pytest.fixture
def output():
    return Output(db_schema="abc", table_prefix="p1")
