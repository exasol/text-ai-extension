import pytest
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol.ai.text.extraction.abstract_extraction import (
    Defaults,
    Output,
)
from exasol.ai.text.impl.stage_graph.graph_builders import ExtractorContext


@pytest.fixture
def empty_exa() -> MockMetaData:
    return MockMetaData(
        None, input_type="SET", input_columns=[], output_type="EMITS", output_columns=[]
    )


@pytest.fixture
def extractor_context() -> ExtractorContext:
    return ExtractorContext(
        language_alias="my_alias",
        output=Output(db_schema="my_schema", table_prefix="my_prefix"),
        defaults=Defaults(),
    )
