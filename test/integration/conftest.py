import pytest
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from spacy.cli.download import download as spacy_download

from exasol.ai.text.impl.utils.constants import (
    DEFAULT_SPACY_PIPELINE,
    DEFAULT_SPACY_PIPELINE_VERSION,
)
from test.utils.helpers import BUCKETFS_SUB_DIR

pytest_plugins = [
    "test.integration.fixtures.language_container_fixture",
    "test.integration.fixtures.setup_database_fixture",
    "test.integration.fixtures.algorithms.named_entity_fixtures",
    "test.integration.fixtures.algorithms.topic_fixtures",
    "test.integration.fixtures.algorithms.text_search_fixtures",
    "test.integration.fixtures.algorithms.keyword_fixtures",
    "test.integration.fixtures.end2end.named_entity_fixtures",
    "test.integration.fixtures.end2end.topic_fixtures",
    "test.integration.fixtures.end2end.keyword_fixtures",
    "test.integration.fixtures.end2end.setup_input_fixtures",
    "test.utils.tokenizer_test_cases",
]


@pytest.fixture
def empty_exa() -> MockMetaData:
    return MockMetaData(
        None, input_type="SET", input_columns=[], output_type="EMITS", output_columns=[]
    )


@pytest.fixture(scope="session")
def spacy_pipeline():
    spacy_download(
        f"{DEFAULT_SPACY_PIPELINE}-{DEFAULT_SPACY_PIPELINE_VERSION}", direct=True
    )


@pytest.fixture(scope="session")
def model_subdir():
    return BUCKETFS_SUB_DIR
