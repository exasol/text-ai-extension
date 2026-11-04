import pytest

from test.utils.helpers import BUCKETFS_CONNECTION_NAME

SCHEMA_NAME = "TEXT_AI_TEST"
LANGUAGE_ALIAS = "PYTHON3_TEXT_AI_TEST"


@pytest.fixture(scope="session")
def db_schema_name() -> str:
    # Database schema is created by default implementation of fixture pyexasol_connection in
    # https://github.com/exasol/pytest-plugins/blob/main/pytest-extension/exasol/pytest_extension/__init__.py#L35
    return SCHEMA_NAME


@pytest.fixture(scope="session")
def language_alias(project_short_tag):
    return LANGUAGE_ALIAS


@pytest.fixture(scope="session")
def bucketfs_conn_name() -> str:
    return BUCKETFS_CONNECTION_NAME


@pytest.fixture(scope="session")
def setup_database(
    use_onprem, use_saas, deployed_slc, bucketfs_conn_name, bucketfs_connection_factory
):
    if not (use_onprem or use_saas):
        pytest.skip("The test requires a database.")
    bucketfs_connection_factory(bucketfs_conn_name)


@pytest.fixture(scope="session")
def setup_no_database(use_onprem, use_saas):
    if use_onprem or use_saas:
        pytest.skip("The test doesn't use the database.")
