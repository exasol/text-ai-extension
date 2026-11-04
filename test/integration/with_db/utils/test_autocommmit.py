import contextlib

import pyexasol
import pytest

from exasol.ai.text.impl.utils.pyexasol_utils import autocommit_off


@pytest.fixture(scope="session")
def pyexasol_conn_factory(backend_aware_database_params):
    def connect(autocommit: bool) -> pyexasol.ExaConnection:
        return pyexasol.connect(
            **backend_aware_database_params,
            autocommit=autocommit,
            compression=True,
        )

    return connect


def schema_exists(conn: pyexasol.ExaConnection, schema: str) -> bool:
    row = conn.execute(
        "SELECT * from EXA_USER_SCHEMAS WHERE" f" SCHEMA_NAME = '{schema}'"
    ).fetchone()
    return row is not None


def autocommit_enabled(conn: pyexasol.ExaConnection) -> bool:
    return conn.attr["autocommit"] == True


@contextlib.contextmanager
def autocommit_db_schema(conn: pyexasol.ExaConnection):
    schema = "AUTOCOMMIT_TEST_SCHEMA"
    conn.execute(f"CREATE SCHEMA {schema}")
    try:
        yield schema
    finally:
        conn.execute(f"DROP SCHEMA {schema}")


@pytest.fixture
def inspect_connection(pyexasol_conn_factory) -> pyexasol.ExaConnection:
    """
    inspect_connection must have autocommit=True, as otherwise changes
    from the sample_connection could be hidden until the inspect_connection
    does a commit.
    """
    return pyexasol_conn_factory(autocommit=True)


def test_default_behavior(pyexasol_conn_factory, inspect_connection):
    sample_connection = pyexasol_conn_factory(autocommit=True)
    with autocommit_db_schema(sample_connection) as schema:
        assert schema_exists(inspect_connection, schema)


def test_autocommit_off(pyexasol_conn_factory, inspect_connection):
    sample_connection = pyexasol_conn_factory(autocommit=True)
    with autocommit_off(sample_connection):
        assert not autocommit_enabled(sample_connection)
        with autocommit_db_schema(sample_connection) as schema:
            assert not schema_exists(inspect_connection, schema)
    assert autocommit_enabled(sample_connection)


class SampleException(Exception):
    pass


def test_exception(pyexasol_conn_factory, inspect_connection):
    """
    This test verifies that in case an exception is raised inside the
    autocommit_off context: 1. besides restoring the connection's autocommit
    setting (as always) 2. also the changes are rolled back.
    """
    sample_connection = pyexasol_conn_factory(autocommit=True)
    schema = "AUTOCOMMIT_TEST_SCHEMA"
    with pytest.raises(SampleException):
        with autocommit_off(sample_connection):
            assert not autocommit_enabled(sample_connection)
            sample_connection.execute(f"CREATE SCHEMA {schema}")
            raise SampleException("Some Exception")
    assert autocommit_enabled(sample_connection)
    assert not schema_exists(inspect_connection, schema)
