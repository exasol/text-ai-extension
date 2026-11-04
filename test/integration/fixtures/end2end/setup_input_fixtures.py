import pytest
from exasol.analytics.schema import SchemaName

from test.utils.helpers import (
    input_schema,
    input_table,
)


@pytest.fixture
def end2end_input_schema(pyexasol_connection):
    input_schema_name = input_schema()
    pyexasol_connection.execute(
        f"DROP SCHEMA IF EXISTS {input_schema_name.quoted_name} CASCADE"
    )
    pyexasol_connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {input_schema_name.quoted_name}"
    )
    return input_schema_name


@pytest.fixture
def end2end_output_schema(pyexasol_connection):
    output_schema_name = SchemaName("OUTPUT_SCHEMA")
    pyexasol_connection.execute(
        f"DROP SCHEMA IF EXISTS {output_schema_name.quoted_name} CASCADE"
    )
    pyexasol_connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {output_schema_name.quoted_name}"
    )
    return output_schema_name


@pytest.fixture
def end2end_input_table(pyexasol_connection, end2end_input_schema):
    pyexasol_connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {end2end_input_schema.quoted_name}"
    )
    table = input_table()
    pyexasol_connection.execute(table.create_statement)
    pyexasol_connection.execute(f"TRUNCATE TABLE {table.name.fully_qualified};")
    return table
