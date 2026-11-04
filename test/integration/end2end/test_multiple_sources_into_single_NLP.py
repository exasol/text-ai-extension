import pytest

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import BranchExtractor as Branch
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.helpers import (
    input_table,
    input_table_name,
    named_entity_extractor,
    not_raises,
    source_table_extractor,
    source_tables,
)


@pytest.fixture
def other_input_table(pyexasol_connection, end2end_input_schema):
    pyexasol_connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {end2end_input_schema.quoted_name}"
    )
    table = input_table(input_table_name(2))
    pyexasol_connection.execute(table.create_statement)
    return table


@pytest.mark.xfail(
    reason="Blocked by https://github.com/exasol/text-ai-extension/issues/210",
    strict=True,
)
def test_extraction_run(
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    other_input_table,
    end2end_output_schema,
    bucketfs_conn_name,
    model_subdir,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    """
    Create a branch with two source tables fed into a single NLP algorithm
    and verify sucessful extraction: Pipeline(Branch(S1, S2), NLP)
    """
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES"
        " (1, 'This is a test.'),"
        " (2, 'Exasol is awesome.')"
    )
    pyexasol_connection.execute(
        f"INSERT INTO {other_input_table.name.fully_qualified} VALUES"
        " (3, 'I see many green trees.'),"
        " (4, 'Biology is fascinating.')"
    )
    extraction = Extraction(
        extractor=Pipeline(
            steps=[
                Branch(
                    paths=[
                        source_table_extractor(),
                        source_table_extractor(source_tables(2)),
                    ]
                ),
                named_entity_extractor(),
            ]
        ),
        output=Output(db_schema=end2end_output_schema.name),
    )

    with not_raises(Exception):
        extraction.run(pyexasol_connection, db_schema_name, language_alias)
