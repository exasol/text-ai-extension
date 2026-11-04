import pytest

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import BranchExtractor as Branch
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.helpers import (
    input_table,
    input_table_name,
    keyword_search_extractor,
    named_entity_extractor,
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
    reason="Blocked by https://github.com/exasol/text-ai-extension/issues/214",
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
    end2end_feature_extraction_model,
    end2end_feature_extraction_model_download,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    """
    Create two source tables with different document IDs.  Setup an
    extraction of a branch, with 2 pipelines, each pipeline using a different
    source table and a different NLP algorithm. Pipeline 1 uses a
    keyword_search_extractor(), while pipeline 2 uses a named_entity_extractor.

    Verify that the output of each NLP algorithm only contains the
    TEXT_DOC_IDs from the resp. source table.
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
    Extraction(
        extractor=Branch(
            paths=[
                Pipeline(
                    steps=[
                        source_table_extractor(),
                        keyword_search_extractor(),
                    ]
                ),
                Pipeline(
                    steps=[
                        source_table_extractor(source_tables(2)),
                        named_entity_extractor(),
                    ]
                ),
            ]
        ),
        output=Output(db_schema=end2end_output_schema.name),
    ).run(
        pyexasol_connection,
        db_schema_name,
        language_alias,
    )

    for algo, expected in [
        ("KEYWORD_SEARCH", {1, 2}),
        ("NAMED_ENTITY", {3, 4}),
    ]:
        actual = {
            row[0]
            for row in pyexasol_connection.execute(
                f'SELECT DISTINCT "TEXT_DOC_ID" from OUTPUT_SCHEMA.{algo}'
            )
        }
        assert actual <= expected, (
            f"Table OUTPUT_SCHEMA.{algo} contains"
            f" unexpected TEXT_DOC_IDs {actual - expected}"
        )
