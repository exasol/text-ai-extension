from exasol.analytics.sql_executor.pyexasol_impl import PyexasolSQLExecutor

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.impl.orchestrator import PythonExtractionRunner
from test.utils.helpers import source_table_extractor


def test_runner_run(
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    end2end_output_schema,
    bucketfs_conn_name,
    model_subdir,
    end2end_feature_extraction_model,
    end2end_feature_extraction_model_download,
    end2end_topic_model,
    end2end_topic_model_download,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES (1, 'This is a test.')"
    )
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES (2, 'Exasol is awesome.')"
    )
    runner = PythonExtractionRunner()
    extraction = Extraction(
        extractor=source_table_extractor(),
        output=Output(db_schema=end2end_output_schema.name),
    )
    sql_executor = PyexasolSQLExecutor(pyexasol_connection)
    runner.run(
        extraction,
        sql_executor,
        db_schema_name,
        language_alias,
    )
    document_count = pyexasol_connection.export_to_pandas(
        f"SELECT count(1) as document_count FROM {end2end_output_schema.fully_qualified}.DOCUMENTS"
    )
    assert document_count.squeeze() == 2
