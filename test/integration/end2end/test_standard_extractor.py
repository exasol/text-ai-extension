from exasol.ai.text.extraction.abstract_extraction import (
    Output,
)
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.helpers import (
    defaults,
    source_table_extractor,
    standard_extractor,
)


def test_default_extractor(
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
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES"
        " (1, 'This is a test.'),"
        " (2, 'Exasol is awesome.')"
    )
    Extraction(
        extractor=Pipeline(steps=[source_table_extractor(), standard_extractor()]),
        output=Output(db_schema=end2end_output_schema.name),
        defaults=defaults(),
    ).run(
        pyexasol_connection,
        db_schema_name,
        language_alias,
    )

    for algo in ["KEYWORD", "TOPIC", "ENTITY"]:
        df = pyexasol_connection.export_to_pandas(
            f'SELECT "ERROR_MESSAGE" FROM OUTPUT_SCHEMA.{algo}'
        )
        assert df["ERROR_MESSAGE"].isnull().all()
