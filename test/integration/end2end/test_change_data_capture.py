from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.helpers import source_table_extractor


def test_change_data_capture(
    e2e_scenario,  # parametrized, iterates through a list of NLP extractors
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    end2end_output_schema,
    model_subdir,
    end2end_topic_model,
    end2end_topic_model_download,
    end2end_feature_extraction_model,
    end2end_feature_extraction_model_download,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    """
    This end-to-end test verifies change data capture (CDC), i.e. that
    additional input data is properly detected and results in additional
    output data.

    The test uses the fixture e2e_scenario, parametrized to iterate through
    all relevant NLP extractors with resp. SQL output tables.

    Test strategy:

    * Insert initial data into Source Table
    * Run extraction and verify initial results to be available
    * Add data from another document to the Source Table
    * Run extraction again and verify that output table contains
      additional results from new document
    """

    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES"
        " (1, 'This is a test.'),"
        " (2, 'Exasol is awesome.')"
    )
    steps = [source_table_extractor()]
    if e2e_scenario.extractor_factory:
        steps.append(e2e_scenario.extractor_factory())
    extraction = Extraction(
        extractor=Pipeline(steps=steps),
        output=Output(db_schema=end2end_output_schema.name),
    )

    def run_and_expect_doc_id(expected_doc_id: int) -> None:
        extraction.run(pyexasol_connection, db_schema_name, language_alias)
        rows = pyexasol_connection.execute(
            "SELECT DISTINCT TEXT_DOC_ID FROM "
            f"{e2e_scenario.output_table.fully_qualified}"
        )
        doc_ids = [r[0] for r in rows]
        assert expected_doc_id in doc_ids, f"{e2e_scenario.explain()}"

    # run first time and expect results
    run_and_expect_doc_id(2)

    # add additional input data
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES"
        " (3, 'Exasol is even better.')"
    )

    # expect results from additional document
    run_and_expect_doc_id(3)
