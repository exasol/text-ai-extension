from typing import Any

from exasol.analytics.schema import (
    SchemaName,
    UDFNameImpl,
)
from pyexasol import ExaConnection

from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from test.integration.with_db.utils.db_test_utils import (
    CommonOutputs,
    common_checks,
    run_udf_with_text_input,
)

UPPER_SCORE_THRESHOLD = 0.5
LOWER_SCORE_THRESHOLD = 0.2


class _Outputs(CommonOutputs):
    """Indices of the fields in the UDF output"""

    topic = 3
    topic_score = 4
    topic_rank = 5


def create_and_run_udf(
    db_schema_name: str,
    language_alias: str,
    pyexasol_connection: ExaConnection,
    topic_test_cases,
    bucketfs_conn_name: str,
    model_subdir: str,
    topic_model: str,
    topic_list: list[str],
) -> list[tuple[Any]]:

    udf_name = UDFNameImpl("test_hft_topic", SchemaName(db_schema_name))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias=language_alias)
    try:
        # Configure the task.
        test_runner = UDFRunner(
            algorithm={
                "algorithm_name": "huggingface-zero-shot-classification",
                "model_name": topic_model,
                "bucketfs_conn_name": bucketfs_conn_name,
                "sub_dir": model_subdir,
                "topics": topic_list,
            },
            batch_size=10,
        )

        # Run the task and return the data.
        return run_udf_with_text_input(
            pyexasol_connection, udf_factory, test_runner, topic_test_cases
        )
    finally:
        # Delete the UDF.
        delete_udf_query = udf_factory.delete_udf_query()
        pyexasol_connection.execute(delete_udf_query)


def test_topic_classification(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    topic_test_cases,
    bucketfs_conn_name,
    model_subdir,
    topic_list,
    topic_model,
    topic_model_download,
):
    """
    Tests that the topic classification algorithm ranks the correct topic as the top
    one and assigns a high score to it. All other topics should get a low score.
    """
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        topic_test_cases,
        bucketfs_conn_name,
        model_subdir,
        topic_model,
        topic_list,
    )
    common_checks(topic_test_cases, output)

    # Check that the correct topic was ranked at the top.
    top_rows = {
        (row[_Outputs.text_doc_id], row[_Outputs.topic])
        for row in output
        if (
            row[_Outputs.topic_score] > UPPER_SCORE_THRESHOLD
            and row[_Outputs.topic_rank] == 1
        )
    }
    expected_rows = {
        (text_case.text_doc_id, text_case.topic) for text_case in topic_test_cases
    }
    assert top_rows == expected_rows

    # Check that all other topics have a low score.
    highest_other_score = max(
        row[_Outputs.topic_score] for row in output if row[_Outputs.topic_rank] > 1
    )
    assert highest_other_score <= LOWER_SCORE_THRESHOLD
