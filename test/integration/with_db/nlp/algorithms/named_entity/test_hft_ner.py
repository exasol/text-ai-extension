from typing import Any

import pytest
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

MIN_SCORE = 0.85


class _Outputs(CommonOutputs):
    """Indices of the fields in the UDF output"""

    entity_type = 3
    entity_score = 4
    entity_text = 5
    entity_position = 7


def create_and_run_udf(
    db_schema_name: str,
    language_alias: str,
    pyexasol_connection: ExaConnection,
    named_entity_test_cases,
    bucketfs_conn_name: str,
    model_subdir: str,
    named_entity_model: str,
    strategy: str,
    ignore_labels: list[str] | None = None,
) -> list[tuple[Any]]:

    udf_name = UDFNameImpl("test_hft_ner", SchemaName(db_schema_name))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias=language_alias)
    try:
        # Configure the task.
        test_runner = UDFRunner(
            algorithm={
                "algorithm_name": "huggingface-token-classification",
                "model_name": named_entity_model,
                "bucketfs_conn_name": bucketfs_conn_name,
                "sub_dir": model_subdir,
                "aggregation_strategy": strategy,
                "ignore_labels": ignore_labels,
            },
            batch_size=10,
        )

        # Run the task and return the data.
        return run_udf_with_text_input(
            pyexasol_connection, udf_factory, test_runner, named_entity_test_cases
        )
    finally:
        # Delete the UDF.
        delete_udf_query = udf_factory.delete_udf_query()
        pyexasol_connection.execute(delete_udf_query)


@pytest.mark.parametrize("strategy", ["none", "simple", "max"])
def test_named_entity_recognition(
    setup_database,
    strategy,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    named_entity_test_cases,
    bucketfs_conn_name,
    model_subdir,
    named_entity_model,
    named_entity_model_download,
):
    """
    Tests that the named entity algorithm retrieves all the expected entities.
    Only entities with the score greater or equal to MIN_SCORE are counted.
    """
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        named_entity_test_cases,
        bucketfs_conn_name,
        model_subdir,
        named_entity_model,
        strategy,
    )
    common_checks(named_entity_test_cases, output)

    output_rows = {
        (
            row[_Outputs.text_doc_id],
            row[_Outputs.entity_type],
            row[_Outputs.entity_text],
            row[_Outputs.entity_position],
        )
        for row in output
        if row[_Outputs.entity_score] >= MIN_SCORE
    }

    expected_rows = {
        (
            text_case.text_doc_id,
            expected_entity.entity_type,
            expected_entity.entity_text,
            expected_entity.position + text_case.text_char_begin,
        )
        for text_case in named_entity_test_cases
        for strategy_entities in text_case.expected_entities
        for expected_entity in strategy_entities.entities
        if strategy in strategy_entities.strategies
    }
    assert expected_rows.issubset(output_rows)


@pytest.mark.parametrize("strategy", ["none", "simple", "max"])
def test_ner_ignore_labels(
    setup_database,
    strategy,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    named_entity_test_cases,
    bucketfs_conn_name,
    model_subdir,
    named_entity_model,
    named_entity_ignore_labels,
    named_entity_model_download,
):
    """
    Tests that the labels provided in the ignore_labels list are excluded from the
    output.
    """
    ignore_labels = named_entity_ignore_labels[strategy]
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        named_entity_test_cases,
        bucketfs_conn_name,
        model_subdir,
        named_entity_model,
        strategy,
        ignore_labels,
    )
    common_checks(named_entity_test_cases, output)

    # There should be some output for all test cases
    expected_text_ids = {text_case.text_doc_id for text_case in named_entity_test_cases}
    text_ids = {row[_Outputs.text_doc_id] for row in output}
    assert text_ids == expected_text_ids
    # But not including the labels designated to be ignored.
    all_labels = {row[_Outputs.entity_type] for row in output}
    assert not all_labels.intersection(ignore_labels)
