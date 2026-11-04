import pytest
from exasol.analytics.schema import (
    SchemaName,
    UDFNameImpl,
)
from pyexasol import ExaConnection

from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.utils.sql_utils import sql_text_value
from test.integration.with_db.utils.db_test_utils import common_checks

PHRASE_COLUMN = "phrase"


def create_and_run_udf(
    db_schema_name: str,
    language_alias: str,
    pyexasol_connection: ExaConnection,
    text_search_test_cases,
    ignore_spaces: bool = True,
):

    udf_name = UDFNameImpl("test_tokenized_search", SchemaName(db_schema_name))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias=language_alias)
    try:
        # Configure the task.
        test_runner = UDFRunner(
            algorithm={
                "algorithm_name": "token-match",
                "phrase_column": PHRASE_COLUMN,
                "ignore_spaces": ignore_spaces,
            },
            batch_size=10,
        )

        # Create a UDF for the task.
        create_udf_query = udf_factory.create_udf_query(test_runner)
        pyexasol_connection.execute(create_udf_query)

        # Prepare the UDF input.
        values = ", ".join(
            f"({test_case.text_doc_id}, "
            f"{test_case.text_char_begin}, "
            f"{test_case.text_char_begin + len(test_case.text)}, "
            f"{sql_text_value(test_case.text)}, "
            f"{sql_text_value(phrase.phrase)})"
            for test_case in text_search_test_cases
            for phrase in test_case.phrases_occurrences
        )

        # Run the UDF and return the results.
        run_udf_query = (
            f"SELECT {udf_factory.udf_name.fully_qualified}"
            "(t.text_doc_id, t.text_char_begin, t.text_char_end, "
            f"t.text, t.{PHRASE_COLUMN}) "
            f"FROM (VALUES {values} "
            "AS t(text_doc_id, text_char_begin, text_char_end, text, "
            f"{PHRASE_COLUMN}))"
        )
        return pyexasol_connection.execute(run_udf_query).fetchall()

    finally:
        # Delete the UDF.
        delete_udf_query = udf_factory.delete_udf_query()
        pyexasol_connection.execute(delete_udf_query)


@pytest.mark.parametrize("ignore_spaces", [True, False])
def test_text_search(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    text_search_test_cases,
    ignore_spaces,
):
    """
    The test validates the tokenized text search called from a UDF. In this test
    we expect an exact output.
    """

    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        text_search_test_cases,
        ignore_spaces=ignore_spaces,
    )
    common_checks(text_search_test_cases, output)

    expected_output = {
        (
            test_case.text_doc_id,
            test_case.text_char_begin,
            test_case.text_char_begin + len(test_case.text),
            phrase.phrase,
            test_case.text_doc_id,
            test_case.text_char_begin + loc,
            test_case.text_char_begin + loc + len(phrase.phrase),
        )
        for test_case in text_search_test_cases
        for phrase in test_case.phrases_occurrences
        for loc in phrase.occurrences
        if ignore_spaces or ("\n" not in test_case.text[loc : loc + len(phrase.phrase)])
    }
    assert {row[:-2] for row in output} == expected_output
