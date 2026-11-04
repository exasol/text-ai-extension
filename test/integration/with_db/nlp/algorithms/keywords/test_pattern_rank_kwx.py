from typing import Any

from exasol.analytics.schema import (
    SchemaName,
    UDFNameImpl,
)
from pyexasol import ExaConnection

from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.utils.sql_utils import sql_text_value
from test.integration.with_db.utils.db_test_utils import (
    CommonOutputs,
    common_checks,
)


class _Outputs(CommonOutputs):
    """Indices of the fields in the UDF output"""

    keyword = 3
    keyword_score = 4


def create_and_run_udf(
    db_schema_name: str,
    language_alias: str,
    pyexasol_connection: ExaConnection,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    vec_kwargs: dict[str, Any] | None = None,
    kbx_kwargs: dict[str, Any] | None = None,
):

    udf_name = UDFNameImpl("test_pattern_rank", SchemaName(db_schema_name))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias=language_alias)
    try:
        # Configure the task.
        test_runner = UDFRunner(
            algorithm={
                "algorithm_name": "pattern-rank-keyword-extractor",
                "model_name": feature_extraction_model,
                "bucketfs_conn_name": bucketfs_conn_name,
                "sub_dir": model_subdir,
                "vec_kwargs": vec_kwargs or {},
                "kbx_kwargs": kbx_kwargs or {},
            },
            batch_size=10,
        )

        # Create a UDF for the task.
        create_udf_query = udf_factory.create_udf_query(test_runner)
        pyexasol_connection.execute(create_udf_query)

        # Prepare the UDF input.
        values = ", ".join(
            f"({text_case.text_doc_id}, "
            f"{text_case.text_char_begin}, "
            f"{text_case.text_char_begin + len(text_case.text)}, "
            f"{sql_text_value(text_case.text)})"
            for text_case in keyword_test_cases
        )

        # Run the UDF and return the results.
        run_udf_query = (
            f"SELECT {udf_factory.udf_name.fully_qualified}"
            "(t.text_doc_id, t.text_char_begin, t.text_char_end, t.text) "
            f"FROM (VALUES {values} "
            "AS t(text_doc_id, text_char_begin, text_char_end, text))"
        )
        return pyexasol_connection.execute(run_udf_query).fetchall()

    finally:
        # Delete the UDF.
        delete_udf_query = udf_factory.delete_udf_query()
        pyexasol_connection.execute(delete_udf_query)


def test_keyword_extractor(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_download,
):
    """
    The test validates the keyword extractor called from a UDF. Since we don't
    know exactly what keywords we are going to get from the model we will do only
    basic checks.
    """

    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        keyword_test_cases,
        bucketfs_conn_name,
        model_subdir,
        feature_extraction_model,
    )
    common_checks(keyword_test_cases, output)
    for text_case in keyword_test_cases:
        # Check that every gold keyword is contained in at least one returned keyword.
        assert all(
            any(
                kw in row[_Outputs.keyword]
                for row in output
                if (row[_Outputs.text_doc_id] == text_case.text_doc_id)
                and (row[_Outputs.text_char_begin] == text_case.text_char_begin)
            )
            for kw in text_case.keywords
        )


def test_vectorizer_params(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_download,
):
    """
    This test validates passing vectorizer parameters to the keyword extractor that
    we call from a UDF.
    """
    stop_word = keyword_test_cases[0].keywords[0]
    vec_kwargs = {"stop_words": [stop_word]}
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        keyword_test_cases,
        bucketfs_conn_name,
        model_subdir,
        feature_extraction_model,
        vec_kwargs=vec_kwargs,
    )
    common_checks(keyword_test_cases, output)
    # Check that the returned keywords do not contain the first gold keyword of the
    # first text, as it is supposed to be excluded from the output by declaring it a
    # stop word.
    assert all(stop_word not in row[_Outputs.keyword] for row in output)


def test_extractor_params(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_download,
):
    """
    This test validates passing KeyBERT parameters to the keyword extractor that
    we call from a UDF.
    """
    top_n = 3
    kbx_kwargs = {"top_n": top_n}
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        keyword_test_cases,
        bucketfs_conn_name,
        model_subdir,
        feature_extraction_model,
        kbx_kwargs=kbx_kwargs,
    )
    common_checks(keyword_test_cases, output)
    # Check that the number of unique keywords for each text is exactly as specified.
    for text_case in keyword_test_cases:
        keywords = {
            row[_Outputs.keyword]
            for row in output
            if (row[_Outputs.text_doc_id] == text_case.text_doc_id)
            and (row[_Outputs.text_char_begin] == text_case.text_char_begin)
        }
        assert len(keywords) == top_n


def test_vectorizer_error(
    setup_database,
    pyexasol_connection,
    db_schema_name,
    language_alias,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_download,
):
    """
    This test validates error propagation in a scenario when a keyword extractor is
    misconfigured.
    """

    # Let's provide an invalid parameter that should trigger an error.
    vec_kwargs = {"max_df": -1}
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        keyword_test_cases,
        bucketfs_conn_name,
        model_subdir,
        feature_extraction_model,
        vec_kwargs=vec_kwargs,
    )
    # Check for the presence of an error message.
    common_checks(keyword_test_cases, output, no_error=False)
