from exasol.analytics.schema import (
    SchemaName,
    UDFNameImpl,
)
from pyexasol import ExaConnection

from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearch,
)
from exasol.ai.text.impl.nlp.algorithms.keywords.pattern_rank_keyword_extractor import (
    PatternRankKeywordExtractor,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)
from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.utils.sql_utils import sql_text_value
from test.integration.with_db.utils.db_test_utils import (
    CommonOutputs,
    common_checks,
)

KEYWORD_COLUMN = "my_keyword"


class _Outputs(CommonOutputs):
    """Indices of the fields in the UDF output"""

    keyword = 3
    keyword_score = 4
    keyword_doc_id = 5
    keyword_char_begin = 6
    keyword_char_end = 7
    setup = -1


def create_and_run_udf(
    db_schema_name: str,
    language_alias: str,
    pyexasol_connection: ExaConnection,
    keyword_test_cases,
    bucketfs_conn_name,
    model_subdir,
    feature_extraction_model,
    setup,
):

    udf_name = UDFNameImpl("test_pattern_rank_search", SchemaName(db_schema_name))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias=language_alias)
    try:
        # Configure the task.
        keywords_extractor = PatternRankKeywordExtractor(
            model_name=feature_extraction_model,
            bucketfs_conn_name=bucketfs_conn_name,
            sub_dir=model_subdir,
        )
        text_searcher = TokenizedTextSearch()
        keyword_searcher = KeywordSearch(
            algorithms=(keywords_extractor, text_searcher), setup=setup
        )
        test_runner = UDFRunner(algorithm=keyword_searcher, batch_size=10)

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


def test_keyword_extractor_with_spans(
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
    This test validates a pipline made of a keyword extractor and a text search.
    The output is expected to include the keywords and their spans. Since we don't
    know exactly what keywords we are going to get from the model we will do only
    basic checks.
    """

    setup = '{"kw-searcher": "my-searcher"}'
    output = create_and_run_udf(
        db_schema_name,
        language_alias,
        pyexasol_connection,
        keyword_test_cases,
        bucketfs_conn_name,
        model_subdir,
        feature_extraction_model,
        setup,
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

    # Validate keyword spans
    assert all(
        row[_Outputs.keyword_doc_id] == row[_Outputs.text_doc_id] for row in output
    )
    assert all(row[_Outputs.keyword_char_begin] is not None for row in output)
    assert all(row[_Outputs.keyword_char_end] is not None for row in output)

    # Validate the setup
    assert all(row[_Outputs.setup] == setup for row in output)
