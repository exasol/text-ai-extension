from typing import cast
from unittest.mock import patch

import spacy
from exasol_udf_mock_python.column import Column as MockColumn
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol.ai.text.extractors import PatternRankKeywordSettings
from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearch,
)
from exasol.ai.text.impl.nlp.algorithms.keywords import PatternRankKeywordExtractor
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from test.integration.with_db.utils.db_test_utils import (
    CommonOutputs,
    common_checks,
)


class _Outputs(CommonOutputs):
    """Indices of the fields in the UDF output"""

    keyword = 3
    keyword_score = 4
    keyword_doc_id = 5
    keyword_char_begin = 6
    keyword_char_end = 7
    setup = -1


def create_meta_data(keyword_searcher: KeywordSearch):
    in_text_span = [
        MockColumn(col, int, "INTEGER")
        for col in keyword_searcher.inputs.text_span.column_names
    ]
    in_text = MockColumn(keyword_searcher.inputs.text.name, str, "VARCHAR(2000000)")
    out_text_span = [
        MockColumn(col, int, "INTEGER")
        for col in keyword_searcher.outputs.text_span.column_names
    ]
    out_keyword = MockColumn(
        keyword_searcher.outputs.keyword.name, str, "VARCHAR(2000000)"
    )
    out_keyword_score = MockColumn(
        keyword_searcher.outputs.keyword_score.name, float, "DOUBLE PRECISION"
    )
    out_keyword_span = [
        MockColumn(col, int, "INTEGER")
        for col in keyword_searcher.outputs.keyword_span.column_names
    ]
    out_error_message = MockColumn(
        keyword_searcher.error_message_column, str, "VARCHAR(2000000)"
    )
    out_setup = MockColumn(keyword_searcher.setup_column, str, "VARCHAR(2000000)")
    return MockMetaData(
        None,
        input_type="SET",
        input_columns=in_text_span + [in_text],
        output_type="EMITS",
        output_columns=out_text_span
        + [out_keyword, out_keyword_score]
        + out_keyword_span
        + [out_error_message, out_setup],
    )


def create_runner(
    model_subdir: str, feature_extraction_model: str, setup: str | None
) -> UDFRunner:
    keywords_extractor = PatternRankKeywordExtractor(
        model_name=feature_extraction_model, bucketfs_conn_name="", sub_dir=model_subdir
    )
    text_searcher = TokenizedTextSearch()
    keyword_searcher = KeywordSearch(
        algorithms=(keywords_extractor, text_searcher), setup=setup
    )
    runner = UDFRunner(algorithm=keyword_searcher, batch_size=10)

    runner_json = f"{runner.model_dump_json(serialize_as_any=True)}"
    return UDFRunner.model_validate_json(runner_json)


def test_keyword_extractor(
    setup_no_database,
    spacy_pipeline,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_location,
    keyword_test_cases,
    empty_exa,
):
    """
    This test tests a pipline algorithm, and unlike other "no-db" tests it closely
    mimics the execution of the algorithm in a database. The algorithm is loaded
    into a UDFRunner. The runner is serialised into json and then de-serialised back
    into an UDFRunner object as it would be in a UDF.
    """

    setup = '{"kw-searcher": "my-searcher"}'
    runner = create_runner(
        model_subdir=model_subdir,
        feature_extraction_model=feature_extraction_model,
        setup=setup,
    )
    input_data = [
        (
            text_case.text_doc_id,
            text_case.text_char_begin,
            text_case.text_char_begin + len(text_case.text),
            text_case.text,
        )
        for text_case in keyword_test_cases
    ]
    meta_data = create_meta_data(cast(KeywordSearch, runner.algorithm))
    ctx = StandaloneMockContext(input_data, meta_data)

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = feature_extraction_model_location
        runner.run(ctx, empty_exa)
        output = ctx.output

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


def test_keywords_vectorizer_default_pattern(setup_no_database, spacy_pipeline):
    settings = PatternRankKeywordSettings()
    keywords_extractor = PatternRankKeywordExtractor(
        model_name="",
        bucketfs_conn_name="",
        sub_dir="",
        vec_kwargs={"pos_pattern": settings.pos_pattern},
    )
    vectorizer = keywords_extractor.create_vectorizer()
    texts = [
        "This was a eye-opening presentation.",
        "My yummy peanut-caramel-covered pizza is delicious.",
        "My yummy peanut-jelly-covered pizza is delicious.",
        "My jelly sandwich is delicious.",
        "My wi-fi doesn't work.",
        "My wi-fi is broken.",
        "I have wi-fi problems.",
        "I have a slow wi-fi.",
        "I think this is a hardware issue.",
        "I covered our base.",
        "The movie was watched by millions.",
        "The Samsung Galaxy 21 is cool.",
        "Angela Merkel was chancelor of Germany.",
    ]
    nlp = spacy.load(vectorizer.spacy_pipeline, exclude=vectorizer.spacy_exclude)
    parsed = nlp.pipe(texts)
    for parse in parsed:
        print("input", " ".join([f"{token.text}/{token.tag_}" for token in parse]))
    vectorizer.fit(texts)
    result = vectorizer.get_feature_names_out()
    expected_keywords = {
        "angela merkel",
        "base",
        "cool",
        "delicious",
        "eye - opening presentation",
        "chancelor of germany",
        "hardware issue",
        "samsung galaxy 21",
        "jelly sandwich",
        "movie",
        "slow wi - fi",
        "watched by millions",
        "wi - fi",
        "wi - fi problems",
        "yummy peanut - caramel - covered pizza",
        "yummy peanut - jelly - covered pizza",
    }
    assert set(result) == expected_keywords
