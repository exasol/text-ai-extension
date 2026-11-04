from unittest.mock import patch

import pandas as pd
import pytest

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.keywords import PatternRankKeywordExtractor
from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import KeywordOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec
from test.integration.utils.db_test_utils import (
    common_checks,
    get_input_df,
)

TEXT_COLUMN = "text"
ERROR_MESSAGE_COLUMN = "error_message"


@pytest.fixture
def text_span() -> SpanColumns:
    return SpanColumns.create(TEXT_COLUMN)


@pytest.fixture
def keyword_inputs(text_span) -> TextInput:
    return TextInput(text=varchar_column_spec(TEXT_COLUMN), text_span=text_span)


@pytest.fixture
def keyword_outputs(text_span) -> KeywordOutput:
    return KeywordOutput(
        error_message=varchar_column_spec(ERROR_MESSAGE_COLUMN), text_span=text_span
    )


@pytest.fixture
def input_df(keyword_test_cases, text_span) -> pd.DataFrame:
    return get_input_df(keyword_test_cases, TEXT_COLUMN, text_span)


@pytest.mark.parametrize("max_cases", [1, -1], ids=["single_doc", "all_docs"])
def test_keyword_extractor(
    setup_no_database,
    spacy_pipeline,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_location,
    keyword_test_cases,
    empty_exa,
    max_cases,
    text_span,
    keyword_inputs,
    keyword_outputs,
):

    extractor = PatternRankKeywordExtractor(
        model_name=feature_extraction_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        inputs=keyword_inputs,
        outputs=keyword_outputs,
    )
    test_cases = keyword_test_cases[:max_cases] if max_cases > 0 else keyword_test_cases
    input_df = get_input_df(test_cases, TEXT_COLUMN, text_span)

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = feature_extraction_model_location
        output_df = extractor.run_impl(input_df, empty_exa)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )
    # Check that every gold keyword is contained in at least one returned keyword.
    text_id_column, text_begin_column, _ = extractor.outputs.text_span.column_names
    for text_case in test_cases:
        output_case_df = output_df.loc[
            (output_df[text_id_column] == text_case.text_doc_id)
            & (output_df[text_begin_column] == text_case.text_char_begin)
        ]
        assert all(
            output_case_df[extractor.keyword_column].str.contains(gold_kw).any()
            for gold_kw in text_case.keywords
        )


def test_vectorizer_params(
    setup_no_database,
    spacy_pipeline,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_location,
    keyword_test_cases,
    input_df,
    empty_exa,
    text_span,
    keyword_inputs,
    keyword_outputs,
):

    stop_word = keyword_test_cases[0].keywords[0]
    vec_kwargs = {"stop_words": [stop_word]}
    extractor = PatternRankKeywordExtractor(
        model_name=feature_extraction_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        inputs=keyword_inputs,
        outputs=keyword_outputs,
        vec_kwargs=vec_kwargs,
    )

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = feature_extraction_model_location
        output_df = extractor.run_impl(input_df, empty_exa)
    output_df.reset_index(drop=True, inplace=True)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )
    # Check that the returned keywords do not contain the first gold keyword of the
    # first text, as it is supposed to be excluded from the output by declaring it a
    # stop word.
    assert not output_df[extractor.keyword_column].str.contains(stop_word).any()


def test_extractor_params(
    setup_no_database,
    spacy_pipeline,
    model_subdir,
    feature_extraction_model,
    feature_extraction_model_location,
    input_df,
    empty_exa,
    text_span,
    keyword_inputs,
    keyword_outputs,
):

    top_n = 3
    kbx_kwargs = {"top_n": top_n}
    extractor = PatternRankKeywordExtractor(
        model_name=feature_extraction_model,
        bucketfs_conn_name="",
        sub_dir=model_subdir,
        inputs=keyword_inputs,
        outputs=keyword_outputs,
        kbx_kwargs=kbx_kwargs,
    )

    with patch(
        "exasol.ai.text.impl.utils.transformers_utils.get_bucketfs_location"
    ) as mock_bfs_location:
        mock_bfs_location.return_value = feature_extraction_model_location
        output_df = extractor.run_impl(input_df, empty_exa)

    common_checks(
        input_df=input_df,
        output_df=output_df,
        text_column=TEXT_COLUMN,
        error_message_column=ERROR_MESSAGE_COLUMN,
        text_span=text_span,
    )
    # Check that the number of unique keywords for each text is exactly as specified.
    count_df = output_df.groupby(extractor.outputs.text_span.column_names).count()
    assert (count_df[TEXT_COLUMN] == top_n).all()
