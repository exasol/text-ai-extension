from collections.abc import (
    Callable,
    Generator,
)
from dataclasses import dataclass
from itertools import (
    chain,
    combinations,
)
from typing import (
    cast,
)

import pandas as pd
import pytest
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.python_query_handler_runner import (
    PythonQueryHandlerRunner,
)
from exasol.analytics.query_handler.query.select import LogSpan
from exasol.analytics.schema import (
    SchemaName,
    TableLikeNameImpl,
)
from exasol.analytics.sql_executor.pyexasol_impl import PyexasolSQLExecutor
from pyexasol import ExaConnection

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearchOutput,
)
from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityOutput,
)
from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import AlgoOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.orchestrator.python_extraction_runner import (
    query_handler_context,
    temporary_db_object_name_prefix,
)
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.input_output.list_input_output import (
    ListInputOutput,
)
from exasol.ai.text.impl.stage_graph.query_handlers.co_occurrence_sql_stage import (
    CoOccurrenceSQLStage,
)
from test.integration.with_db.utils.db_test_utils import (
    create_table,
    delete_table,
    populate_table,
)
from test.utils.dataframe_matcher import DataFrameMatcher
from test.utils.word_span_columns import WordSpanColumns

SEGMENT_VER = 10
SEGMENT_NAME = "SEG"

TOKEN_VIEW_NAME = "CO_OCCURRENCE"


@dataclass(frozen=True)
class _TokenizedKeywordOutput(KeywordSearchOutput):
    word_span: WordSpanColumns = WordSpanColumns.create(SEGMENT_NAME)


@dataclass(frozen=True)
class _TokenizedEntityOutput(NamedEntityOutput):
    word_span: WordSpanColumns = WordSpanColumns.create(SEGMENT_NAME)


@dataclass
class _Span:
    begin: int
    end: int


@dataclass
class _SegmentSpan(_Span):
    ver: int = 1


@dataclass
class _SpanSet:
    char_span: _Span
    word_span: _SegmentSpan


@dataclass
class _Keyword:
    keyword: str
    score: float
    locations: list[_SpanSet]


@dataclass
class _NamedEntity:
    entity_type: str
    score: float
    entity: str
    locations: list[_SpanSet]


@dataclass
class _TextResult:
    doc_id: int
    text_span: _Span
    keywords: list[_Keyword]
    geo_entities: list[_NamedEntity]
    bio_entities: list[_NamedEntity]


@pytest.fixture(params=[(1, 1), (1, 2)], ids=["same_doc_id", "diff_doc_ids"])
def doc_ids(request) -> tuple[int]:
    """
    We will test two scenarios. One is when two documents have the same DOC_ID. In
    this scenario we validate that the inputs are merged only on the doc_id, and not
    the span. In the second scenario the documents have different DOC_IDs. This is to
    validate that the result is not a Cartesian product, the inputs actually get merged
    on the doc_id.
    """
    return request.param


@pytest.fixture
def source_data(doc_ids) -> list[_TextResult]:
    return [
        _TextResult(
            doc_id=doc_ids[0],
            text_span=_Span(10000, 20000),
            keywords=[
                _Keyword(
                    keyword="species distribution",
                    score=0.91,
                    locations=[
                        _SpanSet(
                            char_span=_Span(13565, 13585),
                            word_span=_SegmentSpan(2260, 2262),
                        )
                    ],
                ),
                _Keyword(
                    keyword="loss of biodiversity",
                    score=0.88,
                    locations=[
                        _SpanSet(
                            char_span=_Span(16782, 16802),
                            word_span=_SegmentSpan(2797, 2800),
                        )
                    ],
                ),
                _Keyword(
                    keyword="deforestation",
                    score=0.95,
                    locations=[
                        _SpanSet(
                            char_span=_Span(14356, 14369),
                            word_span=_SegmentSpan(2392, 2393),
                        ),
                        _SpanSet(
                            char_span=_Span(18345, 18358),
                            word_span=_SegmentSpan(3056, 3057),
                        ),
                    ],
                ),
            ],
            geo_entities=[
                _NamedEntity(
                    entity="lowland rainforest",
                    entity_type="Ecosystem",
                    score=0.82,
                    locations=[
                        _SpanSet(
                            char_span=_Span(10857, 10875),
                            word_span=_SegmentSpan(1809, 1811),
                        ),
                        _SpanSet(
                            char_span=_Span(14385, 14403),
                            word_span=_SegmentSpan(2397, 2399),
                        ),
                    ],
                ),
                _NamedEntity(
                    entity="montane rainforest",
                    entity_type="Ecosystem",
                    score=0.77,
                    locations=[
                        _SpanSet(
                            char_span=_Span(11428, 11446),
                            word_span=_SegmentSpan(1904, 1906),
                        ),
                        _SpanSet(
                            char_span=_Span(16672, 16690),
                            word_span=_SegmentSpan(2778, 2780),
                        ),
                    ],
                ),
                _NamedEntity(
                    entity="Monteverde cloud forest",
                    entity_type="National Park",
                    score=0.89,
                    locations=[
                        _SpanSet(
                            char_span=_Span(14367, 14390),
                            word_span=_SegmentSpan(2394, 2397),
                        )
                    ],
                ),
            ],
            bio_entities=[
                _NamedEntity(
                    entity="mantled howler monkey",
                    entity_type="Mammals",
                    score=0.83,
                    locations=[
                        _SpanSet(
                            char_span=_Span(11764, 11785),
                            word_span=_SegmentSpan(1960, 1963),
                        )
                    ],
                ),
                _NamedEntity(
                    entity="resplendent quetzal",
                    entity_type="Birds",
                    score=0.88,
                    locations=[
                        _SpanSet(
                            char_span=_Span(10625, 10644),
                            word_span=_SegmentSpan(1770, 1772),
                        ),
                        _SpanSet(
                            char_span=_Span(16211, 16230),
                            word_span=_SegmentSpan(2701, 2703),
                        ),
                    ],
                ),
            ],
        ),
        _TextResult(
            doc_id=doc_ids[1],
            text_span=_Span(20000, 30000),
            keywords=[
                _Keyword(
                    keyword="palm oil production",
                    score=0.97,
                    locations=[
                        _SpanSet(
                            char_span=_Span(28785, 28804),
                            word_span=_SegmentSpan(4797, 4800),
                        ),
                        _SpanSet(
                            char_span=_Span(29213, 29232),
                            word_span=_SegmentSpan(4868, 4871),
                        ),
                    ],
                ),
                _Keyword(
                    keyword="logging",
                    score=0.71,
                    locations=[
                        _SpanSet(
                            char_span=_Span(25442, 25449),
                            word_span=_SegmentSpan(4240, 4241),
                        ),
                        _SpanSet(
                            char_span=_Span(25583, 25590),
                            word_span=_SegmentSpan(4263, 4264),
                        ),
                    ],
                ),
                _Keyword(
                    keyword="mining",
                    score=0.68,
                    locations=[
                        _SpanSet(
                            char_span=_Span(20850, 20856),
                            word_span=_SegmentSpan(3475, 3476),
                        ),
                        _SpanSet(
                            char_span=_Span(22615, 22621),
                            word_span=_SegmentSpan(3769, 3770),
                        ),
                    ],
                ),
            ],
            geo_entities=[
                _NamedEntity(
                    entity="Sumatra",
                    entity_type="Geographical Area",
                    score=0.99,
                    locations=[
                        _SpanSet(
                            char_span=_Span(22212, 22219),
                            word_span=_SegmentSpan(3702, 3703),
                        ),
                        _SpanSet(
                            char_span=_Span(25524, 25531),
                            word_span=_SegmentSpan(4254, 4255),
                        ),
                    ],
                ),
                _NamedEntity(
                    entity="Borneo",
                    entity_type="Geographical Area",
                    score=0.99,
                    locations=[
                        _SpanSet(
                            char_span=_Span(29190, 29196),
                            word_span=_SegmentSpan(4865, 4866),
                        ),
                        _SpanSet(
                            char_span=_Span(25468, 25474),
                            word_span=_SegmentSpan(4244, 4245),
                        ),
                    ],
                ),
            ],
            bio_entities=[
                _NamedEntity(
                    entity="orangutan",
                    entity_type="Mammals",
                    score=0.98,
                    locations=[
                        _SpanSet(
                            char_span=_Span(24016, 24025),
                            word_span=_SegmentSpan(4002, 4003),
                        )
                    ],
                ),
                _NamedEntity(
                    entity="sumatran tiger",
                    entity_type="Mammals",
                    score=0.97,
                    locations=[
                        _SpanSet(
                            char_span=_Span(22683, 22697),
                            word_span=_SegmentSpan(3780, 3782),
                        )
                    ],
                ),
            ],
        ),
    ]


def create_keyword_df(
    source_data: list[_TextResult], outputs: _TokenizedKeywordOutput
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                outputs.text_span.doc_id.name: result.doc_id,
                outputs.text_span.begin.name: result.text_span.begin,
                outputs.text_span.end.name: result.text_span.end,
                outputs.keyword.name: keyword.keyword,
                outputs.keyword_score.name: keyword.score,
                outputs.keyword_span.doc_id.name: result.doc_id,
                outputs.keyword_span.begin.name: loc.char_span.begin,
                outputs.keyword_span.end.name: loc.char_span.end,
                outputs.word_span.doc_id.name: result.doc_id,
                outputs.word_span.segment_id.name: SEGMENT_VER,
                outputs.word_span.begin.name: loc.word_span.begin,
                outputs.word_span.end.name: loc.word_span.end,
                outputs.error_message.name: None,
                outputs.setup.name: None,
            }
            for result in source_data
            for keyword in result.keywords
            for loc in keyword.locations
        ]
    )


def create_entity_df(
    source_data: list[_TextResult],
    outputs: _TokenizedEntityOutput,
    entity_selector: Callable[[_TextResult], list[_NamedEntity]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                outputs.text_span.doc_id.name: result.doc_id,
                outputs.text_span.begin.name: result.text_span.begin,
                outputs.text_span.end.name: result.text_span.end,
                outputs.entity_type.name: entity.entity_type,
                outputs.entity_score.name: entity.score,
                outputs.entity.name: entity.entity,
                outputs.entity_span.doc_id.name: result.doc_id,
                outputs.entity_span.begin.name: loc.char_span.begin,
                outputs.entity_span.end.name: loc.char_span.end,
                outputs.word_span.doc_id.name: result.doc_id,
                outputs.word_span.segment_id.name: SEGMENT_VER,
                outputs.word_span.begin.name: loc.word_span.begin,
                outputs.word_span.end.name: loc.word_span.end,
                outputs.error_message.name: None,
                outputs.setup.name: None,
            }
            for result in source_data
            for entity in entity_selector(result)
            for loc in entity.locations
        ]
    )


@pytest.fixture
def sql_executor(pyexasol_connection) -> PyexasolSQLExecutor:
    return PyexasolSQLExecutor(pyexasol_connection)


@pytest.fixture
def temp_prefix(sql_executor) -> str:
    return temporary_db_object_name_prefix(sql_executor)


@pytest.fixture
def text_output() -> TextOutput:
    return TextOutput()


@pytest.fixture
def text_span(text_output) -> SpanColumns:
    return text_output.text_span


@pytest.fixture
def keyword_data(db_schema_name, temp_prefix, text_span) -> AlgoIOData:
    return AlgoIOData(
        datasource=TableLikeNameImpl(
            f"{temp_prefix}_KEYWORDS", SchemaName(db_schema_name)
        ),
        columns=_TokenizedKeywordOutput(name="KEYWORD", text_span=text_span),
    )


@pytest.fixture
def geo_entity_data(db_schema_name, temp_prefix, text_span) -> AlgoIOData:
    return AlgoIOData(
        datasource=TableLikeNameImpl(
            f"{temp_prefix}_GEO_ENTITIES", SchemaName(db_schema_name)
        ),
        columns=_TokenizedEntityOutput(name="GEO", text_span=text_span),
    )


@pytest.fixture
def bio_entity_data(db_schema_name, temp_prefix, text_span) -> AlgoIOData:
    return AlgoIOData(
        datasource=TableLikeNameImpl(
            f"{temp_prefix}_BIO_ENTITIES", SchemaName(db_schema_name)
        ),
        columns=_TokenizedEntityOutput(name="BIO", text_span=text_span),
    )


@pytest.fixture
def all_columns(
    keyword_data, geo_entity_data, bio_entity_data
) -> tuple[_TokenizedKeywordOutput, _TokenizedEntityOutput, _TokenizedEntityOutput]:
    return (
        cast(_TokenizedKeywordOutput, keyword_data.columns),
        cast(_TokenizedEntityOutput, geo_entity_data.columns),
        cast(_TokenizedEntityOutput, bio_entity_data.columns),
    )


@pytest.fixture
def keyword_df(source_data, all_columns) -> pd.DataFrame:
    keyword_columns, _, _ = all_columns
    return create_keyword_df(source_data, keyword_columns)


@pytest.fixture
def geo_entity_df(source_data, all_columns) -> pd.DataFrame:
    _, geo_entity_columns, _ = all_columns
    return create_entity_df(
        source_data, geo_entity_columns, lambda res: res.geo_entities
    )


@pytest.fixture
def bio_entity_df(source_data, all_columns) -> pd.DataFrame:
    _, _, bio_entity_columns = all_columns
    return create_entity_df(
        source_data, bio_entity_columns, lambda res: res.bio_entities
    )


def add_distance_column(
    df: pd.DataFrame, span1: SpanColumns, span2: SpanColumns
) -> None:
    *_, begin1, end1 = span1.column_names
    *_, begin2, end2 = span2.column_names
    metric_name = f"DISTANCE_{span1.name}_TO_{span2.name}"
    df.loc[df[begin2] >= df[end1], metric_name] = df[begin2] - df[end1]


def add_before_column(df: pd.DataFrame, span1: SpanColumns, span2: SpanColumns) -> None:
    *_, begin1, end1 = span1.column_names
    *_, begin2, end2 = span2.column_names
    metric_name = f"{span1.name}_BEFORE_{span2.name}"
    df[metric_name] = df[begin2] >= df[end1]


def add_contains_column(
    df: pd.DataFrame, span1: SpanColumns, span2: SpanColumns
) -> None:
    *_, begin1, end1 = span1.column_names
    *_, begin2, end2 = span2.column_names
    metric_name = f"{span1.name}_CONTAINS_{span2.name}"
    df[metric_name] = (df[begin1] <= df[begin2]) & (df[end1] >= df[end2])


def add_span_metric_columns(
    df: pd.DataFrame,
    spans: list[SpanColumns],
    metric_func: Callable[[pd.DataFrame, SpanColumns, SpanColumns], None],
) -> None:
    for span1, span2 in combinations(spans, 2):
        metric_func(df, span1, span2)
        metric_func(df, span2, span1)


def get_char_spans(
    keyword_outputs: _TokenizedKeywordOutput,
    geo_entity_outputs: _TokenizedEntityOutput,
    bio_entity_outputs: _TokenizedEntityOutput,
) -> list[SpanColumns]:
    return [
        keyword_outputs.keyword_span,
        SpanColumns.create(
            f"{geo_entity_outputs.name}_{geo_entity_outputs.entity_span.name}"
        ),
        SpanColumns.create(
            f"{bio_entity_outputs.name}_{bio_entity_outputs.entity_span.name}"
        ),
    ]


def get_word_spans(
    keyword_outputs: _TokenizedKeywordOutput,
    geo_entity_outputs: _TokenizedEntityOutput,
    bio_entity_outputs: _TokenizedEntityOutput,
) -> list[SpanColumns]:
    return [
        WordSpanColumns.create(
            f"{keyword_outputs.name}_{keyword_outputs.word_span.name}"
        ),
        WordSpanColumns.create(
            f"{geo_entity_outputs.name}_{geo_entity_outputs.word_span.name}"
        ),
        WordSpanColumns.create(
            f"{bio_entity_outputs.name}_{bio_entity_outputs.word_span.name}"
        ),
    ]


@pytest.fixture
def expected_df(
    keyword_df, geo_entity_df, bio_entity_df, all_columns, text_output, text_span
) -> pd.DataFrame:
    keyword_columns, geo_entity_columns, bio_entity_columns = all_columns
    # Create a copy of the doc_id in each table and merge on it
    all_dfs = [keyword_df, geo_entity_df, bio_entity_df]
    common_doc_id = "COMMON_DOC_ID"
    for df in all_dfs:
        df[common_doc_id] = df[text_span.doc_id.name]
    tmp_result_df = geo_entity_df.merge(
        bio_entity_df,
        on=common_doc_id,
        suffixes=(f"__{geo_entity_columns.name}", f"__{bio_entity_columns.name}"),
    )
    # Put the suffixes in front
    tmp_result_df.columns = [
        "_".join(col.split("__")[::-1]) for col in tmp_result_df.columns
    ]
    result_df = keyword_df.merge(tmp_result_df, on=common_doc_id)
    result_df.rename(
        {
            col: f"{keyword_columns.name}_{col}"
            for col in [
                *text_span.column_names,
                *keyword_columns.word_span.column_names,
                keyword_columns.error_message.name,
                keyword_columns.setup.name,
            ]
        },
        axis=1,
        inplace=True,
    )
    result_df.rename({common_doc_id: text_span.doc_id.name}, axis=1, inplace=True)
    # Now, all the output columns are sorted.
    for df in all_dfs:
        df.drop(common_doc_id, axis=1, inplace=True)

    # Create extra columns based on spans
    span_groups = [get_char_spans(*all_columns), get_word_spans(*all_columns)]
    for spans in span_groups:
        add_span_metric_columns(result_df, spans, add_distance_column)
        add_span_metric_columns(result_df, spans, add_before_column)
        add_span_metric_columns(result_df, spans, add_contains_column)

    result_df[text_output.error_message.name] = None
    result_df[text_output.setup.name] = None

    # Sort by the document and the character spans
    keyword_text_span = SpanColumns.create(f"{keyword_columns.name}_{text_span.name}")
    order = list(
        chain(*[span.column_names for span in [keyword_text_span] + span_groups[0]])
    )
    result_df.sort_values(by=order, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def _populate_table(
    data: AlgoIOData, content: pd.DataFrame, pyexasol_connection: ExaConnection
) -> None:
    columns = cast(AlgoOutput, data.columns)
    content = content.copy(deep=False)
    content[columns.error_message.name] = None
    content[columns.setup.name] = None
    populate_table(data, content, pyexasol_connection)


@pytest.fixture
def setup_co_occurrence_test(
    keyword_data,
    geo_entity_data,
    bio_entity_data,
    keyword_df,
    geo_entity_df,
    bio_entity_df,
    pyexasol_connection,
) -> Generator[None, None, None]:
    all_inputs = [keyword_data, geo_entity_data, bio_entity_data]
    all_frames = [keyword_df, geo_entity_df, bio_entity_df]
    for inp in all_inputs:
        delete_table(inp, pyexasol_connection)
    try:
        for inp, df in zip(all_inputs, all_frames):
            create_table(inp, pyexasol_connection)
            _populate_table(inp, df, pyexasol_connection)
        yield
    finally:
        for inp in all_inputs:
            delete_table(inp, pyexasol_connection)


@pytest.fixture
def query_handler_runner_factory(
    db_schema_name,
    sql_executor,
    temp_prefix,
    keyword_data,
    geo_entity_data,
    bio_entity_data,
) -> Callable[[], PythonQueryHandlerRunner]:
    def factory() -> PythonQueryHandlerRunner:
        context = query_handler_context(db_schema_name, temp_prefix)
        output_view = TableLikeNameImpl(TOKEN_VIEW_NAME, SchemaName(db_schema_name))
        log_span = LogSpan(name="test")
        stage = CoOccurrenceSQLStage(output_view=output_view, log_span=log_span)
        stage_input = SQLStageQueryHandlerInput(
            sql_stage_inputs=[
                ListInputOutput(
                    input_output_list=[keyword_data, geo_entity_data, bio_entity_data]
                )
            ],
            result_bucketfs_location=None,
        )  # type: ignore

        return PythonQueryHandlerRunner(
            sql_executor=sql_executor,
            top_level_query_handler_context=context,
            parameter=stage_input,
            query_handler_factory=stage.create_query_handler,
        )

    return factory


def test_query_handler(
    setup_database,
    db_schema_name,
    pyexasol_connection,
    setup_co_occurrence_test,
    query_handler_runner_factory,
    text_span,
    all_columns,
    keyword_data,
    expected_df,
):
    query_handler_runner = query_handler_runner_factory()
    query_handler_runner.run()

    keyword_text_span = SpanColumns.create(
        f"{keyword_data.columns.name}_{text_span.name}"
    )
    char_spans = get_char_spans(*all_columns)
    order = ", ".join(
        chain(*[span.quoted_column_names for span in [keyword_text_span] + char_spans])
    )
    query = f'SELECT * FROM "{db_schema_name}"."{TOKEN_VIEW_NAME}" ORDER BY {order}'
    co_occurrence_df = pyexasol_connection.export_to_pandas(query)
    matcher = DataFrameMatcher(co_occurrence_df, expected_df)
    assert matcher, matcher.diff()
