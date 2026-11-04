import contextlib
from typing import Any

import pyexasol
import pytest
from exasol.analytics.schema import (
    Column,
    ColumnName,
    SchemaName,
    Table,
    TableName,
    TableNameImpl,
)

from exasol.ai.text.extraction.abstract_extraction import Defaults
from exasol.ai.text.extractors import (
    HftNamedEntitySettings,
    HftTopicSettings,
    KeywordExtractor,
    KeywordSearchExtractor,
    NamedEntityExtractor,
    NameSelector,
    PatternRankKeywordSettings,
    SchemaSource,
    SourceTableExtractor,
    SpacyTokenizerSettings,
    StandardExtractor,
    TableSource,
    TokenizedTextSearchSettings,
    TokenizerExtractor,
    TopicClassifierExtractor,
)
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.default_models import (
    DEFAULT_FEATURE_EXTRACTION_MODEL,
    DEFAULT_NAMED_ENTITY_MODEL,
    DEFAULT_NLI_MODEL,
)
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.utils.sql_utils import (
    integer_column_spec,
    varchar_column_spec,
)

BUCKETFS_SUB_DIR = "model_sub_dir"

BUCKETFS_CONNECTION_NAME = "TEST_TEXT_AI_BFS_CONNECTION"


@contextlib.contextmanager
def not_raises(exception_class):
    def all_causes(ex):
        while ex.__cause__:
            ex = ex.__cause__
            yield f"\n- caused by {type(ex).__name__}: {ex}"

    try:
        yield
    except exception_class as ex:
        causes = "".join(all_causes(ex))
        raise pytest.fail(
            f"Test raised unexpected {exception_class.__name__}: {ex}{causes}"
        )


def model_repository():
    return BucketFSRepository(
        connection_name=BUCKETFS_CONNECTION_NAME, sub_dir=BUCKETFS_SUB_DIR
    )


def named_entty_model(name: str = DEFAULT_NAMED_ENTITY_MODEL):
    return HuggingFaceModel(repository=model_repository(), name=name)


def named_entity_settings():
    return HftNamedEntitySettings(model=named_entty_model())


def named_entity_extractor(
    named_entity_settings=named_entity_settings(),
    batch_size=10,
    parallelism_per_node=1,
):
    return NamedEntityExtractor(
        named_entity_settings=named_entity_settings,
        batch_size=batch_size,
        parallelism_per_node=parallelism_per_node,
    )


def standard_extractor() -> StandardExtractor:
    return StandardExtractor(
        topics=topics,
        entity_db_output="ENTITY",
        keyword_db_output="KEYWORD",
        topic_db_output="TOPIC",
        entity_tokenized_db_output="TOK_ENTITY",
        keyword_tokenized_db_output="TOK_KEYWORD",
    )


def defaults() -> Defaults:
    return Defaults(
        parallelism_per_node=1, batch_size=10, model_repository=model_repository()
    )


def feature_extraction_model(name: str = DEFAULT_FEATURE_EXTRACTION_MODEL):
    return HuggingFaceModel(repository=model_repository(), name=name)


def keyword_settings():
    return PatternRankKeywordSettings(ranking_model=feature_extraction_model())


def keyword_extractor(
    keyword_settings=keyword_settings(), batch_size=10
) -> KeywordExtractor:
    return KeywordExtractor(keyword_settings=keyword_settings, batch_size=batch_size)


def text_search_settings():
    return TokenizedTextSearchSettings()


def keyword_search_extractor(
    keyword_settings=keyword_settings(),
    text_search_settings=text_search_settings(),
    batch_size=10,
    parallelism_per_node=1,
):
    return KeywordSearchExtractor(
        keyword_settings=keyword_settings,
        text_search_settings=text_search_settings,
        batch_size=batch_size,
        parallelism_per_node=parallelism_per_node,
    )


def nli_model(name: str = DEFAULT_NLI_MODEL):
    return HuggingFaceModel(repository=model_repository(), name=name)


topics = ["urgent", "not urgent"]


def topic_settings() -> HftTopicSettings:
    return HftTopicSettings(ranking_model=nli_model(), topics=topics)


def topic_classifier_extractor(
    topic_settings=topic_settings(),
    batch_size=10,
    parallelism_per_node=1,
):
    return TopicClassifierExtractor(
        topic_settings=topic_settings,
        batch_size=batch_size,
        parallelism_per_node=parallelism_per_node,
    )


def tokenizer_settings() -> SpacyTokenizerSettings:
    return SpacyTokenizerSettings()


def tokenizer_extractor(settings=tokenizer_settings(), batch_size=10):
    return TokenizerExtractor(tokenizer_settings=settings, batch_size=batch_size)


def schema_sources(
    schema: SchemaName,
    table: TableName,
    text_column: ColumnName,
    key_column: ColumnName,
) -> list[SchemaSource]:
    return [
        SchemaSource(
            db_schema=NameSelector(pattern=schema.name),
            tables=[
                TableSource(
                    table=NameSelector(pattern=table.name),
                    columns=[NameSelector(pattern=text_column.name)],
                    keys=[NameSelector(pattern=key_column.name)],
                )
            ],
        )
    ]


def input_schema() -> SchemaName:
    return SchemaName("INPUT_SCHEMA")


def input_table_name(id: int = 0) -> TableName:
    suffix = f"_{id}" if id else ""
    return TableNameImpl(table_name=f"INPUT_TABLE{suffix}", schema=input_schema())


def input_table(table_name=input_table_name()) -> Table:
    return Table(name=table_name, columns=[key_column(), text_column()])


def text_column_name() -> ColumnName:
    return ColumnName("text")


def key_column_name() -> ColumnName:
    return ColumnName("key")


def text_column(name: ColumnName = text_column_name()) -> Column:
    return varchar_column_spec(name=name.name).to_column()


def key_column(name: ColumnName = key_column_name()) -> Column:
    return integer_column_spec(name=name.name).to_column()


def source_tables(input_table_id: int = 0) -> list[SchemaSource]:
    return schema_sources(
        schema=input_schema(),
        table=input_table_name(input_table_id),
        text_column=text_column_name(),
        key_column=key_column_name(),
    )


def source_table_extractor(sources=-1):
    if sources == -1:
        sources = source_tables()
    return SourceTableExtractor(sources=sources, name="DOCUMENTS")


def report_table_sizes(pyexasol_connection: pyexasol.ExaConnection):
    for row in pyexasol_connection.execute(
        "select ROOT_NAME, OBJECT_NAME, RAW_OBJECT_SIZE"
        " from EXA_ALL_OBJECT_SIZES"
        " where OBJECT_TYPE='TABLE'"
        " order by RAW_OBJECT_SIZE desc"
    ):
        schema, table, size = row
        if size < 1000:
            return
        table_name = TableNameImpl(table, SchemaName(schema))
        print(f"{table_name.fully_qualified}: {size} Bytes")


def explain_table(
    pyexasol_connection: pyexasol.ExaConnection,
    table: TableName,
) -> str:
    """
    Especially in end-to-end tests you often have a failure and need to
    inspect the data in order to find the root cause.  This fixture enables
    creating a pytest assert statement with an info string to be displayed
    when the assertion fails:

    assert False, f'{explain_table(connection, table)}'

    The info string then contains the definition of the SQL table together
    with its content formated as SQL statements in order to enable isolated
    experiments outside the end-to-end test environment.

    Please note that is is only feasible for tables with a low amount of data.
    """
    describe = ",\n  ".join(
        f'"{row[0]}" {row[1]}'
        for row in pyexasol_connection.execute(f"DESCRIBE {table.fully_qualified}")
    )

    def fmt(row: tuple[Any]) -> str:
        def value(x: Any) -> str:
            if x is None:
                return "NULL"
            if isinstance(x, str):
                return f"'{x}'"
            return str(x)

        return "(" + ", ".join(value(x) for x in row) + ")"

    content = ",\n  ".join(
        fmt(row)
        for row in pyexasol_connection.execute(f"SELECT * FROM {table.fully_qualified}")
    )
    return (
        f"\nCREATE TABLE IF NOT EXISTS {table.fully_qualified} (\n"
        f"  {describe}\n"
        ");\n\n"
        f"INSERT INTO {table.fully_qualified} VALUES (\n"
        f"  {content}\n"
        ");\n"
    )
