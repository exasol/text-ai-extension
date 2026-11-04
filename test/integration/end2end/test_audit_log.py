from inspect import cleandoc
from typing import Any

import pandas as pd
import pyexasol
from exasol.analytics.audit.columns import BaseAuditColumns
from exasol.analytics.schema import (
    Column,
    TableName,
    TableNameImpl,
)

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import PipelineExtractor
from test.utils.dataframe_matcher import DataFrameMatcher
from test.utils.helpers import (
    defaults,
    source_table_extractor,
    standard_extractor,
)


def where_clause(operator: str, *pairs: tuple[Column, Any]) -> str:
    """
    Build an SQL WHERE clause using the specified pairs of columns and
    values, connected by the specified operator, e.g. "OR" or "AND".
    """
    return f"\n {operator} ".join(f"{k.name.quoted_name} = '{v}'" for k, v in pairs)


def select_clause(*columns: Column) -> str:
    return ", ".join(col.name.quoted_name for col in columns)


def query(
    connection: pyexasol.ExaConnection,
    table: TableName,
    select: str,
    where: str,
    suffix: str = "",
) -> pyexasol.ExaStatement:
    return connection.execute(f"""
        SELECT {select} FROM {table.fully_qualified}
        WHERE {where}
        {suffix}
        """)


def column_size(
    connection: pyexasol.ExaConnection,
    table: TableName,
    column_name: str,
):
    statement = cleandoc(f"""
        select COLUMN_MAXSIZE from EXA_ALL_COLUMNS
        where column_schema = '{table.schema_name.name}'
          and column_table = '{table.name}'
          and column_name = '{column_name}'
        """)
    row = connection.execute(statement).fetchone()
    return row[0]


def test_extraction_run(
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    end2end_output_schema,
):
    """
    This test runs an extraction consisting of only a
    source_table_extractor().
    """
    # Create input data
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES"
        " (1, 'This is a test.'),"
        " (2, 'Exasol is awesome.')"
    )
    # Run the extraction
    Extraction(
        extractor=PipelineExtractor(
            steps=[source_table_extractor(), standard_extractor()]
        ),
        output=Output(db_schema=end2end_output_schema.name),
        defaults=defaults(),
    ).run(pyexasol_connection, db_schema_name, language_alias)

    # Verify that explicit audit messages are contained in the audit log.
    audit_table = TableNameImpl("TXAIE_AUDIT_LOG", end2end_output_schema)
    audit_query_count = query(
        pyexasol_connection,
        audit_table,
        "count(1)",
        where_clause(
            "OR",
            (BaseAuditColumns.EVENT_NAME, "Start"),
            (BaseAuditColumns.EVENT_NAME, "End"),
        ),
    ).fetchone()[0]
    assert audit_query_count >= 2
    # Verify that a modify query with audit=True is captured in the audit log, too.
    columns = [
        BaseAuditColumns.RUN_ID,
        BaseAuditColumns.LOG_SPAN_NAME,
        BaseAuditColumns.LOG_SPAN_ID,
        BaseAuditColumns.PARENT_LOG_SPAN_ID,
        BaseAuditColumns.EVENT_NAME,
        BaseAuditColumns.ROW_COUNT,
        BaseAuditColumns.DB_OBJECT_TYPE,
        BaseAuditColumns.DB_OBJECT_NAME,
    ]
    result_set = query(
        pyexasol_connection,
        audit_table,
        select_clause(*columns),
        "TRUE",
        "ORDER BY LOG_TIMESTAMP ASC",
    )
    rows_df = pd.DataFrame(result_set.fetchall(), columns=result_set.column_names())
    match_columns = [
        "LOG_SPAN_NAME",
        "EVENT_NAME",
        "ROW_COUNT",
        "DB_OBJECT_NAME",
        "DB_OBJECT_TYPE",
    ]
    # print("rows_df", rows_df.to_string())
    expected_df = pd.DataFrame(
        [
            ["SourceTableExtractor DOCUMENTS", "Start", None, None, None],
            ["INSERT", "Begin", "0", "DOCUMENTS_INPUT_SCHEMA_INPUT_TABLE", "TABLE"],
            ["INSERT", "End", "2", "DOCUMENTS_INPUT_SCHEMA_INPUT_TABLE", "TABLE"],
            ["INSERT", "Begin", "0", "DOCUMENTS", "TABLE"],
            ["INSERT", "End", "2", "DOCUMENTS", "TABLE"],
            ["SourceTableExtractor DOCUMENTS", "End", None, None, None],
            ["NamedEntityExtractor NAMED_ENTITY", "Start", None, None, None],
            ["INSERT", "Begin", "0", "ENTITY_LOOKUP_ENTITY_TYPE", "TABLE"],
            ["INSERT", "End", "0", "ENTITY_LOOKUP_ENTITY_TYPE", "TABLE"],
            ["INSERT", "Begin", "0", "ENTITY_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "End", "1", "ENTITY_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "Begin", "0", "ENTITY", "TABLE"],
            ["INSERT", "End", "2", "ENTITY", "TABLE"],
            ["NamedEntityExtractor NAMED_ENTITY", "Error", None, None, None],
            ["NamedEntityExtractor NAMED_ENTITY", "End", None, None, None],
            ["KeywordSearchExtractor KEYWORD_SEARCH", "Start", None, None, None],
            ["INSERT", "Begin", "0", "KEYWORD_LOOKUP_KEYWORD", "TABLE"],
            ["INSERT", "End", "0", "KEYWORD_LOOKUP_KEYWORD", "TABLE"],
            ["INSERT", "Begin", "0", "KEYWORD_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "End", "1", "KEYWORD_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "Begin", "0", "KEYWORD", "TABLE"],
            ["INSERT", "End", "2", "KEYWORD", "TABLE"],
            ["KeywordSearchExtractor KEYWORD_SEARCH", "Error", None, None, None],
            ["KeywordSearchExtractor KEYWORD_SEARCH", "End", None, None, None],
            ["TokenizerExtractor TOKENIZER", "Start", None, None, None],
            ["INSERT", "Begin", "0", "TOKENIZER_LOOKUP_LEMMA", "TABLE"],
            ["INSERT", "End", "7", "TOKENIZER_LOOKUP_LEMMA", "TABLE"],
            ["INSERT", "Begin", "0", "TOKENIZER_LOOKUP_POS", "TABLE"],
            ["INSERT", "End", "6", "TOKENIZER_LOOKUP_POS", "TABLE"],
            ["INSERT", "Begin", "0", "TOKENIZER_LOOKUP_TAG", "TABLE"],
            ["INSERT", "End", "6", "TOKENIZER_LOOKUP_TAG", "TABLE"],
            ["INSERT", "Begin", "0", "TOKENIZER_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "End", "1", "TOKENIZER_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "Begin", "0", "TOKENIZER", "TABLE"],
            ["INSERT", "End", "9", "TOKENIZER", "TABLE"],
            ["TokenizerExtractor TOKENIZER", "Error", None, None, None],
            ["TokenizerExtractor TOKENIZER", "End", None, None, None],
            ["TopicClassifierExtractor TOPIC_CLASSIFIER", "Start", None, None, None],
            ["INSERT", "Begin", "0", "TOPIC_LOOKUP_TOPIC", "TABLE"],
            ["INSERT", "End", "0", "TOPIC_LOOKUP_TOPIC", "TABLE"],
            ["INSERT", "Begin", "0", "TOPIC_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "End", "1", "TOPIC_LOOKUP_SETUP", "TABLE"],
            ["INSERT", "Begin", "0", "TOPIC", "TABLE"],
            ["INSERT", "End", "2", "TOPIC", "TABLE"],
            ["TopicClassifierExtractor TOPIC_CLASSIFIER", "Error", None, None, None],
            ["TopicClassifierExtractor TOPIC_CLASSIFIER", "End", None, None, None],
        ],
        columns=match_columns,
    )
    matcher = DataFrameMatcher(
        expected=expected_df.sort_values(match_columns).reset_index(drop=True),
        result=rows_df[match_columns].sort_values(match_columns).reset_index(drop=True),
    )
    assert matcher, matcher.diff()
    assert not rows_df["RUN_ID"].isnull().any()
    assert not rows_df["LOG_SPAN_ID"].isnull().any()
    assert (
        not rows_df[rows_df["LOG_SPAN_NAME"] == "INSERT"]["PARENT_LOG_SPAN_ID"]
        .isnull()
        .any()
    )
    assert (
        rows_df[rows_df["LOG_SPAN_NAME"] != "INSERT"]["PARENT_LOG_SPAN_ID"]
        .isnull()
        .all()
    )

    error_message_column_size = column_size(
        pyexasol_connection,
        audit_table,
        BaseAuditColumns.ERROR_MESSAGE.name.name,
    )
    assert error_message_column_size == 2000000
