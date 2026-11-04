from dataclasses import dataclass

import pandas as pd

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.dataframe_matcher import DataFrameMatcher
from test.utils.helpers import (
    named_entity_extractor,
    source_table_extractor,
)


@dataclass
class _NamedQuery:
    name: str
    query: str


def test_idempotency_of_named_entity_extractor(
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    end2end_output_schema,
    model_subdir,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    named_queries = [
        _NamedQuery(
            "Output",
            f"SELECT * FROM {end2end_output_schema.fully_qualified}.NAMED_ENTITY",
        ),
        _NamedQuery(
            "Entity-Name",
            f"SELECT * FROM {end2end_output_schema.fully_qualified}.NAMED_ENTITY_LOOKUP_ENTITY_TYPE",
        ),
        _NamedQuery(
            "Setup",
            f"SELECT * FROM {end2end_output_schema.fully_qualified}.NAMED_ENTITY_LOOKUP_SETUP",
        ),
        _NamedQuery(
            "Input", f"SELECT * FROM {end2end_input_table.name.fully_qualified}"
        ),
        _NamedQuery(
            "Documents",
            f"SELECT * FROM {end2end_output_schema.fully_qualified}.DOCUMENTS",
        ),
    ]

    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES (1, 'This is a test.')"
    )
    pyexasol_connection.execute(
        f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES (2, 'Exasol is awesome.')"
    )

    extraction = Extraction(
        extractor=Pipeline(
            steps=[
                source_table_extractor(),
                named_entity_extractor(),
            ]
        ),
        output=Output(db_schema=end2end_output_schema.name),
    )

    def run_and_observe() -> list[pd.DataFrame]:
        extraction.run(pyexasol_connection, db_schema_name, language_alias)
        table_contents = [
            pyexasol_connection.export_to_pandas(named_query.query)
            for named_query in named_queries
        ]
        assert table_contents[0]["ERROR_MESSAGE"].isnull().all()
        return table_contents

    run_results = [run_and_observe() for _ in range(2)]

    def pd_to_string(df: pd.DataFrame):
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            return df.to_string()

    # Prepare an error message in case the test fails.
    all_content = "\n\n".join(
        f"Run {run_no} {named_query.name}\n{pd_to_string(tbl_content)}"
        for run_no, table_contents in enumerate(run_results)
        for named_query, tbl_content in zip(named_queries, table_contents)
    )

    matchers = [
        DataFrameMatcher(df0, df1) for df0, df1 in zip(run_results[0], run_results[1])
    ]
    matches = [bool(matcher) for matcher in matchers]
    diffs = "\n\n".join(matcher.diff() for matcher in matchers)
    error_message = (
        "A redundant update has changed the content of the output table "
        f"or one of the lookup tables.\n\nContent:\n\n{all_content}\n\nDiffs:\n\n{diffs}"
    )
    assert all(matches), error_message
