from collections.abc import (
    Callable,
    Generator,
)
from typing import (
    cast,
)

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
    TableLikeName,
    TableLikeNameImpl,
)
from exasol.analytics.sql_executor.pyexasol_impl import PyexasolSQLExecutor
from pyexasol import ExaConnection

from exasol.ai.text.impl.db_schema import SpanColumns
from exasol.ai.text.impl.nlp.algorithms.testing.span_size_categorizer import (
    SizeCategory,
    SpanSizeCategorizer,
)
from exasol.ai.text.impl.nlp.algorithms.testing.span_size_logger import (
    SpanSizeLogger,
    SpanSizeOutput,
)
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.orchestrator.python_extraction_runner import (
    query_handler_context,
    temporary_db_object_name_prefix,
)
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_query_handler import (
    get_lookup_table,
    get_output_view,
)
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)
from exasol.ai.text.impl.utils.sql_utils import (
    column_decl_to_str,
    varchar_column_spec,
)


def _delete_tables(
    tables: TableLikeName | list[TableLikeName], pyexasol_connection: ExaConnection
) -> None:
    if not isinstance(tables, list):
        tables = [tables]
    for tbl in tables:
        query = f"DROP TABLE IF EXISTS {tbl.fully_qualified}"
        pyexasol_connection.execute(query)


@pytest.fixture
def data_source(
    db_schema_name, pyexasol_connection
) -> Generator[TableLikeName, None, None]:
    doc_table = TableLikeNameImpl("documents", SchemaName(db_schema_name))
    text_span_columns = SpanColumns.create("TEXT").order
    text_column = varchar_column_spec("TEXT")

    query = (
        f"CREATE OR REPLACE TABLE {doc_table.fully_qualified}"
        f"({column_decl_to_str(text_span_columns + [text_column])})"
    )
    pyexasol_connection.execute(query)

    yield doc_table
    _delete_tables(doc_table, pyexasol_connection)


@pytest.fixture
def query_handler_runner_factory(
    language_alias, pyexasol_connection, data_source
) -> Callable[[UDFRunner, TableLikeName], PythonQueryHandlerRunner]:
    def factory(
        udf_runner: UDFRunner,
        output_table: TableLikeName,
    ) -> PythonQueryHandlerRunner:
        sql_executor = PyexasolSQLExecutor(pyexasol_connection)
        temp_prefix = temporary_db_object_name_prefix(sql_executor)
        context = query_handler_context(
            data_source.schema_name.name,
            temp_prefix,
        )
        algo = cast(SpanSizeLogger | SpanSizeCategorizer, udf_runner.algorithm)
        log_span = LogSpan(name="test")
        stage = UDFAlgoSQLStage(
            udf_runner=udf_runner,
            output_table=output_table,
            language_alias=language_alias,
            parallelism_per_node=2,
            log_span=log_span,
        )
        data_in = AlgoIOData(datasource=data_source, columns=algo.inputs)
        stage_input = SQLStageQueryHandlerInput(
            sql_stage_inputs=[data_in], result_bucketfs_location=None
        )  # type: ignore

        return PythonQueryHandlerRunner(
            sql_executor=sql_executor,
            top_level_query_handler_context=context,
            parameter=stage_input,
            query_handler_factory=stage.create_query_handler,
        )

    return factory


@pytest.mark.parametrize(
    "setup", ["my_setup", None], ids=["explicit_setup", "implicit_setup"]
)
def test_non_normalized_udf_algo_qh(
    setup_database,
    pyexasol_connection,
    data_source,
    query_handler_runner_factory,
    setup,
):
    """
    This test tests the case when nothing in the algorithm output is normalisable.
    """
    outputs = SpanSizeOutput(setup=varchar_column_spec("SETUP"))
    algorithm = SpanSizeLogger(outputs=outputs, setup=setup)
    udf_runner = UDFRunner(algorithm=algorithm)
    expected_setup = udf_runner.algo_setup
    output_table = TableLikeNameImpl("span_sizes", data_source.schema_name)
    _delete_tables(output_table, pyexasol_connection)

    try:
        expected_output = []
        doc_id = 0
        for update_no in range(1, 4):
            input_values = [
                (doc_id := doc_id + 1, 0, 10 * update_no + line_no)
                for line_no in range(1, 10)
            ]
            sql_values = ", ".join(
                f"({', '.join(str(val) for val in row)}, 'nothing')"
                for row in input_values
            )
            expected_output.extend(
                (*row, row[-1], None, expected_setup) for row in input_values
            )
            query = (
                f"INSERT INTO {data_source.fully_qualified} " f"VALUES({sql_values})"
            )
            pyexasol_connection.execute(query)

            query_handler_runner = query_handler_runner_factory(
                udf_runner, output_table
            )
            query_handler_runner.run()

            query = f"SELECT * from {output_table.fully_qualified}"
            result = pyexasol_connection.execute(query).fetchall()
            assert sorted(result) == sorted(expected_output)
    finally:
        _delete_tables(output_table, pyexasol_connection)


@pytest.mark.parametrize(
    "setup", ["my_setup", None], ids=["explicit_setup", "implicit_setup"]
)
def test_normalized_udf_algo_qh(
    setup_database,
    pyexasol_connection,
    data_source,
    query_handler_runner_factory,
    setup,
):
    """
    This test tests the case when the algorithm output has normalisable columns
    The setup column is also configured to be normalisable.

    The test uses the span-size-categorizer algorithm. It is designed such that every
    update of the Document table includes texts with the spans that belong to a new
    category. This way it easy to check the IDs of the normalised elements. Every
    iteration add a new ID.
    """

    categories = [
        SizeCategory(min_size=0, category="small"),
        SizeCategory(min_size=10, category=None),
        SizeCategory(min_size=20, category="large"),
    ]
    algorithm = SpanSizeCategorizer(categories=categories, setup=setup)
    udf_runner = UDFRunner(algorithm=algorithm)
    expected_setup = udf_runner.algo_setup
    output_table = TableLikeNameImpl("span_size_categories", data_source.schema_name)
    output_view = get_output_view(output_table)
    setup_table = get_lookup_table(output_table, udf_runner.algorithm.outputs.setup)
    category_table = get_lookup_table(output_table, algorithm.outputs.size_category)
    all_tables = [setup_table, category_table, output_table, output_view]
    _delete_tables(all_tables, pyexasol_connection)

    try:
        expected_output = []
        expected_view = []
        expected_setups = [(1, expected_setup)]
        expected_categories = []
        expected_results = [
            expected_setups,
            expected_categories,
            expected_output,
            expected_view,
        ]
        doc_id = 0
        category_id = 0
        for update_no in range(3):
            input_values = [
                (doc_id := doc_id + 1, 0, 10 * update_no + line_no)
                for line_no in range(1, 10)
            ]
            sql_values = ", ".join(
                f"({', '.join(str(val) for val in row)}, 'nothing')"
                for row in input_values
            )
            category = categories[update_no].category
            if category:
                category_id += 1
                cat_id_or_null = category_id
                expected_categories.append((category_id, category))
            else:
                cat_id_or_null = None
            expected_output.extend(
                (*row, cat_id_or_null, None, 1) for row in input_values
            )
            expected_view.extend(
                (*row, category, None, expected_setup) for row in input_values
            )
            query = (
                f"INSERT INTO {data_source.fully_qualified} " f"VALUES({sql_values})"
            )
            pyexasol_connection.execute(query)

            query_handler_runner = query_handler_runner_factory(
                udf_runner, output_table
            )
            query_handler_runner.run()

            for tbl, expected_content in zip(all_tables, expected_results):
                query = f"SELECT * from {tbl.fully_qualified}"
                result = pyexasol_connection.execute(query).fetchall()
                assert sorted(result) == sorted(expected_content)
    finally:
        _delete_tables(all_tables, pyexasol_connection)
