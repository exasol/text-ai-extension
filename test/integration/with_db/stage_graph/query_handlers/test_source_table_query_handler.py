from collections.abc import (
    Callable,
    Generator,
)
from dataclasses import dataclass
from textwrap import dedent
from typing import (
    Any,
)

import pytest
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
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

from exasol.ai.text.extractors.source_table_extractor import (
    NameSelector,
    SchemaSource,
    SelectorPredicate,
    TableSource,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.orchestrator.python_extraction_runner import (
    query_handler_context,
    temporary_db_object_name_prefix,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.metadata_table import (
    SOURCE_COLUMN,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table.utils import (
    get_map_table,
    get_source_view,
)
from exasol.ai.text.impl.stage_graph.query_handlers.source_table_sql_stage import (
    SourceTableSQLStage,
)
from exasol.ai.text.impl.utils.sql_utils import sql_text_value

DOC_ID_COLUMN_ID = 0
TEXT_BEGIN_COLUMN_ID = 1
TEXT_END_COLUMN_ID = 2
TEXT_COLUMN_ID = 3


@dataclass
class _TestDbObject:
    name: str
    pattern: str
    predicate: SelectorPredicate


@dataclass
class _TestColumn(_TestDbObject):
    sql_type: str
    is_text: bool = False
    is_key: bool = False
    is_primary_key: bool = False


@dataclass
class _TestTable(_TestDbObject):
    columns: list[_TestColumn]


@dataclass
class _TestSchema(_TestDbObject):
    tables: list[_TestTable]


@dataclass
class _TestTableUpdate:
    table: TableLikeName
    columns: list[_TestColumn]
    values: list[tuple[Any, ...]]


def _column_sql(col: _TestColumn) -> str:
    constraint = " PRIMARY KEY" if col.is_primary_key else ""
    return f'"{col.name}" {col.sql_type}{constraint}'


def _sql_value(val: Any, sql_type: str) -> str:
    if val is None:
        return "NULL"
    sql_type = sql_type.upper()
    if sql_type.startswith("CHAR") or sql_type.startswith("VARCHAR"):
        return sql_text_value(val)
    return str(val)


def _sql_row(columns: list[_TestColumn], values: tuple[Any]) -> str:
    return ", ".join(_sql_value(val, col.sql_type) for col, val in zip(columns, values))


def _delete_all_tables(
    doc_table: TableLikeName,
    objects: list[_TestSchema],
    pyexasol_connection: ExaConnection,
) -> None:
    # Full list of tables expected to be created during tests.
    source_tables: list[TableLikeName] = [
        TableLikeNameImpl(test_table.name, SchemaName(test_schema.name))
        for test_schema in objects
        for test_table in test_schema.tables
    ]
    all_tables = [get_map_table(doc_table, tbl) for tbl in source_tables]
    all_tables.append(doc_table)
    all_tables.extend(source_tables)

    for tbl in all_tables:
        query = f"DROP TABLE IF EXISTS {tbl.fully_qualified}"
        pyexasol_connection.execute(query)


def _create_source_tables(
    objects: list[_TestSchema], pyexasol_connection: ExaConnection
) -> None:
    for test_schema in objects:
        for test_table in test_schema.tables:
            columns = ", ".join(_column_sql(col) for col in test_table.columns)
            query = (
                f'CREATE OR REPLACE TABLE "{test_schema.name}"."{test_table.name}"'
                f"({columns})"
            )
            pyexasol_connection.execute(query)


@pytest.fixture()
def document_table(db_schema_name) -> TableLikeName:
    return TableLikeNameImpl("documents", SchemaName(db_schema_name))


@pytest.fixture
def db_objects(
    db_schema_name, document_table, pyexasol_connection
) -> Generator[list[_TestSchema], None, None]:
    """
    The fixture defines the source tables. For each database object it provides the
    actual name and a pattern that might be used in the Source Table Extractor in
    order to select this name.
    The fixture creates the source tables. At the exit it cleans up all the tables
    created during the tests.
    """
    objects = [
        _TestSchema(
            db_schema_name,
            db_schema_name,
            SelectorPredicate.EXACT,
            [
                _TestTable(
                    "Orchids",
                    "Orchids",
                    SelectorPredicate.EXACT,
                    [
                        _TestColumn(
                            "Common_Name",
                            "Common_Name",
                            SelectorPredicate.EXACT,
                            "VARCHAR(1000)",
                            is_text=True,
                        ),
                        _TestColumn(
                            "Latin_Name",
                            "Latin%",
                            SelectorPredicate.LIKE,
                            "VARCHAR(1000)",
                            is_text=True,
                        ),
                        _TestColumn(
                            "Plant_ID",
                            "Plant_ID",
                            SelectorPredicate.EXACT,
                            "DECIMAL(18,0)",
                            is_key=True,
                        ),
                    ],
                ),
                _TestTable(
                    "Bear_Habitats",
                    "Bear%",
                    SelectorPredicate.LIKE,
                    [
                        _TestColumn(
                            "Animal_Name",
                            "Animal_Name",
                            SelectorPredicate.EXACT,
                            "VARCHAR(1000)",
                            is_text=True,
                        ),
                        _TestColumn(
                            "Country_or_Region",
                            r"[\w|_]*Region",
                            SelectorPredicate.REGEXP_LIKE,
                            "VARCHAR(1000)",
                            is_text=True,
                        ),
                        _TestColumn(
                            "Country_Code",
                            "%Code",
                            SelectorPredicate.LIKE,
                            "CHAR(3)",
                            is_key=True,
                        ),
                        _TestColumn(
                            "Animal_ID",
                            r"\w*_ID",
                            SelectorPredicate.REGEXP_LIKE,
                            "DECIMAL(18,0)",
                            is_key=True,
                        ),
                    ],
                ),
            ],
        ),
        _TestSchema(
            db_schema_name,
            db_schema_name,
            SelectorPredicate.EXACT,
            [
                _TestTable(
                    "Biogeographic_realms",
                    "Biogeographic%",
                    SelectorPredicate.LIKE,
                    [
                        _TestColumn(
                            "Realm",
                            "Realm",
                            SelectorPredicate.EXACT,
                            "VARCHAR(100)",
                            is_primary_key=True,
                        ),
                        _TestColumn(
                            "Lands_Included",
                            r"Lands.*",
                            SelectorPredicate.REGEXP_LIKE,
                            "VARCHAR(20000)",
                            is_text=True,
                        ),
                    ],
                )
            ],
        ),
    ]

    _delete_all_tables(document_table, objects, pyexasol_connection)
    _create_source_tables(objects, pyexasol_connection)
    try:
        yield objects
    finally:
        _delete_all_tables(document_table, objects, pyexasol_connection)


@pytest.fixture
def sources(db_objects) -> list[SchemaSource]:
    """
    The fixture provides the definition of the source texts, as it would be done by the
    Source Table Extractor.
    """
    return [
        SchemaSource(
            db_schema=NameSelector(
                pattern=test_schema.pattern, predicate=test_schema.predicate
            ),
            tables=[
                TableSource(
                    table=NameSelector(
                        pattern=test_table.pattern, predicate=test_table.predicate
                    ),
                    columns=[
                        NameSelector(
                            pattern=test_column.pattern, predicate=test_column.predicate
                        )
                        for test_column in test_table.columns
                        if test_column.is_text
                    ],
                    keys=[
                        NameSelector(
                            pattern=test_column.pattern, predicate=test_column.predicate
                        )
                        for test_column in test_table.columns
                        if test_column.is_key
                    ],
                )
                for test_table in test_schema.tables
            ],
        )
        for test_schema in db_objects
    ]


@pytest.fixture
def db_updates(db_objects) -> list[list[_TestTableUpdate]]:
    """
    The fixture provides the data for mimicking a sequence of source table updates.
    """
    tables_with_columns = [
        _TestTableUpdate(
            TableLikeNameImpl(test_table.name, SchemaName(test_schema.name)),
            test_table.columns,
            [],
        )
        for test_schema in db_objects
        for test_table in test_schema.tables
    ]

    def create_table_update(
        table_no: int, values: list[tuple[Any, ...]]
    ) -> _TestTableUpdate:
        return _TestTableUpdate(
            tables_with_columns[table_no].table,
            tables_with_columns[table_no].columns,
            values,
        )

    return [
        [
            create_table_update(
                0,
                [
                    ("White Helleborine", "Cephalanthera damasonium", 1001),
                    ("Lesser twayblade", "Neottia cordata", 1002),
                    ("Early purple orchid", None, 1100),
                ],
            ),
            create_table_update(
                1,
                [
                    ("American Black Bear", "Canada", "CAN", 1256),
                    ("North American Brown Bear", "Canada", "CAN", 1244),
                    ("American Black Bear", "United States", "USA", 1256),
                    ("North American Brown Bear", "United States", "USA", 1244),
                    ("Polar Bear", "Canada", "CAN", 1278),
                    ("Polar Bear", "United States", "USA", 1278),
                ],
            ),
            create_table_update(
                2,
                [
                    ("Palearctic", "The bulk of Eurasia and North Africa."),
                    ("Nearctic", "Greenland and most of North America."),
                    ("Afrotropic", "Trans-Saharan Africa, Madagascar and Arabia."),
                    (
                        "Neotropic",
                        (
                            "South America, Central America, the Caribbean, "
                            "South Florida and the Falkland Islands."
                        ),
                    ),
                    (
                        "Antarctic",
                        (
                            "Antarctica, Alexander Island, South Georgia "
                            "and the South Sandwich Islands."
                        ),
                    ),
                ],
            ),
        ],
        [
            create_table_update(
                0,
                [
                    ("Autumn Lady's Tresses", "Spiranthes spiralis", 1003),
                    ("Small white orchid", "Pseudorchis albida", 1004),
                    (None, "Dactylorhiza fuchsia", 1105),
                ],
            ),
            create_table_update(
                1,
                [
                    ("Polar Bear", "Norway", "NOR", 1278),
                    ("Polar Bear", "Russia", "RUS", 1278),
                ],
            ),
            create_table_update(
                2,
                [
                    (
                        "Australasia",
                        (
                            "Australia, Melanesia, New Zealand, "
                            "Lesser Sunda Islands, Sulawesi, Maluku Islands."
                        ),
                    ),
                    (
                        "Indomalaya",
                        (
                            "The Indian subcontinent, Southeast Asia, "
                            "southern China and Greater Sunda Islands."
                        ),
                    ),
                    (
                        "Oceania",
                        (
                            "Polynesia (except New Zealand), Micronesia, "
                            "and the Fijian Islands."
                        ),
                    ),
                ],
            ),
        ],
        [
            create_table_update(
                1,
                [
                    ("Bron Bear", "Russia", "RUS", 1235),
                    ("Himalayan Brown Bear", "Nepal", "NPL", 1236),
                    ("Panda Bear", "China", "CHN", 1301),
                ],
            )
        ],
    ]


@pytest.fixture
def query_handler_runner_factory(
    db_schema_name, pyexasol_connection, document_table, sources
) -> Callable[[], PythonQueryHandlerRunner]:
    def factory() -> PythonQueryHandlerRunner:
        sql_executor = PyexasolSQLExecutor(pyexasol_connection)
        temp_prefix = temporary_db_object_name_prefix(sql_executor)
        context = query_handler_context(db_schema_name, temp_prefix)
        log_span = LogSpan(name="test")
        stage = SourceTableSQLStage(
            sources=sources, document_table=document_table, log_span=log_span
        )
        stage_input = SQLStageQueryHandlerInput(
            sql_stage_inputs=[SQLStageInputOutput()], result_bucketfs_location=None
        )  # type: ignore

        return PythonQueryHandlerRunner(
            sql_executor=sql_executor,
            top_level_query_handler_context=context,
            parameter=stage_input,
            query_handler_factory=stage.create_query_handler,
        )

    return factory


def test_source_table_qh_document(
    setup_database,
    pyexasol_connection,
    document_table,
    db_objects,
    query_handler_runner_factory,
    db_updates,
):
    """
    The test performs a sequence of the source table updates and verifies that after
    each update the Document table has the expected content.
    """

    expected_texts: list[str] = []
    for update in db_updates:
        for tbl in update:
            sql_values = ", ".join(
                f"({_sql_row(tbl.columns, row)})" for row in tbl.values
            )
            expected_texts.extend(
                val
                for row in tbl.values
                for col, val in zip(tbl.columns, row)
                if col.is_text and (val is not None)
            )
            query = f"INSERT INTO {tbl.table.fully_qualified} VALUES({sql_values})"
            pyexasol_connection.execute(query)

        query_handler_runner = query_handler_runner_factory()
        query_handler_runner.run()

        query = f"SELECT * FROM {document_table.fully_qualified}"
        results = pyexasol_connection.execute(query).fetchall()

        # Verify that all expected texts are extracted.
        texts = sorted(row[TEXT_COLUMN_ID] for row in results)
        assert texts == sorted(expected_texts)
        # Verify that all document IDs are unique.
        doc_ids = {row[DOC_ID_COLUMN_ID] for row in results}
        assert len(doc_ids) == len(results)
        # Verify that all spans are (0, len(text)
        assert all(row[TEXT_BEGIN_COLUMN_ID] == 0 for row in results)
        assert all(
            row[TEXT_END_COLUMN_ID] == len(row[TEXT_COLUMN_ID]) for row in results
        )


def collect_source_texts(pyexasol_connection, document_table, db_objects) -> list[str]:
    """
    Collects all texts from all source tables using SELECT queries where the source
    tables are joint with the mapping and the document tables.
    """
    source_texts: list[str] = []
    for test_schema in db_objects:
        schema_name = SchemaName(test_schema.name)
        for test_table in test_schema.tables:
            source_tbl = TableLikeNameImpl(test_table.name, schema_name)
            source_view = get_source_view(document_table, source_tbl)
            for source_col in test_table.columns:
                if source_col.is_text:
                    query = dedent(f"""
                        SELECT "{source_col.name}"
                        FROM {source_view.fully_qualified}
                        WHERE "{SOURCE_COLUMN}"='{source_col.name}'
                    """)
                    source_texts.extend(pyexasol_connection.execute(query).fetchcol())
    return source_texts


def test_source_table_qh_mapping(
    setup_database,
    pyexasol_connection,
    document_table,
    db_objects,
    query_handler_runner_factory,
    db_updates,
):
    """
    This test verifies that the texts in the Document table can be traced back to the
    source tables where the texts were loaded from. We will build a combined list of
    the texts that can be obtained from the source tables if they are joint with the
    correspondent mapping tables and the Document table. The list should contain the
    same texts as those in the Document table.
    """

    # Do all the updates
    for update in db_updates:
        for tbl in update:
            sql_values = ", ".join(
                f"({_sql_row(tbl.columns, row)})" for row in tbl.values
            )
            query = f"INSERT INTO {tbl.table.fully_qualified} VALUES({sql_values})"
            pyexasol_connection.execute(query)

        query_handler_runner = query_handler_runner_factory()
        query_handler_runner.run()

    # Collect all texts from the source tables
    source_texts = collect_source_texts(pyexasol_connection, document_table, db_objects)

    # Get all texts from the document table
    text = TextInput().text.quoted_name
    query = f"SELECT {text} FROM {document_table.fully_qualified}"
    expected_texts = pyexasol_connection.execute(query).fetchcol()

    assert sorted(source_texts) == sorted(expected_texts)
