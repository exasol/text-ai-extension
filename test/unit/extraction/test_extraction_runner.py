from unittest.mock import (
    Mock,
    call,
    patch,
)

from exasol.analytics.sql_executor.pyexasol_impl import PyexasolSQLExecutor

from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.impl.orchestrator import PythonExtractionRunner
from test.utils.helpers import source_table_extractor


def test_extraction_runner(output):
    extraction = Extraction(extractor=source_table_extractor(), output=output)
    pyexasol_con = Mock(options={"autocommit": True})
    tmp_db_schema = "my-db-schema"
    language_alias = "my-alias"
    with patch.object(PythonExtractionRunner, attribute="run") as run_method:
        extraction.run(pyexasol_con, tmp_db_schema, language_alias)
    assert run_method.called
    sql_executor = run_method.call_args_list[0].args[1]
    assert isinstance(sql_executor, PyexasolSQLExecutor)
    assert sql_executor._connection == pyexasol_con
    assert run_method.call_args_list == [
        call(extraction, sql_executor, tmp_db_schema, language_alias)
    ]


@patch.object(PythonExtractionRunner, attribute="run")
def test_autocommit(run_method, output):
    extraction = Extraction(extractor=source_table_extractor(), output=output)
    pyexasol_con = Mock(options={"autocommit": True})
    tmp_db_schema = "my-db-schema"
    language_alias = "my-alias"
    extraction.run(pyexasol_con, tmp_db_schema, language_alias)
    assert pyexasol_con.set_autocommit.call_args_list == [call(False), call(True)]
