from __future__ import annotations

from exasol.analytics.sql_executor.pyexasol_impl import PyexasolSQLExecutor
from pyexasol.connection import ExaConnection

from exasol.ai.text.extraction import AbstractExtraction
from exasol.ai.text.impl.orchestrator import PythonExtractionRunner
from exasol.ai.text.impl.utils.pyexasol_utils import autocommit_off


class Extraction(AbstractExtraction):
    """
    An Extraction is the top-level configuration which extractor to run
    and where to store the results. The method run() will create an instance
    of class PythonExtractionRunner and pass the configuration together with
    all other parameters to its method run().

    :language_alias: Required as exasol-text-ai-extension generates UDFs
    on-the-fly.
    """

    def run(
        self,
        pyexasol_con: ExaConnection,
        temporary_db_object_schema: str,
        language_alias: str,
    ) -> None:
        runner = PythonExtractionRunner()
        with autocommit_off(pyexasol_con):
            sql_executor = PyexasolSQLExecutor(pyexasol_con)
            runner.run(
                self,
                sql_executor,
                temporary_db_object_schema,
                language_alias,
            )
