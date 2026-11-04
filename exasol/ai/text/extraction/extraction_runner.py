from typing import Protocol

from exasol.analytics.sql_executor.interface import SQLExecutor

from exasol.ai.text.extraction.extraction import AbstractExtraction


class ExtractionRunner(Protocol):
    """
    An :class:`ExtractionRunner` provides the actual implementation, while
    an :class:`Extraction` is only the declaration without actual
    implementation. Additionally, this approach gives the option to replace
    the actual implementation with alternative variants.

    :language_alias: Required as exasol-text-ai-extension generates UDFs
    on-the-fly.
    """

    def run(
        self,
        extraction: AbstractExtraction,
        sql_executor: SQLExecutor,
        temporary_db_object_schema: str,
        language_alias: str,
    ) -> None:
        pass
