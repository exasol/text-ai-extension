from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
)

import pyexasol
import pytest
from exasol.analytics.schema import (
    TableName,
    TableNameImpl,
)

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from test.utils.helpers import (
    explain_table,
    keyword_search_extractor,
    named_entity_extractor,
    topic_classifier_extractor,
)


class ExtractorFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> NLPExtractor: ...


@dataclass
class E2eScenario:
    """
    Scenario for End-to-end tests defined by an NLP extractor and the
    related output table.
    """

    db: pyexasol.ExaConnection
    output_table: TableName
    extractor_factory: ExtractorFactory | None

    def explain(self) -> str:
        return explain_table(self.db, self.output_table)

    @property
    def row_count(self) -> int:
        return self.db.execute(
            f"SELECT COUNT(1) FROM {self.output_table.fully_qualified}"
        ).fetchone()[0]

    def __str__(self):
        return (
            f"{self.extractor_factory.__name__}"
            f" -> {self.output_table.fully_qualified}"
        )


@pytest.fixture(
    params=[
        (None, "DOCUMENTS"),
        (named_entity_extractor, "NAMED_ENTITY"),
        (topic_classifier_extractor, "TOPIC_CLASSIFIER"),
        (keyword_search_extractor, "KEYWORD_SEARCH"),
    ]
)
def e2e_scenario(request, pyexasol_connection, end2end_output_schema) -> E2eScenario:
    """
    Executes each test case using this fixture multiple times, once for
    each parameter tuple.
    """
    table = TableNameImpl(request.param[1], end2end_output_schema)
    return E2eScenario(
        db=pyexasol_connection,
        output_table=table,
        extractor_factory=request.param[0],
    )
