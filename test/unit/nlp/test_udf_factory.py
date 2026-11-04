import json
import re
from textwrap import dedent

import pytest
from exasol.analytics.schema import (
    SchemaName,
    UDFNameImpl,
)

from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


@pytest.fixture
def udf_create_query() -> str:
    udf_name = UDFNameImpl("my_udf", SchemaName("my_schema"))
    udf_factory = UDFFactory(udf_name=udf_name, language_alias="MY_PYTHON3")

    test_runner = UDFRunner(
        algorithm={"algorithm_name": "dummy-word-counter"}, batch_size=10
    )

    sql_query = udf_factory.create_udf_query(test_runner)
    return _collapse_spaces(sql_query)


def test_query_header(udf_create_query):

    expected_header = dedent("""
    CREATE OR REPLACE MY_PYTHON3 SET SCRIPT "my_schema"."my_udf"(
        "TEXT_DOC_ID" INTEGER, "TEXT_CHAR_BEGIN" INTEGER, "TEXT_CHAR_END" INTEGER,
        "TEXT" VARCHAR(2000000)
    ) EMITS (
        "TEXT_DOC_ID" INTEGER, "TEXT_CHAR_BEGIN" INTEGER, "TEXT_CHAR_END" INTEGER,
        "WORD_COUNT" INTEGER, "ERROR_MESSAGE" VARCHAR(2000000), "SETUP" VARCHAR(2000000)
    ) AS

    from exasol.ai.text.impl.nlp.udf_runner import UDFRunner

    runner_json =
    """)
    assert udf_create_query.startswith(_collapse_spaces(expected_header))


def test_query_footer(udf_create_query):

    expected_footer = dedent("""
    runner = UDFRunner.model_validate_json(runner_json)

    def run(ctx):
        runner.run(ctx, exa)
    /
    """)
    assert udf_create_query.endswith(_collapse_spaces(expected_footer))


def test_query_json(udf_create_query):

    runner_fields = {"algorithm", "batch_size"}
    algorithm_fields = {"inputs", "outputs", "setup", "algorithm_name"}
    runner_json_match = re.match(r'.*"""({.+})"""', udf_create_query)
    assert runner_json_match is not None
    runner_json = json.loads(runner_json_match[1])
    assert set(runner_json) == runner_fields
    assert set(runner_json["algorithm"]) == algorithm_fields
