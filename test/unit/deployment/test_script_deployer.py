from dataclasses import fields

from exasol.ai.text.deployment.script_deployer import (
    enum_script_names,
    enum_scripts,
)
from exasol.ai.text.impl.scripts import TextFunc
from test.unit.utils.test_stage_graph_utils import collapse_spaces


def test_enum_script_names():
    """
    The test checks that all functions declared in the `TextFunc` dataclass have
    correspondent CREATE FUNCTION queries in the `Scripts` class.
    """
    deploy_script_names = set(enum_script_names())
    declared_script_names = {field.name for field in fields(TextFunc)}
    assert deploy_script_names == declared_script_names


def test_enum_scripts():
    queries = [collapse_spaces(query) for query in enum_scripts()]
    assert len(queries) == len(fields(TextFunc))
    assert all(query.startswith("CREATE OR REPLACE") for query in queries)
