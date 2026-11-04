from dataclasses import dataclass

import pytest

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_envelope import (
    TextEnvelope,
    TextEnvelopeMapper,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec


@dataclass(frozen=True)
class FooTextOutput(TextOutput):
    name: str = "FOO"
    foo_column1: ColumnSpec = varchar_column_spec("C_FOO")
    foo_column2: ColumnSpec = varchar_column_spec("ENTITY")
    foo_span1: SpanColumns = SpanColumns.create("S_FOO")
    foo_span2: SpanColumns = SpanColumns.create("ENTITY")


@dataclass(frozen=True)
class BarTextOutput(TextOutput):
    name: str = "BAR"
    bar_column1: ColumnSpec = varchar_column_spec("C_BAR")
    bar_column2: ColumnSpec = varchar_column_spec("ENTITY")
    bar_span1: SpanColumns = SpanColumns.create("S_BAR")
    bar_span2: SpanColumns = SpanColumns.create("ENTITY")


def test_from_text_output():
    foo = FooTextOutput()
    foo_env = TextEnvelope.from_text_output(foo)
    assert foo_env.column_names == foo.column_names
    assert len(foo_env.envelope) == 4
    assert foo_env.original_type == FooTextOutput


@pytest.fixture
def foo_output() -> FooTextOutput:
    return FooTextOutput()


@pytest.fixture
def bar_output() -> BarTextOutput:
    return BarTextOutput()


@pytest.fixture
def foo_bar_mapper(foo_output, bar_output) -> TextEnvelopeMapper:
    foo_env = TextEnvelope.from_text_output(foo_output)
    bar_env = TextEnvelope.from_text_output(bar_output)
    return TextEnvelopeMapper([foo_env, bar_env])


def test_non_unique_env_names():
    foo = FooTextOutput(name="MY_ALGO")
    foo_env = TextEnvelope.from_text_output(foo)
    bar = BarTextOutput(name="MY_ALGO")
    bar_env = TextEnvelope.from_text_output(bar)
    with pytest.raises(ValueError, match=r"The names.+must be unique"):
        TextEnvelopeMapper([foo_env, bar_env])


def test_mapper_kwargs(foo_output):
    foo_env = TextEnvelope.from_text_output(foo_output)
    text_span = SpanColumns.create("MY_TEXT")
    error_message = varchar_column_spec("MY_ERROR_MESSAGE")
    setup = varchar_column_spec("MY_SETUP")
    mapper = TextEnvelopeMapper(
        [foo_env], text_span=text_span, error_message=error_message, setup=setup
    )
    assert mapper.text_span == text_span
    assert mapper.error_message == error_message
    assert mapper.setup == setup


@pytest.mark.parametrize(
    ["env", "name_in", "expected_name_out"],
    [
        ("FOO", "C_FOO", "C_FOO"),
        ("FOO", "ENTITY", "FOO_ENTITY"),
        (
            "FOO",
            SpanColumns.create("S_FOO").doc_id.name,
            SpanColumns.create("S_FOO").doc_id.name,
        ),
        (
            "FOO",
            SpanColumns.create("ENTITY").doc_id.name,
            SpanColumns.create("FOO_ENTITY").doc_id.name,
        ),
        ("BAR", "C_BAR", "C_BAR"),
        ("BAR", "ENTITY", "BAR_ENTITY"),
        (
            "BAR",
            SpanColumns.create("S_BAR").doc_id.name,
            SpanColumns.create("S_BAR").doc_id.name,
        ),
        (
            "BAR",
            SpanColumns.create("ENTITY").doc_id.name,
            SpanColumns.create("BAR_ENTITY").doc_id.name,
        ),
    ],
)
def test_map_column(foo_bar_mapper, env, name_in, expected_name_out):
    column_in = varchar_column_spec(name_in)
    column_out = foo_bar_mapper.map_column(env, column_in)
    assert column_out.name == expected_name_out


@pytest.mark.parametrize(
    ["env", "span_name_in", "expected_names_out"],
    [
        ("FOO", "S_FOO", SpanColumns.create("S_FOO").column_names),
        ("FOO", "ENTITY", SpanColumns.create("FOO_ENTITY").column_names),
        ("BAR", "S_BAR", SpanColumns.create("S_BAR").column_names),
        ("BAR", "ENTITY", SpanColumns.create("BAR_ENTITY").column_names),
    ],
)
def test_map_span(foo_bar_mapper, env, span_name_in, expected_names_out):
    span_in = SpanColumns.create(span_name_in)
    span_out = foo_bar_mapper.map_span(env, span_in)
    assert span_out.column_names == expected_names_out


def test_map_same_output():
    """
    This test checks the use case when the outputs come from the same algorithm.
    In such case the outputs are of the same type but have different names in the
    `name` field. We will check that for both outputs the resulting column names
    are prefixed with the output name.
    """
    env1, env2 = "FOO", "BAR"
    name_in = "ENTITY"
    expected_name_out1, expected_name_out2 = "FOO_ENTITY", "BAR_ENTITY"
    foo1 = FooTextOutput(name=env1)
    foo1_env = TextEnvelope.from_text_output(foo1)
    foo2 = FooTextOutput(name=env2)
    foo2_env = TextEnvelope.from_text_output(foo2)
    foo_foo_mapper = TextEnvelopeMapper([foo1_env, foo2_env])
    column_in = varchar_column_spec(name_in)
    column_out1 = foo_foo_mapper.map_column(env1, column_in)
    column_out2 = foo_foo_mapper.map_column(env2, column_in)
    assert column_out1.name == expected_name_out1
    assert column_out2.name == expected_name_out2


def test_map_column_column_error(foo_bar_mapper):
    column_in = varchar_column_spec("NON_EXISTENT")
    with pytest.raises(ValueError):
        foo_bar_mapper.map_column("FOO", column_in)


def test_map_column_env_error(foo_bar_mapper):
    column_in = varchar_column_spec("ENTITY")
    with pytest.raises(ValueError):
        foo_bar_mapper.map_column("UNKNOWN", column_in)


def test_merge(foo_bar_mapper):
    merged_env = foo_bar_mapper.merge("FOO_BAR")
    assert merged_env.name == "FOO_BAR"
    expected_column_names = {
        "C_FOO",
        "FOO_ENTITY",
        *SpanColumns.create("S_FOO").column_names,
        *SpanColumns.create("FOO_ENTITY").column_names,
        "C_BAR",
        "BAR_ENTITY",
        *SpanColumns.create("S_BAR").column_names,
        *SpanColumns.create("BAR_ENTITY").column_names,
    }
    assert len(merged_env.envelope) == 8
    column_names = {col.name for col in merged_env.envelope_order}
    assert column_names == expected_column_names
    assert merged_env.text_span == foo_bar_mapper.text_span
    assert merged_env.error_message == foo_bar_mapper.error_message
    assert merged_env.setup == foo_bar_mapper.setup


def test_merge_with_extras(foo_bar_mapper, foo_output, bar_output):
    merged_env = foo_bar_mapper.merge(
        "FOO_BAR", include_error_message=True, include_setup=True
    )
    assert len(merged_env.envelope) == 12
    column_names = {col.name for col in merged_env.envelope_order}
    assert f"FOO_{foo_output.error_message.name}" in column_names
    assert f"FOO_{foo_output.setup.name}" in column_names
    assert f"BAR_{bar_output.error_message.name}" in column_names
    assert f"BAR_{bar_output.setup.name}" in column_names
