from dataclasses import (
    dataclass,
    fields,
)

import pytest

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    OrderedColumns,
    SpanColumns,
    select_column_fields,
)
from exasol.ai.text.impl.utils.sql_utils import (
    integer_column_spec,
    varchar_column_spec,
)


@dataclass(frozen=True)
class DummyOrderedColumns(OrderedColumns):
    dummy_column1: ColumnSpec
    dummy_column2: ColumnSpec
    something_else: str = "whatever"

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.dummy_column1, self.dummy_column2]


@dataclass(frozen=True)
class GoodOrderedColumns(OrderedColumns):
    dummy_columns: DummyOrderedColumns
    another_column: ColumnSpec

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.another_column, *self.dummy_columns.order]


@dataclass(frozen=True)
class MissingColumnInOrderI(OrderedColumns):
    dummy_columns: DummyOrderedColumns
    another_column: ColumnSpec

    @property
    def order(self) -> list[ColumnSpec]:
        # Missed another_column
        return self.dummy_columns.order


@dataclass(frozen=True)
class MissingColumnInOrderII(OrderedColumns):
    dummy_columns: DummyOrderedColumns
    another_column: ColumnSpec

    @property
    def order(self) -> list[ColumnSpec]:
        # Missed dummy_columns.dummy_column1
        return [self.another_column, self.dummy_columns.dummy_column2]


@dataclass(frozen=True)
class ExtraColumnInOrder(OrderedColumns):
    dummy_columns: DummyOrderedColumns
    another_column: ColumnSpec

    @property
    def order(self) -> list[ColumnSpec]:
        # Non-existing column
        return [
            self.another_column,
            varchar_column_spec("GHOST"),
            *self.dummy_columns.order,
        ]


def _get_dummy_columns() -> DummyOrderedColumns:
    return DummyOrderedColumns(
        name="DUMMY",
        dummy_column1=varchar_column_spec("VARCHAR_DUMMY"),
        dummy_column2=integer_column_spec("INT_DUMMY"),
    )


def _get_another_column() -> ColumnSpec:
    return varchar_column_spec("SOMETHING_ELSE")


@pytest.mark.parametrize(
    ["input_type", "output_type_name", "output_type_args"],
    [
        ("VARCHAR(100)", "VARCHAR", {"size": 100}),
        ("INTEGER", "INTEGER", None),
        ("DOUBLE PRECISION", "DOUBLE PRECISION", None),
        ("DECIMAL(10, 5)", "DECIMAL", {"precision": 10, "scale": 5}),
        ("DECIMAL(12)", "DECIMAL", {"precision": 12, "scale": None}),
    ],
    ids=[
        "VARCHAR(100)",
        "INTEGER",
        "DOUBLE PRECISION",
        "DECIMAL(10, 5)",
        "DECIMAL(12)",
    ],
)
def test_column_spec_to_column(input_type, output_type_name, output_type_args):
    """
    Validates the conversion of a ColumnSpec to a Column
    """
    column_name = "my_column"
    column_spec = ColumnSpec(name=column_name, sql_type=input_type)
    column = column_spec.to_column()
    assert column.name.name == column_name
    assert column.type.name == output_type_name
    if output_type_args:
        assert all(
            getattr(column.type, arg_name) == arg_value
            for arg_name, arg_value in output_type_args.items()
        )


def test_ordered_column():
    GoodOrderedColumns(
        name="DUMMY",
        dummy_columns=_get_dummy_columns(),
        another_column=_get_another_column(),
    )
    assert True


@pytest.mark.parametrize(
    "test_class",
    [MissingColumnInOrderI, MissingColumnInOrderII, ExtraColumnInOrder],
    ids=["missing column", "missing nested column", "extra column"],
)
def test_ordered_column_error(test_class):
    with pytest.raises(Exception):
        test_class(
            name="DUMMY",
            dummy_column=_get_dummy_columns(),
            another_column=_get_another_column(),
        )


def test_select_column_fields_class():
    expected_fields = [
        DummyOrderedColumns.__dataclass_fields__[field_name]
        for field_name in ["dummy_column1", "dummy_column2"]
    ]
    column_fields = select_column_fields(fields(DummyOrderedColumns))
    assert column_fields == expected_fields


def test_select_column_fields_instance():
    expected_fields = [
        DummyOrderedColumns.__dataclass_fields__[field_name]
        for field_name in ["dummy_column1", "dummy_column2"]
    ]
    dummy_columns = _get_dummy_columns()
    column_fields = select_column_fields(fields(dummy_columns))
    assert column_fields == expected_fields


def test_span_column_names():
    span = SpanColumns.create("abc")
    common_names = SpanColumns.common_column_names()
    expected_names = [f"abc_{name}" for name in common_names]
    assert span.column_names == expected_names


@pytest.mark.parametrize("key_value", [False, True])
def test_span_key(key_value):
    span = SpanColumns.create("abc", is_key=key_value)
    assert all(col.is_key == key_value for col in span.order)


@pytest.mark.parametrize(
    ["flags", "expected_flags"],
    [
        (
            IOColumnFlags.NONE,
            [IOColumnFlags.NONE, IOColumnFlags.NONE, IOColumnFlags.NONE],
        ),
        (
            IOColumnFlags.Normalisation,
            [IOColumnFlags.Normalisation, IOColumnFlags.NONE, IOColumnFlags.NONE],
        ),
        (
            [
                IOColumnFlags.Normalisation,
                IOColumnFlags.Distribution,
                IOColumnFlags.Partitioning,
            ],
            [
                IOColumnFlags.Normalisation,
                IOColumnFlags.Distribution,
                IOColumnFlags.Partitioning,
            ],
        ),
    ],
    ids=["none", "scalar", "list"],
)
def test_span_flags(flags, expected_flags):
    span = SpanColumns.create("abc", flags=flags)
    assert [col.flags for col in span.order] == expected_flags
