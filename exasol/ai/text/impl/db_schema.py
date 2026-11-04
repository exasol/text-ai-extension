import re
from dataclasses import (
    Field,
    dataclass,
    fields,
)
from enum import (
    IntFlag,
    auto,
)
from typing import (
    Any,
    cast,
)

from exasol.analytics.schema import (
    Column,
    ColumnName,
    ColumnType,
)


class IOColumnFlags(IntFlag):
    """
    The flags hint the desired usage of the column in an input or output table.

    Distribution:
        The column should be part of a distribution key. See
        https://docs.exasol.com/db/latest/performance/best_practices.htm#Distribu
    Partition:
        The column should be part of a partition key. See
        https://docs.exasol.com/db/latest/performance/best_practices.htm#Partitio
        Note that the data types like CHAR or VARCHAR cannot be used for table
        partitioning.
    Normalisation:
        Distinct column values should be extracted into a separate table.
    """

    NONE = 0
    Distribution = auto()
    Partitioning = auto()
    Normalisation = auto()


@dataclass(frozen=True)
class ColumnSpec:
    """
    Pydantic-friendly definition of a column.
    """

    name: str
    """
    Unquoted name of the column.
    """

    sql_type: str
    """
    Native sql type spec, e.g. VARCHAR(1000)
    """

    is_key: bool = False
    """
    An indicator that the column is a key. A collection of keys form an identity
    of the UDF input. This can be used to select a "new" input - i.e. the data the
    operation has not been applied for.
    IMPORTANT! A column designated as a key cannot be normalisable, i.e. it cannot
    have the IOColumnFlags.Normalisation flag.
    """

    flags: IOColumnFlags = IOColumnFlags.NONE
    """
    Column usage flags.
    """

    @property
    def quoted_name(self) -> str:
        return f'"{self.name}"'

    def to_column(self, **type_kwargs) -> Column:
        """
        Converts this class to the `Column` defined in the AAF.

        The type_kwargs is a temporary workaround for the CHARACTER SET.
        See https://github.com/exasol/text-ai-extension/issues/258
        """
        return Column(
            name=ColumnName(self.name), type=self._to_column_type(**type_kwargs)
        )

    def rename(self, new_name: str) -> "ColumnSpec":
        """
        Creates a copy of itself with a new name.
        """
        return ColumnSpec(
            name=new_name, sql_type=self.sql_type, is_key=self.is_key, flags=self.flags
        )

    def __str__(self):
        return f"{self.quoted_name} {self.sql_type}"

    def _to_column_type(self, **type_kwargs) -> ColumnType:
        """
        A simple sql type parser limited to most commonly used types and aliases.
        To be replaced with a `from_sql()` function when it is implemented in the AAF.
        """
        pattern = r"^((?:[a-zA-Z_]+\s*)+)(?:\(((?:\s*[0-9]+\s*,)*\s*[0-9]+)\s*\))?\s*$"
        match = re.fullmatch(pattern, self.sql_type)
        if match is None:
            raise ValueError(f"The SQL type {self.sql_type} is not supported")
        type_name = match.group(1).strip().upper()
        if match.group(2) is not None:
            type_args = [int(arg.strip()) for arg in match.group(2).split(",")]
        else:
            type_args = []
        if type_name in ["VARCHAR", "CHAR"]:
            arg_names = ["size"]
        elif type_name in ["DECIMAL", "TIMESTAMP"]:
            arg_names = ["precision", "scale"]
        else:
            arg_names = []
        kwargs = {arg_name: arg for arg, arg_name in zip(type_args, arg_names)}
        return ColumnType(name=type_name, **kwargs, **type_kwargs)  # type: ignore

    def __post_init__(self):
        # Validate the provided name
        ColumnName(self.name)


ColumnDecl = list[ColumnSpec]


def select_column_fields(data_class_fields: tuple[Field, ...]) -> list[Field]:
    """
    Selects the fields of the type ColumnSpec in the provided list of fields of
    a dataclass or an instance of a dataclass.
    """
    return [
        fld
        for fld in data_class_fields
        if (isinstance(fld.type, str) and fld.type == ColumnSpec.__name__)
        or (isinstance(fld.type, type) and issubclass(fld.type, ColumnSpec))
    ]


@dataclass(frozen=True)
class OrderedColumns:
    """
    A base for a dataclass defining columns, i.e. ColumnSpec fields, and their order.
    The column definition can be nested - a field can be another OrderedColumns object.
    The `order` should list all the columns, including the nested ones.
    """

    name: str
    """
    Name of the column collection.
    This can be useful for resolving name collisions when merging two or more
    collections together.
    Also it can be used for creating a logical link between two collections, for
    example between a character span and a correspondent segment spans. If the latter
    is derived form the former it can inherit the name with added prefix or suffix.
    """

    @property
    def order(self) -> ColumnDecl:
        """
        Expected order of columns in a DataFrame.
        """
        return []

    @property
    def column_names(self) -> list[str]:
        return [col.name for col in self.order]

    @property
    def quoted_column_names(self) -> list[str]:
        return [col.quoted_name for col in self.order]

    def __post_init__(self):
        # Validate the name is a valid Exasol identifier.
        ColumnName(self.name)

        # Validate that all columns are listed in the `order`
        def is_field_instance(
            field_type: Any, field_obj: Any, test_class: type
        ) -> bool:
            """
            The function checks if a field of a dataclass is of a specified type.
            A simple check `isinstance(object, class)` does not always work here.
            The exact reason is unclear, but it is related to how __post_init__ works.
            Thus, the function uses `issubclass(field.type, class)`, which is more
            reliable. But this check does not work if `field.type` is a string, for
            example if the annotation is a string. In this case, the function uses
            `isinstance()` on the field value.

            Parameters:
                field_type:     The type property of the dataclass field. This can be
                                a string.
                field_obj:      The value of this field in a dataclass instance.
                test_class:     The type the field value should be checked against.
            """
            if isinstance(field_type, str):
                return isinstance(field_obj, test_class)
            return issubclass(field_type, test_class)  # type: ignore

        def unique_column_names(dc) -> set[str]:
            columns: set[str] = set()
            for fld in fields(dc):
                obj = getattr(dc, fld.name)
                if is_field_instance(fld.type, obj, OrderedColumns):
                    columns.update(unique_column_names(obj))
                elif is_field_instance(fld.type, obj, ColumnSpec):
                    columns.add(cast(ColumnSpec, obj).name)
            return columns

        all_columns = unique_column_names(self)
        ordered_columns = {col.name for col in self.order}
        missing_columns = all_columns - ordered_columns
        if missing_columns:
            raise ValueError(
                f"{type(self).__name__} defined columns {missing_columns} "
                f"are not ordered"
            )
        extra_columns = ordered_columns - all_columns
        if extra_columns:
            raise ValueError(
                f"{type(self).__name__} ordered columns {extra_columns} "
                f"are not defined"
            )


@dataclass(frozen=True)
class SpanColumns(OrderedColumns):
    """
    The span of a text in a source document. The span would normally indicate the
    location of the text in terms of characters, but other units, such as tokens or
    sentences can also be used.
    """

    doc_id: ColumnSpec
    """
    The ID of the source document.
    """

    begin: ColumnSpec
    """
    Starting index of the text span - the position of the first element (e.g. character)
    of the text chunk identified by the span.
    """

    end: ColumnSpec
    """
    Ending index of the text span - the position after the last element (e.g. character)
    of the text chunk identified by the span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.doc_id, self.begin, self.end]

    @classmethod
    def create(
        cls,
        generic_name: str,
        is_key: bool = False,
        flags: IOColumnFlags | list[IOColumnFlags] = IOColumnFlags.NONE,
    ):
        """
        Creates a new column span using the following name format for the columns:
        "<generic_name>_<common_name>".

        All columns will have the same `is_key` parameter as provided.
        The flags can be provided individually for each column (`doc_id`, `begin`,
        `end`) or just for the `doc_id.` In the latter case the other two columns
        will have no flags.
        """
        column_fields = select_column_fields(fields(cls))
        if not isinstance(flags, list):
            flags = [flags]
        if len(flags) < len(column_fields):
            flags = flags + [IOColumnFlags.NONE] * (len(column_fields) - len(flags))
        columns = {
            fld.name: ColumnSpec(
                name=f"{generic_name}_{name}",
                sql_type="INTEGER",
                is_key=is_key,
                flags=flg,
            )
            for fld, name, flg in zip(column_fields, cls.common_column_names(), flags)
        }
        return cls(**columns, name=generic_name)

    @classmethod
    def common_column_names(cls) -> list[str]:
        """
        These are the common parts of the span column names, same for all spans of
        a given type. The names indicate the span type. The base span class should be
        used for the character spans, hence is the prefix "CHAR".
        """
        return ["DOC_ID", "CHAR_BEGIN", "CHAR_END"]
