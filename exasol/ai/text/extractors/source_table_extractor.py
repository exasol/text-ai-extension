from enum import (
    Enum,
    auto,
)
from typing import (
    Annotated,
    Literal,
)

from annotated_types import Len
from pydantic import Field

from exasol.ai.text.extractors.base_extractor import SourceExtractor
from exasol.ai.text.extractors.base_model import TextAIBase


class SelectorPredicate(Enum):
    """
    Predicate to match the name of a database object - a table or a column.
    This can be either exact match, or LIKE (see
    https://docs.exasol.com/db/latest/sql_references/predicates/not_like.htm),
    or REGEXP_LIKE (see
    https://docs.exasol.com/db/latest/sql_references/predicates/not_regexp_like.htm).
    """

    EXACT = auto()
    LIKE = auto()
    REGEXP_LIKE = auto()


class NameSelector(TextAIBase):
    pattern: str
    predicate: SelectorPredicate = SelectorPredicate.EXACT


class TableSource(TextAIBase):
    """
    Definition of source data located in a single table.

    Parameters:
        table:
            Table name or a name pattern
        columns:
            A collection of the source text column selectors - the column names
            or name patterns.
        keys:
            An optional collection of the table key selectors - the key column names
            or name patterns. If the table is in a regular database schema this
            parameter can be omitted, in which case the table primary keys will be used.
            In case of a virtual schema the keys have to be defined explicitly since a
            table in a virtual schema does not have primary keys.
    """

    table: NameSelector
    columns: Annotated[list[NameSelector], Len(min_length=1)]
    keys: list[NameSelector] | None = None


class SchemaSource(TextAIBase):
    """
    Definition of source data located in a single schema.

    Parameters:
        db_schema:
            The schema name.
        tables:
            A collection of tables with source data within the schema.
    """

    db_schema: NameSelector
    tables: Annotated[list[TableSource], Len(min_length=1)]


class SourceTableExtractor(SourceExtractor):
    extractor_type: Literal["SourceTableExtractor"] = Field(
        default="SourceTableExtractor", repr=False
    )
    sources: Annotated[list[SchemaSource], Len(min_length=1)]
