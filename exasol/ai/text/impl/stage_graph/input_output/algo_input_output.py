from dataclasses import dataclass

from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.schema import TableLikeName

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    OrderedColumns,
)


@dataclass(frozen=True)
class AlgoIOData(SQLStageInputOutput):
    """
    An input/output of an algorithm run by a query handler. It includes a datasource
    (i.e. a table) and the column interface definition provided by the algorithm.
    """

    datasource: TableLikeName
    columns: OrderedColumns

    @property
    def primary_view(self) -> TableLikeName:
        """
        This is a TableLike name of a table or a view where the data should normally be
        selected from for further processing. For the base class this is a no brainer,
        as there is only one datasource. However, a derived class my provide a number of
        supporting tables and views. A prime example being the normalised output.
        """
        return self.datasource


@dataclass(frozen=True)
class AlgoNormalisedIOData(AlgoIOData):
    """
    A possibly normalised output of an algorithm run by a query handler. In addition
    to the `datasource` that may contain a normalised data, there are also placeholders
    for a view that links this data to lookup tables and a dictionary of the lookup
    tables. The lookup table dictionary maps a normalised column to the name of the
    correspondent lookup table.
    """

    dataview: TableLikeName | None
    lookup_tables: dict[ColumnSpec, TableLikeName] | None

    @property
    def primary_view(self) -> TableLikeName:
        return self.dataview or self.datasource
