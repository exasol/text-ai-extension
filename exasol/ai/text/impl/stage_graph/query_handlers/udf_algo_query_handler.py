from collections import defaultdict
from textwrap import dedent

from exasol.analytics.audit.columns import BaseAuditColumns
from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.graph.stage.sql.sql_stage_query_handler import (
    SQLStageQueryHandler,
    SQLStageQueryHandlerInput,
)
from exasol.analytics.query_handler.query.interface import Query
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query.select import (
    AuditQuery,
    LogSpan,
    ModifyQuery,
    SelectQuery,
    SelectQueryWithColumnDefinition,
)
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)
from exasol.analytics.schema import (
    DbObjectType,
    DbOperationType,
    TableLikeName,
    TableLikeNameImpl,
)

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
)
from exasol.ai.text.impl.nlp.udf_factory import UDFFactory
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import (
    AlgoNormalisedIOData,
)
from exasol.ai.text.impl.utils.sql_utils import (
    column_decl_to_str,
    integer_column_spec,
    sql_text_value,
)

LOOKUP_ID_COLUMN = "ID"
ROWID_COLUMN = "ROW_ID"


def _is_normalisable(col: ColumnSpec) -> bool:
    return bool(col.flags & IOColumnFlags.Normalisation)


class _TableAlias:
    """
    A helper class that maps a table name to an alias.
    """

    def __init__(self):
        self._tbl_count = 0
        self._tbl_aliases: dict[TableLikeName, str] = defaultdict(self._new_alias)

    def _new_alias(self) -> str:
        self._tbl_count = self._tbl_count + 1
        return f"T{self._tbl_count}"

    def __getitem__(self, name: TableLikeName) -> str:
        return self._tbl_aliases[name]


def get_output_view(output_table: TableLikeName) -> TableLikeName:
    return TableLikeNameImpl(f"{output_table.name}_VIEW", output_table.schema_name)


def get_lookup_table(output_table: TableLikeName, col: ColumnSpec) -> TableLikeName:
    return TableLikeNameImpl(
        f"{output_table.name}_LOOKUP_{col.name.upper()}", output_table.schema_name
    )


class UDFAlgoQueryHandler(SQLStageQueryHandler):
    """
    Implements a node-level query handler that is a part of an AAF sql graph.
    The node represents a UDF running an algorithm or a pipeline of algorithms.

    The UDF takes the input data from a table or a view specified in the node input.

    The query handler and the sql graph where it belongs are expected to run
    repeatedly, processing continuously updated input data. The algorithm is likely to
    be computationally expensive. Therefore, it is important to persist the results and
    avoid running the algorithm repeatedly on the data that has already been processed.
    To facilitate this the algorithm should define a set of matching input and output
    keys. The query handler will identify the new input rows and run udf only on that
    rows. An error will be raised if the keys are not defined or do not match.

    At the very minimum, the query handler performs the following steps:
    - make sure the output table exists,
    - create the UDF,
    - run the UDF,
    - delete the UDF,
    - commit the results.

    If the output data is to be normalised the "run UDF" step will include the following
    sub-steps:
    1. Write the output of the UDF into a temporary table.
    2. Extract unique values from columns marked with the Normalisation flag, e.g.
       "topic", into separate lookup tables which have IDENTITY PRIMARY KEY. The lookup
       tables are persistent and may already contain all or some of that values.
    3. Copy data from the temporary table into the output table replacing values in the
       normalised columns with the correspondent integer keys.
    4. Deleting the temporary table.
    5. Creating a view, joining the output table with the lookup tables.

    To see the examples of generated queries please refer to the unit tests.

    The node doesn't produce any output, returning a dummy select query.

    Parameters:
        udf_runner:
            The configured algorithm or a pipeline of algorithms.
        input_table:
            The name of the table containing the input data.
        output_table:
            The name of a table where the output should be stored.
        language_alias:
            The UDF language alias.
        parameter:
            The input data. It is expected to come from a single source, hence the
            `sql_stage_inputs` must have exactly one element, an instance of the
            `AlgoStageInputOutput` class.
            The query handler does not store any data into the BucketFS, therefore it
            ignores the `result_bucketfs_location` value.
        query_handler_context:
            The context object.
        parallelism_per_node:
            Number of UDF clients to run at each node during the call to the algorithm
            UDF.
    """

    def __init__(
        self,
        udf_runner: UDFRunner,
        input_table: TableLikeName,
        output_table: TableLikeName,
        language_alias: str,
        parameter: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
        parallelism_per_node: int,
        log_span: LogSpan,
    ) -> None:
        super().__init__(parameter, query_handler_context)
        self.log_span = log_span
        self.udf_runner = udf_runner
        # We need to set an explicit `setup` containing the json of the algorithm
        # settings. It must be the same string we use for data filtering in the SQL
        # query.
        self.algo_setup = self.udf_runner.algo_setup
        self.udf_runner.setup = self.algo_setup
        self.input_table = input_table
        self.output_table = output_table
        self.parameter = parameter
        self.parallelism_per_node = parallelism_per_node
        # Arrange a name for a temporary table to store the UDF output before the
        # normalisation, in case it is needed. At this point the table is not created.
        self.tmp_output_table = query_handler_context.get_temporary_table_name()

        self.udf_factory = UDFFactory(
            udf_name=query_handler_context.get_temporary_udf_name(),
            language_alias=language_alias,
        )

    def _key_clause(self, clause: str, flag: IOColumnFlags) -> str:
        """
        Returns ' <clause> column1, column2, ...' for all columns with the
        specified flag, or an empty string if there are no such columns.
        """
        keys = ", ".join(
            f"{col.quoted_name}"
            for col in self.udf_runner.output_decl
            if col.flags & flag
        )
        return f", {clause} {keys}" if keys else ""

    @property
    def normalised(self) -> bool:
        """
        An indicator if the output should be normalised.
        """
        return any(_is_normalisable(col) for col in self.udf_runner.output_decl)

    @property
    def output_view(self) -> TableLikeName:
        """
        Name of a de-normalising view.
        """
        return get_output_view(self.output_table)

    def get_lookup_table(self, col: ColumnSpec) -> TableLikeName:
        """
        Creates a name of the lookup table for a column with the Normalisation flag.
        """
        if not _is_normalisable(col):
            raise RuntimeError(
                "An attempt to create a lookup table for the column "
                f"{col.name} that is not marked as Normalised."
            )
        return get_lookup_table(self.output_table, col)

    @property
    def output_list(self) -> str:
        """
        List of quoted names of the output table columns.
        """
        return ", ".join(self.udf_runner.algorithm.outputs.quoted_column_names)

    @property
    def create_output_query(self) -> Query:
        """
        Query creating the output table if it doesn't exist.
        """

        def convert_column(col: ColumnSpec) -> ColumnSpec:
            """Changes the type of normalisable columns to INTEGER"""
            if _is_normalisable(col):
                return integer_column_spec(col.name)
            return col

        modified_decl = [convert_column(col) for col in self.udf_runner.output_decl]
        query = (
            f"CREATE TABLE IF NOT EXISTS {self.output_table.fully_qualified}"
            f"({column_decl_to_str(modified_decl)}"
            f"{self._key_clause('DISTRIBUTE BY', IOColumnFlags.Distribution)}"
            f"{self._key_clause('PARTITION BY', IOColumnFlags.Partitioning)})"
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.output_table,
            db_operation_type=DbOperationType.CREATE_IF_NOT_EXISTS,
            audit=True,
            parent_log_span=self.log_span,
        )

    @property
    def create_tmp_output_query(self) -> Query:
        """
        Query creating a temporary table for storing the output of a single call to
        the UDF. This table should not persist, so "IF NOT EXISTS" is not required.
        """
        query = (
            f"CREATE OR REPLACE TABLE {self.tmp_output_table.fully_qualified}"
            f"({column_decl_to_str(self.udf_runner.output_decl)}"
            f"{self._key_clause('DISTRIBUTE BY', IOColumnFlags.Distribution)})"
        )
        return SelectQuery(query)

    def get_create_lookup_table_query(self, col: ColumnSpec) -> Query:
        """
        Builds a query creating a lookup table for the specified column.
        """
        id_col = integer_column_spec(LOOKUP_ID_COLUMN)
        tbl = self.get_lookup_table(col)
        query = (
            f"CREATE TABLE IF NOT EXISTS {tbl.fully_qualified}"
            f"({id_col.quoted_name} {id_col.sql_type} IDENTITY PRIMARY KEY, "
            f"{column_decl_to_str([col])})"
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=tbl,
            db_operation_type=DbOperationType.CREATE_IF_NOT_EXISTS,
            audit=True,
            parent_log_span=self.log_span,
        )

    @property
    def create_lookup_table_queries(self) -> list[Query]:
        """
        List of queries creating the lookup tables if they don't exist.
        """
        return [
            self.get_create_lookup_table_query(col)
            for col in self.udf_runner.output_decl
            if _is_normalisable(col)
        ]

    @property
    def create_view_query(self) -> Query:
        """
        Query creating a de-normalising view of the data.
        """
        tbl_alias = _TableAlias()
        output_alias = tbl_alias[self.output_table]

        def input_selector(c: ColumnSpec):
            if _is_normalisable(c):
                return f"{tbl_alias[self.get_lookup_table(c)]}.{c.quoted_name}"
            return f"{output_alias}.{c.quoted_name}"

        select_list = ", ".join(
            input_selector(col) for col in self.udf_runner.output_decl
        )
        statements = [
            (
                f"CREATE OR REPLACE VIEW {self.output_view.fully_qualified} AS "
                f"SELECT {select_list} "
                f"FROM {self.output_table.fully_qualified} {output_alias}"
            )
        ]
        for col in self.udf_runner.output_decl:
            if _is_normalisable(col):
                lookup_tbl = self.get_lookup_table(col)
                lookup_alias = tbl_alias[lookup_tbl]
                statements.append(
                    f"LEFT OUTER JOIN {lookup_tbl.fully_qualified} {lookup_alias} "
                    f"ON {output_alias}.{col.quoted_name} = "
                    f'{lookup_alias}."{LOOKUP_ID_COLUMN}"'
                )
        query = " ".join(statements)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.VIEW,
            db_object_name=self.output_view,
            db_operation_type=DbOperationType.CREATE_OR_REPLACE,
            audit=True,
            parent_log_span=self.log_span,
        )

    @property
    def create_udf_query(self) -> Query:
        """
        Query creating a UDF to run the algorithm.
        """
        query = self.udf_factory.create_udf_query(self.udf_runner)
        return SelectQuery(query)

    @property
    def delete_udf_query(self) -> Query:
        """
        Query deleting the UDF.
        """
        query = self.udf_factory.delete_udf_query()
        return SelectQuery(query)

    def _get_known_keys_query(
        self, keys: list[ColumnSpec], tbl_alias: _TableAlias
    ) -> str:
        """
        Constructs a query that fetches the keys of the previously processed output
        data. Only the data processed with the same algorithm setup as currently in
        use is selected.
        """
        output_alias = tbl_alias[self.output_table]
        setup = sql_text_value(self.algo_setup)
        setup_col = self.udf_runner.algorithm.outputs.setup
        setup_normalised = _is_normalisable(setup_col)
        setup_tbl = (
            self.get_lookup_table(setup_col) if setup_normalised else self.output_table
        )
        setup_alias = tbl_alias[setup_tbl]

        if setup_normalised:
            join_clause = (
                f"JOIN {setup_tbl.fully_qualified} {setup_alias} ON "
                f"{output_alias}.{setup_col.quoted_name}="
                f'{setup_alias}."{LOOKUP_ID_COLUMN}" '
            )
        else:
            join_clause = ""
        where_clause = f"{setup_alias}.{setup_col.quoted_name}={setup}"
        key_columns = ", ".join(key.quoted_name for key in keys)
        query = (
            f"SELECT {key_columns} "
            f"FROM {self.output_table.fully_qualified} {output_alias} {join_clause}"
            f"WHERE {where_clause}"
        )
        return query

    @property
    def run_udf_query(self) -> Query:
        """
        Query running the UDF.
        Depending on whether the normalisation is required or not the output is
        written either to a temporary table or directly to the output table.
        """

        # Check that the algorith defines matching input and output keys
        # and that the keys are not normalisable.
        keys = [col for col in self.udf_runner.input_decl if col.is_key]
        output_keys = {col for col in self.udf_runner.output_decl if col.is_key}
        if (not keys) or (not output_keys.issuperset(keys)):
            raise RuntimeError(
                "UDFRunner does not have key columns or there is a "
                "mismatch between input and output key columns"
            )
        if any(_is_normalisable(key) for key in keys):
            raise RuntimeError("Algorithm column defined as a key cannot be normalised")

        tbl_alias = _TableAlias()
        input_alias = tbl_alias[self.input_table]
        # Inner alias has nothing to do with the view, we just need a unique name.
        inner_alias = tbl_alias[self.output_view]
        target_tbl = self.tmp_output_table if self.normalised else self.output_table

        inner_query = self._get_known_keys_query(keys, tbl_alias)
        input_params = self.udf_runner.algorithm.inputs.quoted_column_names
        udf_param_list = ", ".join(f"{input_alias}.{inp}" for inp in input_params)
        join_cond = " AND ".join(
            (f"{input_alias}.{key.quoted_name}" f"={inner_alias}.{key.quoted_name}")
            for key in keys
        )
        join_clause = f"LEFT OUTER JOIN {inner_alias} ON {join_cond}"
        where_clause = " AND ".join(
            f"{inner_alias}.{key.quoted_name} IS NULL" for key in keys
        )
        # Here we will use a subquery to select the input data, adding the ROWID.
        # It is needed for controlling the number of UDF clients per node.
        ext_input_query = (
            f'SELECT ROWID AS "{ROWID_COLUMN}", {", ".join(input_params)} '
            f"FROM {self.input_table.fully_qualified}"
        )

        query = dedent(f"""
            INSERT INTO {target_tbl.fully_qualified}({self.output_list})
            WITH {inner_alias} AS ({inner_query})
            WITH {input_alias} AS ({ext_input_query})
            SELECT {self.udf_factory.udf_name.fully_qualified}({udf_param_list})
            FROM {input_alias} {join_clause}
            WHERE {where_clause}
            GROUP BY IPROC(), MOD({input_alias}."{ROWID_COLUMN}", {self.parallelism_per_node})
        """)
        if self.normalised:
            return SelectQuery(query)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.output_table,
            db_operation_type=DbOperationType.INSERT,
            audit=True,
            parent_log_span=self.log_span,
        )

    def get_extract_lookup_query(self, col: ColumnSpec) -> Query:
        """
        Builds a query updating the lookup table for the specified column.
        """
        tbl = self.get_lookup_table(col)
        query = (
            f"MERGE INTO {tbl.fully_qualified} T "
            f"USING (SELECT DISTINCT {col.quoted_name} "
            f"FROM {self.tmp_output_table.fully_qualified} "
            f"WHERE {col.quoted_name} IS NOT NULL) AS S "
            f"ON T.{col.quoted_name} = S.{col.quoted_name} "
            f"WHEN NOT MATCHED THEN INSERT({col.quoted_name}) "
            f"VALUES(S.{col.quoted_name})"
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=tbl,
            db_operation_type=DbOperationType.INSERT,
            audit=True,
            parent_log_span=self.log_span,
        )

    @property
    def extract_lookup_queries(self) -> list[Query]:
        """
        List of queries updating the lookup tables from the data in the temporary table.
        """
        return [
            self.get_extract_lookup_query(col)
            for col in self.udf_runner.output_decl
            if _is_normalisable(col)
        ]

    @property
    def normalise_query(self) -> Query:
        """
        Query inserting normalised rows from temporary table into the output table.
        """
        tbl_alias = _TableAlias()
        tmp_alias = tbl_alias[self.tmp_output_table]

        def input_selector(c: ColumnSpec):
            if _is_normalisable(c):
                return f'{tbl_alias[self.get_lookup_table(c)]}."{LOOKUP_ID_COLUMN}"'
            return f"{tmp_alias}.{c.quoted_name}"

        select_list = ", ".join(
            input_selector(col) for col in self.udf_runner.output_decl
        )
        statements = [
            (
                f"INSERT INTO {self.output_table.fully_qualified}({self.output_list}) "
                f"SELECT {select_list} "
                f"FROM {self.tmp_output_table.fully_qualified} {tmp_alias}"
            )
        ]
        for col in self.udf_runner.output_decl:
            if _is_normalisable(col):
                lookup_tbl = self.get_lookup_table(col)
                lookup_alias = tbl_alias[lookup_tbl]
                statements.append(
                    f"LEFT OUTER JOIN {lookup_tbl.fully_qualified} {lookup_alias} "
                    f"ON {tmp_alias}.{col.quoted_name} = "
                    f"{lookup_alias}.{col.quoted_name}"
                )
        query = " ".join(statements)
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.TABLE,
            db_object_name=self.output_table,
            db_operation_type=DbOperationType.INSERT,
            audit=True,
            parent_log_span=self.log_span,
        )

    @property
    def drop_tmp_output_query(self) -> Query:
        """
        Query deleting the temporary output table.
        """
        query = f"DROP TABLE {self.tmp_output_table.fully_qualified}"
        return SelectQuery(query)

    @property
    def commit_output_query(self) -> Query:
        """
        Query committing the data.
        """
        return SelectQuery("COMMIT")

    @property
    def start_audit_query(self) -> Query:
        return AuditQuery(audit_fields={"EVENT_NAME": "Start"}, log_span=self.log_span)

    @property
    def end_audit_query(self) -> Query:
        return AuditQuery(audit_fields={"EVENT_NAME": "End"}, log_span=self.log_span)

    @property
    def error_audit_query(self) -> Query:
        """
        Query returning the errors from the last execution of the UDF.
        """
        target_column = BaseAuditColumns.ERROR_MESSAGE
        query = (
            f"SELECT DISTINCT {self.udf_runner.algorithm.outputs.error_message.quoted_name} AS"
            f" {target_column.name.quoted_name}"
            f" FROM {self.tmp_output_table.fully_qualified}"
        )
        select_query = SelectQueryWithColumnDefinition(query, [target_column])
        return AuditQuery(
            audit_fields={"EVENT_NAME": "Error"},
            select_with_columns=select_query,
            log_span=self.log_span,
        )

    def start(self) -> Continue | Finish[SQLStageInputOutput]:
        output_column = ColumnSpec(name="ONE", sql_type="DECIMAL(1,0)").to_column()

        query_list = [self.start_audit_query, self.create_output_query]
        if self.normalised:
            query_list.append(self.create_tmp_output_query)
            query_list.extend(self.create_lookup_table_queries)
            query_list.append(self.create_view_query)
        query_list.append(self.create_udf_query)
        query_list.append(self.run_udf_query)
        query_list.append(self.delete_udf_query)
        if self.normalised:
            query_list.extend(self.extract_lookup_queries)
            query_list.append(self.normalise_query)
            query_list.append(self.error_audit_query)
            query_list.append(self.drop_tmp_output_query)
        query_list.append(self.end_audit_query)
        query_list.append(self.commit_output_query)

        return Continue(
            query_list=query_list,
            input_query=SelectQueryWithColumnDefinition(
                query_string=(
                    "SELECT CAST(1 AS DECIMAL(1,0)) "
                    f"AS {output_column.name.quoted_name}"
                ),
                output_columns=[output_column],
            ),
        )

    def handle_query_result(
        self, query_result: QueryResult
    ) -> Continue | Finish[SQLStageInputOutput]:
        if self.normalised:
            output_view = self.output_view
            lookup_tables = {
                col: self.get_lookup_table(col)
                for col in self.udf_runner.output_decl
                if _is_normalisable(col)
            }
        else:
            output_view, lookup_tables = None, None
        algo_output = AlgoNormalisedIOData(
            datasource=self.output_table,
            dataview=output_view,
            lookup_tables=lookup_tables,
            columns=self.udf_runner.algorithm.outputs,
        )
        return Finish(result=algo_output)
