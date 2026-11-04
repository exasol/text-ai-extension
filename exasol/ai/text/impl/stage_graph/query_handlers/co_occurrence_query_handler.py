from collections import Counter
from dataclasses import dataclass
from itertools import (
    chain,
    combinations,
    count,
    product,
)

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
    LogSpan,
    ModifyQuery,
    SelectQueryWithColumnDefinition,
)
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)
from exasol.analytics.schema import (
    DBObjectNameWithSchemaImpl,
    DbObjectType,
    DbOperationType,
    SchemaName,
    TableLikeName,
)

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput
from exasol.ai.text.impl.nlp.algorithms.udf_text_envelope import (
    TextEnvelope,
    TextEnvelopeMapper,
)
from exasol.ai.text.impl.scripts import (
    Scripts,
    TextFunc,
)
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData
from exasol.ai.text.impl.utils.sql_utils import (
    boolean_column_spec,
    integer_column_spec,
)
from exasol.ai.text.impl.utils.stage_graph_utils import get_input_list


@dataclass(frozen=True)
class DataWithAlias(AlgoIOData):
    """
    Helper data structure, attaching an alias to the input data.
    """

    columns: TextEnvelope
    alias: str


@dataclass(frozen=True)
class ExprColumn:
    """
    Helper data structure, keeping together an SQL expression and the column definition
    for this expression.
    """

    column: ColumnSpec
    expr: str


def find_spans_in_envelope(outputs: TextEnvelope) -> list[SpanColumns]:
    return [item for item in outputs.envelope if isinstance(item, SpanColumns)]


def span_to_str(alias: str, span: SpanColumns) -> str:
    return ",".join(f"{alias}.{name}" for name in span.quoted_column_names)


def _load_stage_inputs(stage_input: SQLStageQueryHandlerInput) -> list[DataWithAlias]:
    """
    Light pre-processing of the input data - converting the column definitions to envelopes
    and attaching an alias to each input.
    """
    input_list = get_input_list(stage_input)
    return [
        DataWithAlias(
            datasource=inp.primary_view,
            columns=TextEnvelope.from_text_output(inp.columns),
            alias=f"T{c + 1}",
        )
        for inp, c in zip(input_list, count())
        if isinstance(inp, AlgoIOData) and isinstance(inp.columns, TextOutput)
    ]


def _schema_prefix(inp: DataWithAlias) -> str:
    if inp.datasource.schema_name is None:
        return ""
    return f"{inp.datasource.schema_name.quoted_name}."


def get_prefixed_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def get_unique_name(prefix: str, name: str, name_counter: Counter) -> str:
    """
    Returns the provided name as is, if it's present in the name counter exactly
    once. Failed that, returns the prefixed name, again if it is present in the
    counter exactly once. Otherwise, throws an exception.
    """
    if name in name_counter and name_counter[name] == 1:
        return name
    prefixed_name = get_prefixed_name(prefix, name)
    if prefixed_name in name_counter and name_counter[prefixed_name] == 1:
        return prefixed_name
    raise ValueError(f"Unable to make a unique name for {prefix}.{name}")


def get_span_name_counter(input_data: list[DataWithAlias]) -> Counter:
    """
    Creates a counter for the names of all spans found in all the provided inputs.
    For each span adds to the counter both the unprefixed name and the span name
    prefixed with the input name.
    """
    return Counter(
        chain(
            *[
                [span.name, get_prefixed_name(inp.columns.name, span.name)]
                for inp in input_data
                for span in find_spans_in_envelope(inp.columns)
            ]
        )
    )


def _is_segment_span(span: SpanColumns) -> bool:
    if type(span) is SpanColumns:
        return False
    if len(span.order) == 4:
        return True
    raise ValueError(f"Invalid span type {type(span).__name__}")


def _get_distance_column(
    inp1: DataWithAlias,
    span1: SpanColumns,
    inp2: DataWithAlias,
    span2: SpanColumns,
    name_counter: Counter,
) -> ExprColumn:
    column_name = (
        f"DISTANCE_{get_unique_name(inp1.columns.name, span1.name, name_counter)}"
        f"_TO_{get_unique_name(inp2.columns.name, span2.name, name_counter)}"
    )
    span_func = (
        TextFunc.segment_span_distance
        if _is_segment_span(span1)
        else TextFunc.char_span_distance
    )
    return ExprColumn(
        column=integer_column_spec(column_name),
        expr=(
            f"{_schema_prefix(inp1)}{span_func}({span_to_str(inp1.alias, span1)},"
            f"{span_to_str(inp2.alias, span2)})"
        ),
    )


def _get_before_column(
    inp1: DataWithAlias,
    span1: SpanColumns,
    inp2: DataWithAlias,
    span2: SpanColumns,
    name_counter: Counter,
) -> ExprColumn:
    column_name = (
        f"{get_unique_name(inp1.columns.name, span1.name, name_counter)}"
        f"_BEFORE_{get_unique_name(inp2.columns.name, span2.name, name_counter)}"
    )
    span_func = (
        TextFunc.segment_span_before
        if _is_segment_span(span1)
        else TextFunc.char_span_before
    )
    return ExprColumn(
        column=boolean_column_spec(column_name),
        expr=(
            f"{_schema_prefix(inp1)}{span_func}({span_to_str(inp1.alias, span1)},"
            f"{span_to_str(inp2.alias, span2)})"
        ),
    )


def _get_contains_column(
    inp1: DataWithAlias,
    span1: SpanColumns,
    inp2: DataWithAlias,
    span2: SpanColumns,
    name_counter: Counter,
) -> ExprColumn:
    column_name = (
        f"{get_unique_name(inp1.columns.name, span1.name, name_counter)}"
        f"_CONTAINS_{get_unique_name(inp2.columns.name, span2.name, name_counter)}"
    )
    span_func = (
        TextFunc.segment_span_contains
        if _is_segment_span(span1)
        else TextFunc.char_span_contains
    )
    return ExprColumn(
        column=boolean_column_spec(column_name),
        expr=(
            f"{_schema_prefix(inp1)}{span_func}({span_to_str(inp1.alias, span1)},"
            f"{span_to_str(inp2.alias, span2)})"
        ),
    )


def build_input_pair_columns(
    inp1: DataWithAlias, inp2: DataWithAlias, name_counter: Counter
) -> list[ExprColumn]:
    """
    Builds the expression columns for all combinations of span pairs found in the
    provided two inputs. The paired spans must come from different inputs and be
    comparable with each other, i.e. have the same type.
    """
    expr_columns: list[ExprColumn] = []
    for span1, span2 in product(
        find_spans_in_envelope(inp1.columns), find_spans_in_envelope(inp2.columns)
    ):
        if type(span1) is type(span2):
            expr_columns.extend(
                [
                    _get_distance_column(inp1, span1, inp2, span2, name_counter),
                    _get_distance_column(inp2, span2, inp1, span1, name_counter),
                    _get_before_column(inp1, span1, inp2, span2, name_counter),
                    _get_before_column(inp2, span2, inp1, span1, name_counter),
                    _get_contains_column(inp1, span1, inp2, span2, name_counter),
                    _get_contains_column(inp2, span2, inp1, span1, name_counter),
                ]
            )
    return expr_columns


def build_expr_columns(input_data: list[DataWithAlias]) -> list[ExprColumn]:
    """
    Builds the expression columns for all possible selections of input pairs.

    The output column names include the span names, which may or may not be
    prefixed with the input name, depending on the uniqueness of the span
    name. In order to control this the Counter of all span names is created.
    """
    name_counter = get_span_name_counter(input_data)
    expr_columns: list[ExprColumn] = []
    if len(input_data) >= 2:
        for combos in combinations(input_data, 2):
            expr_columns.extend(
                build_input_pair_columns(*combos, name_counter=name_counter)
            )
    return expr_columns


def _input_join_condition(inp1: DataWithAlias, inp2: DataWithAlias) -> str:
    """
    For a pair of inputs builds the ON condition of the SQL JOIN clause. The inputs
    a linked on their text spans DOC_ID, not the span!.
    """
    return (
        f"{inp1.alias}.{inp1.columns.text_span.doc_id.quoted_name}="
        f"{inp2.alias}.{inp2.columns.text_span.doc_id.quoted_name}"
    )


def create_func_in_schema_queries(schema: SchemaName, log_span: LogSpan) -> list[Query]:
    """
    Builds queries creating the necessary span functions in the provided schema.
    """
    scripts = Scripts(schema)
    func_names = [
        TextFunc.char_span_distance,
        TextFunc.segment_span_distance,
        TextFunc.char_span_before,
        TextFunc.segment_span_before,
        TextFunc.char_span_contains,
        TextFunc.segment_span_contains,
    ]
    func_bodies = [
        scripts.char_span_distance,
        scripts.segment_span_distance,
        scripts.char_span_before,
        scripts.segment_span_before,
        scripts.char_span_contains,
        scripts.segment_span_contains,
    ]

    return [
        ModifyQuery(
            func_body,
            db_object_type=DbObjectType.FUNCTION,
            db_object_name=DBObjectNameWithSchemaImpl(func_name, schema),
            db_operation_type=DbOperationType.CREATE_OR_REPLACE,
            audit=True,
            parent_log_span=log_span,
        )
        for func_name, func_body in zip(func_names, func_bodies)
    ]


class CoOccurrenceQueryHandler(SQLStageQueryHandler):
    """
    A Query Handler creating the Co-Occurrence views for the input data.

    The Stage expects two or more inputs. The type of the inputs must derive from the
    TextOutput. Normally, this would be outputs of NLP algorithm, connected to this
    Stage either directly or through some post-processing. The inputs must be enclosed
    in a ListInputOutput.

    The Query Handler joins the columns of the input data on their text span and adds
    the span expression columns for all combinations of span pairs it can find in the
    inputs. The spans in a pair must be comparable (have the same type) and come from
    different inputs.

    The expression columns for spans A and B include:
    - distance form A to B
    - distance from B to A
    - A is before B
    - B is before A
    - A contains B
    - B contains A

    The view includes the following columns:
    - Text span, Error Message, Setup, similar any algorithm output.
    - Columns from all inputs apart from their text span (but including the error
      message and setup) The column names are prefixed with the input name where the
      name collision is found.
    - Span expression columns for the three span function, with these name patterns:
      * "DISTANCE_<span_name>_TO_<span_name>"
      * "<span_name>_BEFORE_<span_name>"
      * "<span_name>_CONTAINS_<span_name>"
      The span name will be prefixed with the input name in case the name collision.

    Parameters:
        stage_input:
            The input data, which is expected to have 2+ elements.
        query_handler_context:
            The context object.
        output_view:
            The name of the view to be created by this query handler.
        output_name:
            The name to give to the output object. If not provided the view name will
            be used.
    """

    def __init__(
        self,
        stage_input: SQLStageQueryHandlerInput,
        query_handler_context: ScopeQueryHandlerContext,
        output_view: TableLikeName,
        log_span: LogSpan,
        output_name: str | None = None,
    ) -> None:
        super().__init__(stage_input, query_handler_context)
        self.log_span = log_span
        self.input_data = _load_stage_inputs(stage_input)
        if len(self.input_data) < 2:
            raise ValueError(
                "The Co-Occurrence Query Handler requires two or more "
                f"inputs. Found {len(self.input_data)}"
            )
        self.expr_columns = build_expr_columns(self.input_data)
        self.expr_envelope = TextEnvelope(
            name="EXPR", envelope=[dist.column for dist in self.expr_columns]
        )
        self.mapper = TextEnvelopeMapper(
            [data.columns for data in self.input_data] + [self.expr_envelope]
        )
        self.output_view = output_view
        self.output_name = output_name or output_view.name

    @property
    def create_function_queries(self) -> list[Query]:
        """
        Builds queries creating the required functions in all different schemas that
        can be found in the inputs.
        """
        schemas = {inp.primary_view.schema_name for inp in self.input_data}
        queries: list[Query] = []
        for schema in schemas:
            if schema is not None:
                queries.extend(create_func_in_schema_queries(schema, self.log_span))
        return queries

    @property
    def text_span_columns(self) -> list[str]:
        """
        The text ID column in the SQL SELECT.
        Returned as a list, so it will be easy to change to a full span
        if decided later.
        """
        col_in = self.input_data[0].columns.text_span.doc_id.quoted_name
        col_out = self.mapper.text_span.doc_id.quoted_name
        return [f"{self.input_data[0].alias}.{col_in} AS {col_out}"]

    @property
    def algo_columns(self) -> list[str]:
        """
        Columns in the SQL SELECT copied from the inputs.
        """
        return [
            (
                f"{data.alias}.{col.quoted_name} "
                f"AS {self.mapper.map_column(data.columns, col).quoted_name}"
            )
            for data in self.input_data
            for col in data.columns.order
        ]

    @property
    def expression_columns(self) -> list[str]:
        """
        Generated expression columns in the SQL SELECT.
        """
        return [
            (
                f"{expr_col.expr} "
                f"AS {self.mapper.map_column(self.expr_envelope, expr_col.column).quoted_name}"
            )
            for expr_col in self.expr_columns
        ]

    @property
    def from_clause(self) -> str:
        joins = " ".join(
            f"JOIN {inp2.primary_view.fully_qualified} {inp2.alias} "
            f"ON {_input_join_condition(inp1, inp2)}"
            for inp1, inp2 in zip(self.input_data[:-1], self.input_data[1:])
        )
        inp0 = self.input_data[0]
        return f"FROM {inp0.primary_view.fully_qualified} {inp0.alias} {joins}"

    @property
    def create_view_query(self) -> Query:
        """
        Query creating the Co-Occurrence view.
        """
        extra_columns = [
            f"NULL AS {self.mapper.error_message.quoted_name}",
            f"NULL AS {self.mapper.setup.quoted_name}",
        ]
        columns = ", ".join(
            chain(
                self.text_span_columns,
                self.algo_columns,
                self.expression_columns,
                extra_columns,
            )
        )
        query = (
            f"CREATE OR REPLACE VIEW {self.output_view.fully_qualified} AS "
            f"SELECT {columns} {self.from_clause}"
        )
        return ModifyQuery(
            query,
            db_object_type=DbObjectType.VIEW,
            db_object_name=self.output_view,
            db_operation_type=DbOperationType.CREATE_OR_REPLACE,
            audit=True,
            parent_log_span=self.log_span,
        )

    def start(self) -> Continue | Finish[SQLStageInputOutput]:
        output_column = ColumnSpec(name="ONE", sql_type="DECIMAL(1,0)").to_column()
        query_list = self.create_function_queries + [self.create_view_query]
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
        data_output = AlgoIOData(
            datasource=self.output_view, columns=self.mapper.merge(self.output_name)
        )
        return Finish(result=data_output)
