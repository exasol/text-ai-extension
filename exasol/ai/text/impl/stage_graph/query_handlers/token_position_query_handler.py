from dataclasses import (
    dataclass,
    fields,
)
from itertools import chain
from textwrap import dedent
from typing import cast

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
    TableLikeName,
)

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import TokenizerOutput
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
from exasol.ai.text.impl.utils.stage_graph_utils import get_input_list

TOKEN_ID_COLUMN = "TOKEN_ID"
TOKEN_SEQ_COLUMN = "TOKEN_SEQ"
TOKEN_SPAN_SUFFIX = "TOKEN"
HASH_LENGTH = 7


@dataclass(frozen=True)
class TokenSpanColumns(SpanColumns):
    tokenizer_id: ColumnSpec
    """
    The version or a setup hash of the tokenizer that produced the span.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        """See the docstring of the SpanColumns.order."""
        return [self.doc_id, self.tokenizer_id, self.begin, self.end]

    @classmethod
    def common_column_names(cls) -> list[str]:
        """See the docstring of the SpanColumns.common_column_names."""
        return ["DOC_ID", "TOKEN_BEGIN", "TOKEN_END", "TOKEN_VERSION"]


def find_output_spans(outputs: TextOutput) -> list[SpanColumns]:
    """
    Returns the list of spans found in the outputs except the text span.
    """
    return list(
        filter(
            lambda spec: isinstance(spec, SpanColumns) and spec != outputs.text_span,
            (getattr(outputs, fld.name) for fld in fields(outputs)),
        )
    )


def _parse_inputs(
    stage_input: SQLStageQueryHandlerInput,
) -> tuple[AlgoIOData, AlgoIOData]:
    """
    Finds the output from a tokenizer and the output from another algorithm in the
    stage inputs. The other algorithm output must include one or more character spans.

    Raises a ValueError if one of the following conditions is encountered:
    - Invalid stage graph input.
    - No Tokenizer output.
    - Multiple Tokenizer outputs.
    - There is no algorithm output a token position view can be created for.
    """
    input_list = get_input_list(stage_input)
    tokenizer_outputs = [
        cast(AlgoIOData, inp)
        for inp in input_list
        if isinstance(inp, AlgoIOData) and isinstance(inp.columns, TokenizerOutput)
    ]
    other_outputs = [
        cast(AlgoIOData, inp)
        for inp in input_list
        if isinstance(inp, AlgoIOData)
        and isinstance(inp.columns, TextOutput)
        and not isinstance(inp.columns, TokenizerOutput)
        and len(find_output_spans(inp.columns)) >= 1
    ]
    if len(tokenizer_outputs) != 1:
        raise ValueError(
            "Exactly one output from a Tokenizer Extractor is required for "
            f"extracting token positions. Got {len(tokenizer_outputs)}."
        )
    if len(other_outputs) != 1:
        raise ValueError(
            "Exactly one output with one or more spans, other than the "
            "Tokenizer is required for extracting token positions. "
            f"Got {len(other_outputs)}."
        )
    return tokenizer_outputs[0], other_outputs[0]


def get_create_function_query(algo_output: AlgoIOData, log_span: LogSpan) -> Query:
    """
    Gets the query creating the required function in the schema of the
    algorithm output table.
    """
    scripts = Scripts(algo_output.datasource.schema_name)
    func_object_name = DBObjectNameWithSchemaImpl(
        TextFunc.char_span_contains, algo_output.datasource.schema_name
    )
    return ModifyQuery(
        scripts.char_span_contains,
        db_object_type=DbObjectType.FUNCTION,
        db_object_name=func_object_name,
        db_operation_type=DbOperationType.CREATE_OR_REPLACE,
        audit=True,
        parent_log_span=log_span,
    )


def _char_spans_to_token_spans(spans: list[SpanColumns]) -> list[SpanColumns]:
    return [
        TokenSpanColumns.create(f"{span.name}_{TOKEN_SPAN_SUFFIX}") for span in spans
    ]


class TokenPositionQueryHandler(SQLStageQueryHandler):
    """
    A Query Handler creating views of algorithm outputs with added token spans.

    The Stage expects exactly two inputs. These would normally be outputs of NLP
    algorithms. One of them must be a tokenizer. The other one can be, for example, a
    keyword extraction or a named entity recognition. Generally, the algorithms will
    provide text spans of some subtexts, e.g. key phrases. The view will add to that a
    token span. This span identifies the same piece of text as the character span but
    counting tokens instead of characters.

    As an example, consider some entity extraction from the previous paragraph. Let's
    say we are looking for an indication of some quantity. We can find it in the first
    sentence - "two". The output of the algorithm will include the row
    ('exactly two', 18, 29, ...). Here (18, 29) are the (begin, end) of the key phrase's
    character span. Let us assume that a token is simply a word. Then, the correspondent
    row in the view will look like this: ('exactly two', 18, 29, 2, 4, ...), where (2, 4)
    is the added token span. The latter indicates that the key phrase spans the words
    from 2 to (excluding) 4.

    The inputs must be enclosed in a ListInputOutput.

    The token span includes the version id field. This allows keeping separate
    versions of the token span. As the span is determined by the output of the
    tokenizer, the version is derived from the configuration of the tokenizer
    algorithm. The latter can be found in the `setup` column of the Tokenizer output.

    For the sake of performance, the DOC_ID field of the output character span is
    ignored. The DOC_ID of the text span is used instead. These two fields must be the
    same anyway.

    The view output consists of the algorithm output columns and the token span.

    The Query Handler creates a function it uses in the view. The function is created
    in the schema of the output table of the algorithm the view is created for. If
    the schema is None, the function is created in the current schema, which in this
    case must be opened.

    Parameters:
        stage_input:
            The input data, which is expected to have exactly 2 elements enclosed in
            a ListInputOutput.
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
        self.output_view = output_view
        self.output_name = output_name or output_view.name
        self.tokenizer_output, self.algo_output = _parse_inputs(stage_input)
        algo_outputs = cast(TextOutput, self.algo_output.columns)
        tokenizer_outputs = cast(TokenizerOutput, self.tokenizer_output.columns)

        # Put the algo outputs into an envelope
        self.algo_envelope = TextEnvelope.from_text_output(algo_outputs)

        # Find all character spans in the algorithm output
        self.algo_spans = find_output_spans(algo_outputs)

        # Create token spans for all character span in the algo output and put them
        # into a separate envelope.
        token_spans = _char_spans_to_token_spans(self.algo_spans)
        self.token_envelope = TextEnvelope(
            name=tokenizer_outputs.name,
            text_span=tokenizer_outputs.text_span,
            envelope=token_spans,  # type: ignore
        )

        # Create a name mapper with two envelopes - algo outputs and the token span,
        # to use for resolving the name collisions.
        self.mapper = TextEnvelopeMapper([self.algo_envelope, self.token_envelope])

        # Map the token spans to make sure their column names are unique.
        self.token_spans = [
            self.mapper.map_span(self.token_envelope, token_span)
            for token_span in token_spans
        ]

    @property
    def _token_id_query(self) -> str:
        """
        Query selecting the character spans of the tokens and assigning sequential
        numbers to them. These numbers will form the token spans.
        """
        token_outputs = cast(TokenizerOutput, self.tokenizer_output.columns)
        txt_doc_id = token_outputs.text_span.doc_id.quoted_name
        tok_doc_id, tok_begin, tok_end = token_outputs.token_span.quoted_column_names
        hash_format = "x" * HASH_LENGTH
        setup = token_outputs.setup.quoted_name
        # The token ID is computed as a hash of the tokenizer setup string, which will
        # most likely be a json of its configuration.
        return dedent(f"""
            SELECT
                {txt_doc_id} AS {tok_doc_id},
                {tok_begin},
                {tok_end},
                TO_NUMBER(RIGHT(HASH_TIGER({setup}), {HASH_LENGTH}), '{hash_format}') AS "{TOKEN_SEQ_COLUMN}",
                (ROW_NUMBER() OVER (PARTITION BY {txt_doc_id} ORDER BY {tok_begin})) - 1 AS "{TOKEN_ID_COLUMN}"
            FROM {self.tokenizer_output.primary_view.fully_qualified}
        """)

    def _get_token_span_query(
        self, algo_span: SpanColumns, token_span: SpanColumns
    ) -> str:
        """
        Builds a subquery constructing a token span for a given output span (e.g.
        entity span). It assumes the existence of another subquery, built by
        `_token_id_query`, with the alias "T0".
        """
        algo_outputs = cast(TextOutput, self.algo_output.columns)
        # text span
        txt_doc_id, txt_begin, txt_end = algo_outputs.text_span.quoted_column_names
        txt_doc_id_x, txt_begin_x, txt_end_x = self.mapper.text_span.quoted_column_names
        # character spans columns in the algorithm output
        _, algo_begin, algo_end = algo_span.quoted_column_names
        _, algo_begin_x, algo_end_x = self.mapper.map_span(
            self.algo_envelope, algo_span
        ).quoted_column_names
        token_outputs = cast(TokenizerOutput, self.tokenizer_output.columns)
        # token spans columns
        tok_doc_id, tok_begin, tok_end = token_outputs.token_span.quoted_column_names
        tok_doc_id_x, tok_ver, tok_begin_x, tok_end_x = token_span.quoted_column_names
        schema_prefix = (
            ""
            if self.algo_output.datasource.schema_name is None
            else f"{self.algo_output.datasource.schema_name.quoted_name}."
        )
        span_func = f"{schema_prefix}{TextFunc.char_span_contains}"

        return dedent(f"""
            SELECT
                T1.{txt_doc_id} AS {txt_doc_id_x},
                FIRST_VALUE(T1.{txt_begin}) AS {txt_begin_x},
                FIRST_VALUE(T1.{txt_end}) AS {txt_end_x},
                T1.{algo_begin} AS {algo_begin_x}, 
                T1.{algo_end} AS {algo_end_x},
                T1.{txt_doc_id} AS {tok_doc_id_x},
                T0."{TOKEN_SEQ_COLUMN}" AS {tok_ver},
                MIN(T0."{TOKEN_ID_COLUMN}") AS {tok_begin_x},
                MAX(T0."{TOKEN_ID_COLUMN}") + 1 AS {tok_end_x}
            FROM
                {self.algo_output.primary_view.fully_qualified} T1
                LEFT OUTER JOIN T0
                ON T1.{txt_doc_id} = T0.{tok_doc_id} AND {span_func}(
                    T1.{txt_doc_id}, T1.{algo_begin}, T1.{algo_end}, 
                    T0.{tok_doc_id}, T0.{tok_begin}, T0.{tok_end})
            GROUP BY T1.{txt_doc_id}, T1.{algo_begin}, T1.{algo_end}, T0."{TOKEN_SEQ_COLUMN}"
        """)

    @property
    def create_view_query(self) -> Query:
        """
        Builds the query creating the token position view.
        """
        algo_outputs = cast(TextOutput, self.algo_output.columns)
        seg_aliases = [f"T{n + 2}" for n, _ in enumerate(self.algo_spans)]

        def get_join_condition(algo_span: SpanColumns, alias: str) -> str:
            """
            Creates the join condition linking the main select with the subquery for the
            output span (e.g. entity span) on the text span and (begin,end) of the output
            span.
            """
            return " AND ".join(
                f"T1.{col_in.quoted_name}={alias}.{col_out.quoted_name}"
                for col_in, col_out in chain(
                    zip(algo_outputs.text_span.order, self.mapper.text_span.order),
                    zip(
                        algo_span.order[1:],
                        self.mapper.map_span(self.algo_envelope, algo_span).order[1:],
                    ),
                )
            )

        query_joins = " ".join(
            (
                f"JOIN ({self._get_token_span_query(algo_span, token_span)}) {alias} "
                f"ON {get_join_condition(algo_span, alias)}"
            )
            for algo_span, token_span, alias in zip(
                self.algo_spans, self.token_spans, seg_aliases
            )
        )
        text_span_columns = ", ".join(
            f"T1.{col_in} AS {col_out}"
            for col_in, col_out in zip(
                algo_outputs.text_span.quoted_column_names,
                self.mapper.text_span.quoted_column_names,
            )
        )
        algo_columns = ", ".join(
            f"T1.{col.quoted_name} AS "
            f"{self.mapper.map_column(self.algo_envelope, col).quoted_name}"
            for col in self.algo_envelope.envelope_order
        )
        token_span_columns = ", ".join(
            f"{alias}.{col_name}"
            for token_span, alias in zip(self.token_spans, seg_aliases)
            for col_name in token_span.quoted_column_names
        )
        query = dedent(f"""
            CREATE OR REPLACE VIEW {self.output_view.fully_qualified} AS
            WITH T0 AS (
                {self._token_id_query}
            )
            SELECT
                {text_span_columns},
                {algo_columns},
                {token_span_columns},
                T1.{algo_outputs.error_message.quoted_name} AS {self.mapper.error_message.quoted_name},
                T1.{algo_outputs.setup.quoted_name} AS {self.mapper.setup.quoted_name}
            FROM
                {self.algo_output.primary_view.fully_qualified} T1
                {query_joins}
        """)
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
        query_list = [
            get_create_function_query(self.algo_output, self.log_span),
            self.create_view_query,
        ]

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
