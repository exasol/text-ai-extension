from dataclasses import dataclass
from textwrap import dedent

from exasol.analytics.schema import SchemaName

_DOC_ID = "doc_id"
_CHAR_BEGIN = "char_begin"
_CHAR_END = "char_end"
_TEXT = "text"
_DOC_ID1 = f"{_DOC_ID}1"
_CHAR_BEGIN1 = f"{_CHAR_BEGIN}1"
_CHAR_END1 = f"{_CHAR_END}1"
_DOC_ID2 = f"{_DOC_ID}2"
_CHAR_BEGIN2 = f"{_CHAR_BEGIN}2"
_CHAR_END2 = f"{_CHAR_END}2"
_SEGMENT_ID = "segment_id"
_SEGMENT_BEGIN = "segment_begin"
_SEGMENT_END = "segment_end"
_SEGMENT_ID1 = f"{_SEGMENT_ID}1"
_SEGMENT_BEGIN1 = f"{_SEGMENT_BEGIN}1"
_SEGMENT_END1 = f"{_SEGMENT_END}1"
_SEGMENT_ID2 = f"{_SEGMENT_ID}2"
_SEGMENT_BEGIN2 = f"{_SEGMENT_BEGIN}2"
_SEGMENT_END2 = f"{_SEGMENT_END}2"


@dataclass(frozen=True)
class TextFunc:
    """
    Declaration of the Text SQL functions.

    General notes about all functions operating on spans:
    - Functions are provided for two kinds of spans - text span and segment spans.
      The meaning of the term "segment" depends on a context. It can, for example, be
      a token, a sentence or a paragraph. A text span includes an ID of the document
      where the text is located. A segment span also includes an ID of the segment,
      e.g. a paragraph in this document.
    - The provided spans are not validated. The assumption is that all numbers in the
      span (`id`, `begin`, `end`) are non-negative integers and the `end` is greater
      than `begin`, in other words the span has a non-zero length. If this condition
      doesn't hold the result will be meaningless.
    - In the parameter description `char_span` is a compact form of writing three
      parameters - `doc_id`, `char_begin`, `char_end`. Similarly, `segment_span` means
      four parameters - `doc_id`, `segment_id`, `segment_begin`, `segment_end`.

    General notes about the span comparison predicates:
    - They follow the framework laid out by the Allen's interval algebra.
      See https://en.wikipedia.org/wiki/Allen%27s_interval_algebra,
      also refer to INDREX: In-database relation extraction
      https://www.cs.toronto.edu/~periklis/pubs/isj15.pdf.
    - All predicates check that the two spans have the same `doc_id`, and in case of
      segment spans, also the same `segment_id`.

    Attributes:
        char_span_valid:
            Call:
                TextFunc.char_span_valid(char_span)
            Checks that the text span is a valid span with greater than zero length.

        char_span_before:
            Call:
                TextFunc.char_span_precedes(char_span1, char_span2)
            This predicate combines two relations of the Allen's interval algebra -
            "precedes" and "meets". It checks that all characters of the text span
            `char_span1` precede the first character of the `char_span2`.

        char_span_contains:
            Call:
                TextFunc.char_span_contains(char_span1, char_span2)
            This predicate combines four relations of the Allen's interval algebra -
            "contains", "finished by", "started by" and "equals". It checks that the
            character positions of the text span `char_span1` include all character
            positions of `char_span2`.

        char_span_overlaps:
            Call:
                TextFunc.char_span_overlaps(char_span1, char_span2)
            This predicate combines two relations of the Allen's interval algebra -
            "overlaps" and "overlapped by". It checks that the two text spans have at
            least one character position in common, but they are not equal and neither
            of them contains the other one.

        char_span_equals:
            Call:
                TextFunc.char_span_equals(char_span1, char_span2)
            Predicate checks that the two text spans are identical.

        char_span_length:
            Call:
                TextFunc.char_span_length(char_span)
            Gets the length of a text span in the number of characters.

        char_span_distance:
            Call:
                TextFunc.char_span_distance(char_span1, char_span2)
            Gets the distance between the two text spans. The distance is defined as
            the number of characters between the end of the first span and the beginning
            of the second. Returns NULL if the first span does not precede the second,
            or their IDs do not match.

        char_span_substring:
            Call:
                TextFunc.char_span_substring(char_span1, text, char_span2)
            Gets the substring of the given text defined by the second span. The first
            span defines the location of the text in the document. Note that both spans
            are normal text spans, i.e. one is not relative to the other.
            Example:
                 TextFunc.char_span_substring(100, 20, 58
                    "Blue frogs danced under the moonlight.",
                    100, 25, 37) == "frogs danced"
            The function returns NULL if the spans have different IDs or the first
            span does not contain the second. The text length is not checked against
            the length of the first span, but if the text is too short the result can
            also be NULL.

        combine_char_span:
            Call:
                TextFunc.combined_char_span(char_span1, char_span2)
            The function returns the shortest span that contains both of the provided
            text spans. It is implemented as a scalar emit UDF. Returns NULL if the IDs
            of the two spans do not match.

        segment_span_valid:
            Call:
                TextFunc.segment_span_valid(segment_span)
            Checks that the segment span is a valid span with greater than zero length.

        segment_span_before:
            Call:
                TextFunc.segment_span_precedes(segment_span1, segment_span2)
            This predicate combines two relations of the Allen's interval algebra -
            "precedes" and "meets". It checks that all segments of the segment span
            `segment_span1` precede the first segment of the `segment_span2`.

        segment_span_contains:
            Call:
                TextFunc.segment_span_contains(segment_span1, segment_span2)
            This predicate combines four relations of the Allen's interval algebra -
            "contains", "finished by", "started by" and "equals". It checks that the
            segment positions of the segment span`segment_span1` include all segment
            positions of `segment_span2`.

        segment_span_overlaps:
            Call:
                TextFunc.segment_span_overlaps(segment_span1, segment_span2)
            This predicate combines two relations of the Allen's interval algebra -
            "overlaps" and "overlapped by". It checks that the two segment spans have at
            least one segment position in common, but they are not equal and neither
            of them contains the other one.

        segment_span_equals:
            Call:
                TextFunc.segment_span_equals(segment_span1, segment_span2)
            Predicate checks that the two segment spans are identical.

        segment_span_length:
            Call:
                TextFunc.segment_span_length(segment_span)
            Gets the length of a segment span in the number of segments.

        segment_span_distance:
            Call:
                TextFunc.segment_span_distance(segment_span1, segment_span2)
            Gets the distance between the two segment spans. The distance is defined as
            the number of segments between the end of the first span and the beginning
            of the second. Returns NULL if the first span does not precede the second,
            or their IDs do not match.

        combine_segment_span:
            Call:
                TextFunc.combined_segment_span(span1, span2)
            Here the span is a combination of the text span and the segment span:
            (`doc_id`, `char_begin`, `char_end`, `segment_id`, `segment_begin`, `segment_end`)
            For each pair of spans - the text and the segment - the function returns
            the shortest span that contains both of the input spans. It is implemented
            as a scalar emit UDF. Returns NULL if the IDs of the two spans do not match.
    """

    char_span_valid: str = "CHAR_SPAN_VALID"
    char_span_before: str = "CHAR_SPAN_BEFORE"
    char_span_contains: str = "CHAR_SPAN_CONTAINS"
    char_span_overlaps: str = "CHAR_SPAN_OVERLAPS"
    char_span_equals: str = "CHAR_SPAN_EQUALS"
    char_span_length: str = "CHAR_SPAN_LENGTH"
    char_span_distance: str = "CHAR_SPAN_DISTANCE"
    char_span_substring: str = "CHAR_SPAN_SUBSTRING"
    combine_char_span: str = "COMBINE_CHAR_SPAN"
    segment_span_valid: str = "SEGMENT_SPAN_VALID"
    segment_span_before: str = "SEGMENT_SPAN_BEFORE"
    segment_span_contains: str = "SEGMENT_SPAN_CONTAINS"
    segment_span_overlaps: str = "SEGMENT_SPAN_OVERLAPS"
    segment_span_equals: str = "SEGMENT_SPAN_EQUALS"
    segment_span_length: str = "SEGMENT_SPAN_LENGTH"
    segment_span_distance: str = "SEGMENT_SPAN_DISTANCE"
    combine_segment_span: str = "COMBINE_SEGMENT_SPAN"


def _build_span_params(*param_names) -> str:

    return ", ".join(f"{p_name} INTEGER" for p_name in param_names)


class Scripts:
    """
    The class is a collection of function and script creation queries.

    The use of a class rather than a dictionary allows customising query
    building in the future.
    """

    def __init__(self, schema: SchemaName | None = None):
        self._schema = "" if schema is None else f"{schema.quoted_name}."

    def _build_predicate(self, func_name: str, in_params: str, expr: str) -> str:
        return dedent(f"""
            CREATE OR REPLACE FUNCTION {self._schema}{func_name}({in_params})
            RETURN BOOLEAN
            IS
                res BOOLEAN;
            BEGIN
                IF {expr} THEN
                    res := TRUE;
                ELSE
                    res := FALSE;
                END IF;
                RETURN res;
            END;
            /
        """)

    def _build_scalar_function(
        self,
        func_name: str,
        in_params: str,
        expr: str,
        cond: str | None = None,
        return_type: str = "INTEGER",
    ) -> str:
        if not cond:
            return dedent(f"""
                CREATE OR REPLACE FUNCTION {self._schema}{func_name}({in_params})
                RETURN {return_type}
                BEGIN
                    RETURN {expr};
                END;
                /
            """)

        return dedent(f"""
            CREATE OR REPLACE FUNCTION {self._schema}{func_name}({in_params})
            RETURN {return_type}
            IS
                res {return_type};
            BEGIN
                IF {cond} THEN
                    res := {expr};
                ELSE
                    res := NULL;
                END IF;
                RETURN res;
            END;
            /
        """)

    @property
    def _char_span_params(self) -> str:
        return _build_span_params(
            _DOC_ID1, _CHAR_BEGIN1, _CHAR_END1, _DOC_ID2, _CHAR_BEGIN2, _CHAR_END2
        )

    @property
    def _segment_span_params(self) -> str:
        return _build_span_params(
            _DOC_ID1,
            _SEGMENT_ID1,
            _SEGMENT_BEGIN1,
            _SEGMENT_END1,
            _DOC_ID2,
            _SEGMENT_ID2,
            _SEGMENT_BEGIN2,
            _SEGMENT_END2,
        )

    @property
    def char_span_valid(self) -> str:
        span_params = _build_span_params(_DOC_ID, _CHAR_BEGIN, _CHAR_END)
        expr = f"{_DOC_ID} >= 0 AND {_CHAR_BEGIN} >= 0 AND {_CHAR_END} > {_CHAR_BEGIN}"
        return self._build_predicate(TextFunc.char_span_valid, span_params, expr)

    @property
    def char_span_before(self) -> str:
        expr = f"{_DOC_ID1} = {_DOC_ID2} AND {_CHAR_END1} <= {_CHAR_BEGIN2}"
        return self._build_predicate(
            TextFunc.char_span_before, self._char_span_params, expr
        )

    @property
    def char_span_contains(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND "
            f"{_CHAR_BEGIN1} <= {_CHAR_BEGIN2} AND {_CHAR_END1} >= {_CHAR_END2}"
        )
        return self._build_predicate(
            TextFunc.char_span_contains, self._char_span_params, expr
        )

    @property
    def char_span_overlaps(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND "
            f"{_CHAR_BEGIN1} < {_CHAR_END2} AND {_CHAR_BEGIN2} < {_CHAR_END1} AND "
            f"({_CHAR_BEGIN1} - {_CHAR_BEGIN2}) * ({_CHAR_END1} - {_CHAR_END2}) > 0"
        )
        return self._build_predicate(
            TextFunc.char_span_overlaps, self._char_span_params, expr
        )

    @property
    def char_span_equals(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND "
            f"{_CHAR_BEGIN1} = {_CHAR_BEGIN2} AND {_CHAR_END1} = {_CHAR_END2}"
        )
        return self._build_predicate(
            TextFunc.char_span_equals, self._char_span_params, expr
        )

    @property
    def char_span_length(self) -> str:
        span_params = _build_span_params(_DOC_ID, _CHAR_BEGIN, _CHAR_END)
        expr = f"{_CHAR_END} - {_CHAR_BEGIN}"
        return self._build_scalar_function(TextFunc.char_span_length, span_params, expr)

    @property
    def char_span_distance(self) -> str:
        expr = f"{_CHAR_BEGIN2} - {_CHAR_END1}"
        cond = f"{_DOC_ID1} = {_DOC_ID2} AND {_CHAR_BEGIN2} >= {_CHAR_END1}"
        return self._build_scalar_function(
            TextFunc.char_span_distance, self._char_span_params, expr, cond
        )

    @property
    def char_span_substring(self) -> str:
        text_type = "VARCHAR(2000000)"
        span_params = (
            f"{_DOC_ID1} INTEGER, {_CHAR_BEGIN1} INTEGER, {_CHAR_END1} INTEGER, "
            f"{_TEXT} {text_type}, "
            f"{_DOC_ID2} INTEGER, {_CHAR_BEGIN2} INTEGER, {_CHAR_END2} INTEGER"
        )
        expr = (
            f"SUBSTRING({_TEXT}, {_CHAR_BEGIN2} - {_CHAR_BEGIN1} + 1, "
            f"{_CHAR_END2} - {_CHAR_BEGIN2})"
        )
        cond = (
            f"{_DOC_ID1} = {_DOC_ID2} AND "
            f"{_CHAR_BEGIN1} <= {_CHAR_BEGIN2} AND {_CHAR_END1} >= {_CHAR_END2}"
        )
        return self._build_scalar_function(
            TextFunc.char_span_substring, span_params, expr, cond, text_type
        )

    @property
    def combine_char_span(self) -> str:
        return dedent(f"""
            CREATE OR REPLACE LUA SCALAR SCRIPT {self._schema}{TextFunc.combine_char_span}(
                    {_DOC_ID1} INTEGER, {_CHAR_BEGIN1} INTEGER, {_CHAR_END1} INTEGER,
                    {_DOC_ID2} INTEGER, {_CHAR_BEGIN2} INTEGER, {_CHAR_END2} INTEGER
            )
            EMITS ({_DOC_ID} INTEGER, {_CHAR_BEGIN} INTEGER, {_CHAR_END} INTEGER) AS
            function run(ctx)
                if ctx.{_DOC_ID1} == ctx.{_DOC_ID2} then
                    ctx.emit(ctx.{_DOC_ID1}, math.min(ctx.{_CHAR_BEGIN1}, ctx.{_CHAR_BEGIN2}), 
                        math.max(ctx.{_CHAR_END1}, ctx.{_CHAR_END2}))
                else
                    ctx.emit(NULL, NULL, NULL)
                end
            end 
            /
        """)

    @property
    def segment_span_valid(self) -> str:
        span_params = _build_span_params(
            _DOC_ID, _SEGMENT_ID, _SEGMENT_BEGIN, _SEGMENT_END
        )
        expr = (
            f"{_DOC_ID} >= 0 AND {_SEGMENT_ID} >= 0 AND "
            f"{_SEGMENT_BEGIN} >= 0 AND {_SEGMENT_END} > {_SEGMENT_BEGIN}"
        )
        return self._build_predicate(TextFunc.segment_span_valid, span_params, expr)

    @property
    def segment_span_before(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND {_SEGMENT_ID1} = {_SEGMENT_ID2} AND "
            f"{_SEGMENT_END1} <= {_SEGMENT_BEGIN2}"
        )
        return self._build_predicate(
            TextFunc.segment_span_before, self._segment_span_params, expr
        )

    @property
    def segment_span_contains(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND {_SEGMENT_ID1} = {_SEGMENT_ID2} AND "
            f"{_SEGMENT_BEGIN1} <= {_SEGMENT_BEGIN2} AND "
            f"{_SEGMENT_END1} >= {_SEGMENT_END2}"
        )
        return self._build_predicate(
            TextFunc.segment_span_contains, self._segment_span_params, expr
        )

    @property
    def segment_span_overlaps(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND {_SEGMENT_ID1} = {_SEGMENT_ID2} AND "
            f"{_SEGMENT_BEGIN1} < {_SEGMENT_END2} AND "
            f"{_SEGMENT_BEGIN2} < {_SEGMENT_END1} AND "
            f"({_SEGMENT_BEGIN1} - {_SEGMENT_BEGIN2}) * "
            f"({_SEGMENT_END1} - {_SEGMENT_END2}) > 0"
        )
        return self._build_predicate(
            TextFunc.segment_span_overlaps, self._segment_span_params, expr
        )

    @property
    def segment_span_equals(self) -> str:
        expr = (
            f"{_DOC_ID1} = {_DOC_ID2} AND {_SEGMENT_ID1} = {_SEGMENT_ID2} AND "
            f"{_SEGMENT_BEGIN1} = {_SEGMENT_BEGIN2} AND "
            f"{_SEGMENT_END1} = {_SEGMENT_END2}"
        )
        return self._build_predicate(
            TextFunc.segment_span_equals, self._segment_span_params, expr
        )

    @property
    def segment_span_length(self) -> str:
        span_params = _build_span_params(
            _DOC_ID, _SEGMENT_ID, _SEGMENT_BEGIN, _SEGMENT_END
        )
        expr = f"{_SEGMENT_END} - {_SEGMENT_BEGIN}"
        return self._build_scalar_function(
            TextFunc.segment_span_length, span_params, expr
        )

    @property
    def segment_span_distance(self) -> str:
        expr = f"{_SEGMENT_BEGIN2} - {_SEGMENT_END1}"
        cond = (
            f"{_DOC_ID1} = {_DOC_ID2}  AND {_SEGMENT_ID1} = {_SEGMENT_ID2} AND "
            f"{_SEGMENT_BEGIN2} >= {_SEGMENT_END1}"
        )
        return self._build_scalar_function(
            TextFunc.segment_span_distance, self._segment_span_params, expr, cond
        )

    @property
    def combine_segment_span(self) -> str:
        return dedent(f"""
            CREATE OR REPLACE LUA SCALAR SCRIPT {self._schema}{TextFunc.combine_segment_span}(
                    {_DOC_ID1} INTEGER, {_CHAR_BEGIN1} INTEGER, {_CHAR_END1} INTEGER,
                    {_SEGMENT_ID1} INTEGER, {_SEGMENT_BEGIN1} INTEGER, {_SEGMENT_END1} INTEGER,
                    {_DOC_ID2} INTEGER, {_CHAR_BEGIN2} INTEGER, {_CHAR_END2} INTEGER,
                    {_SEGMENT_ID2} INTEGER, {_SEGMENT_BEGIN2} INTEGER, {_SEGMENT_END2} INTEGER
            )
            EMITS (
                {_DOC_ID} INTEGER, {_CHAR_BEGIN} INTEGER, {_CHAR_END} INTEGER,
                {_SEGMENT_ID} INTEGER, {_SEGMENT_BEGIN} INTEGER, {_SEGMENT_END} INTEGER
            ) AS
            function run(ctx)
                if ctx.{_DOC_ID1} == ctx.{_DOC_ID2} and ctx.{_SEGMENT_ID1} == ctx.{_SEGMENT_ID2} then
                    ctx.emit(
                        ctx.{_DOC_ID1},
                        math.min(ctx.{_CHAR_BEGIN1}, ctx.{_CHAR_BEGIN2}), 
                        math.max(ctx.{_CHAR_END1}, ctx.{_CHAR_END2}),
                        ctx.{_SEGMENT_ID1},
                        math.min(ctx.{_SEGMENT_BEGIN1}, ctx.{_SEGMENT_BEGIN2}), 
                        math.max(ctx.{_SEGMENT_END1}, ctx.{_SEGMENT_END2})
                    )
                else
                    ctx.emit(NULL, NULL, NULL, NULL)
                end
            end 
            /
        """)
