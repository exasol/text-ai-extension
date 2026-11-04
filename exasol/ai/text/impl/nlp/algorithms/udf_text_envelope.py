from collections import Counter
from dataclasses import (
    dataclass,
    field,
    fields,
)
from itertools import chain

from exasol.analytics.schema import ColumnName

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    OrderedColumns,
    SpanColumns,
    select_column_fields,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextOutput

EnvelopeItem = ColumnSpec | SpanColumns


@dataclass(frozen=True)
class TextEnvelope(TextOutput):
    """
    Universal columns container that any object derived from the TextOutput
    can be converted to.
    The use cases include post-processing when the outputs of two or more algorithms
    are merged together.

    Parameters:
        envelope:
            A plain list of columns and spans. If the TextEnvelope is created from
            another TextOutput this list contains only the elements defined on top
            of the TextOutput, i.e. it doesn't include the `text_span`, `error_message`
            and `setup`.
        original_type:
            The type of the TextOutput the object is converted from. If created manually
            this field should not be set. The None value will indicate that the
            `error_message` and `setup` fields are meaningless and should be ignored.
    """

    envelope: list[EnvelopeItem] = field(default_factory=list)
    original_type: type[TextOutput] | None = None

    @property
    def envelope_order(self) -> list[ColumnSpec]:
        """
        Lists all columns in the envelope, excluding the `text_span`, `error_message`
        and `setup`.
        """
        columns = []
        for item in self.envelope:
            if isinstance(item, ColumnSpec):
                columns.append(item)
            elif isinstance(item, OrderedColumns):
                columns.extend(item.order)
        return columns

    @property
    def order(self) -> list[ColumnSpec]:
        return [
            *self.text_span.order,
            *self.envelope_order,
            self.error_message,
            self.setup,
        ]

    @classmethod
    def from_text_output(cls, text_output: TextOutput):
        """
        Creates the object from the provided TextOutput.
        """
        if isinstance(text_output, TextEnvelope):
            return text_output

        base_fields = set(fields(TextOutput))

        def collect_items(data_class) -> list[EnvelopeItem]:
            env: list[EnvelopeItem] = []
            for fld in fields(text_output):
                if fld not in base_fields:
                    obj = getattr(data_class, fld.name)
                    if isinstance(obj, EnvelopeItem):
                        env.append(obj)
                    elif isinstance(obj, OrderedColumns):
                        env.extend(collect_items(obj))
            return env

        envelope = collect_items(text_output)
        return cls(
            name=text_output.name,
            original_type=type(text_output),
            text_span=text_output.text_span,
            error_message=text_output.error_message,
            setup=text_output.setup,
            envelope=envelope,
        )

    def __post_init__(self):
        # Here we no longer validate the `order`, but still validate the `name`.
        ColumnName(self.name)


class TextEnvelopeMapper:
    """
    This class facilitates merging together multiple TextEnvelope objects. The main
    purpose of it is the name conflict resolution for the columns of the resulting
    TextEnvelope.

    Parameters:
        envelopes:
            The input list of TextEnvelope objects.
        kwargs:
            Parameters to be passed to the constructor of the merged TextEnvelope,
            apart from the `name` and the `envelope`. This can be used to create
            non-standard column names for the `text_span`, `error_message` or `setup`.
    """

    def __init__(self, envelopes: list[TextEnvelope], **kwargs):
        self._envelope_dict = self._create_envelope_dict(envelopes)
        self._base_output = TextOutput(**kwargs)
        self._name_counts = self._create_name_counts(envelopes, self._base_output)

    @staticmethod
    def _create_envelope_dict(envelopes: list[TextEnvelope]) -> dict[str, TextEnvelope]:
        """
        Checks that the names of the input envelopes are unique and puts the envelopes
        into a dictionary indexed by their names.
        """
        env_name_counts = Counter(env.name for env in envelopes)
        if not all(count == 1 for count in env_name_counts.values()):
            non_unique_names = [
                name for name, count in env_name_counts.items() if count > 1
            ]
            raise ValueError(
                f"The names of the {TextEnvelope.__name__} objects "
                f"provided to a {TextEnvelopeMapper.__name__} must be "
                "unique. The following name(s) have duplicates:  "
                f"{non_unique_names}."
            )
        return {env.name: env for env in envelopes}

    @staticmethod
    def _create_name_counts(
        envelopes: list[TextEnvelope], base_output: TextOutput
    ) -> Counter:
        """
        Creates the counters of the column names in all input envelopes.
        """
        return Counter(
            chain(
                base_output.column_names,
                (col.name for env in envelopes for col in env.order),
            )
        )

    def _is_name_unique(self, column: ColumnSpec) -> bool:
        """
        Checks if the provided column name is unique.
        Raises an exception if its unknown (not in the Counter).
        """
        if column.name not in self._name_counts:
            raise ValueError(
                f"An attempt to map a column name {column.name} that is "
                f"not in any of the input {TextEnvelope.__name__}s."
            )
        return self._name_counts[column.name] == 1

    def _get_prefixed_name(self, env: str | TextEnvelope, column: ColumnSpec) -> str:
        """
        Creates a new name for a column based on the name of its envelope. Since the
        envelope names are unique, and the old column name is unique in the scope of
        its envelope, the new name should be unique in the scope of all envelopes.

        Raises an exception if the provided envelope is unknown or the new name for
        whatever reason is not a valid Exasol name (too long perhaps).
        """
        env_name = env if isinstance(env, str) else env.name
        if env_name not in self._envelope_dict:
            raise ValueError(f"Unknown {TextEnvelope.__name__} {env_name}")
        new_name = f"{env_name}_{column.name}"
        # Validate that the proposed new name is still valid Exasol identifier.
        ColumnName(new_name)
        return new_name

    @property
    def text_span(self) -> SpanColumns:
        return self._base_output.text_span

    @property
    def error_message(self) -> ColumnSpec:
        return self._base_output.error_message

    @property
    def setup(self) -> ColumnSpec:
        return self._base_output.setup

    def map_column(self, env: str | TextEnvelope, column: ColumnSpec) -> ColumnSpec:
        """
        Returns the provided column object if its name is unique, otherwise creates and
        returns a copy of the column with the name prefixed by the name of its envelope.
        """
        if self._is_name_unique(column):
            return column
        new_name = self._get_prefixed_name(env, column)
        return column.rename(new_name)

    def map_span(self, env: str | TextEnvelope, span: SpanColumns) -> SpanColumns:
        """
        Returns the provided span if all of its column names are unique.
        Otherwise, creates a new span with all or some of the names prefixed with
        the envelope name.
        """
        if all(self._is_name_unique(col) for col in span.order):
            return span
        column_fields = select_column_fields(fields(span))
        new_span_columns = {
            fld.name: self.map_column(env, getattr(span, fld.name))
            for fld in column_fields
        }
        return type(span)(**new_span_columns, name=span.name)

    def map_item(self, env: str | TextEnvelope, item: EnvelopeItem) -> EnvelopeItem:
        """
        Returns a reference or a copy of the provided envelope item (column or span).
        """
        if isinstance(item, ColumnSpec):
            return self.map_column(env, item)
        elif isinstance(item, SpanColumns):
            return self.map_span(env, item)
        else:
            raise ValueError(
                f"Invalid item of type {type(item).__name__} in a column "
                f"envelope. Allowed types are {ColumnSpec.__name__} and "
                f"{SpanColumns.__name__}"
            )

    def merge(
        self,
        name: str,
        include_error_message: bool = False,
        include_setup: bool = False,
    ) -> TextEnvelope:
        """
        Creates a new TextEnvelope object by merging the input envelopes.
        Ensures that the column names in the result envelope are unique.

        Optionally includes the `error_message` and `setup` columns from the inputs.
        This is not applied to the envelopes with the `original_type` None. It is
        presumed that such envelopes were created manually, hence their `error_message`
        and `setup` are meaningless.
        """
        envelope: list[EnvelopeItem] = []
        for env in self._envelope_dict.values():
            envelope.extend(self.map_item(env, item) for item in env.envelope)
            if env.original_type is not None:
                if include_error_message:
                    envelope.append(self.map_item(env, env.error_message))
                if include_setup:
                    envelope.append(self.map_item(env, env.setup))

        return TextEnvelope(
            name=name,
            text_span=self.text_span,
            error_message=self.error_message,
            setup=self.setup,
            envelope=envelope,
        )
