from dataclasses import dataclass

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    SpanColumns,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.sql_utils import (
    double_column_spec,
    varchar_column_spec,
)

_default_entity = varchar_column_spec("ENTITY")


@dataclass(frozen=True)
class NamedEntityOutput(TextOutput):
    name: str = "ENTITY"

    entity_type: ColumnSpec = varchar_column_spec(
        f"{_default_entity.name}_TYPE", flags=IOColumnFlags.Normalisation
    )
    """
    The type of the most likely entity, or NULL in case of an error. For example, in the
    sentence "23-year-old woman ..." the first entity type is "Age", the second is "Sex"
    or "Gender".
    """

    entity_score: ColumnSpec = double_column_spec(f"{_default_entity.name}_SCORE")
    """
    The score of the most likely entity, or NULL in case of an error.
    """

    entity: ColumnSpec = _default_entity
    """
    The named entity text, or NULL in case of an error. For example, in the sentence
    "23-year-old woman ..." the first entity is "23-year-old" (entity type is "Age"),
    the second entity is "woman" (entity type is "Gender").
    """

    entity_span: SpanColumns = SpanColumns.create(_default_entity.name)
    """
    The character span of the entity. The span is relative to the source document, not
    to the text chunk the entity is extracted from.
    """


# noinspection PyAbstractClass
class NamedEntityRecognition(UDFTextBase):
    """
    The base class for named entity recognition algorithms.
    """

    outputs: NamedEntityOutput = NamedEntityOutput()

    @property
    def entity_column(self) -> str:
        return self.outputs.entity.name

    @property
    def entity_type_column(self) -> str:
        return self.outputs.entity_type.name

    @property
    def score_column(self) -> str:
        return self.outputs.entity_score.name
