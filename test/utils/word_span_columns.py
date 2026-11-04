from dataclasses import dataclass

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    SpanColumns,
)


@dataclass(frozen=True)
class WordSpanColumns(SpanColumns):

    segment_id: ColumnSpec

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.doc_id, self.segment_id, self.begin, self.end]

    @classmethod
    def common_column_names(cls) -> list[str]:
        return ["DOC_ID", "WORD_BEGIN", "WORD_END", "SEGMENT_ID"]
