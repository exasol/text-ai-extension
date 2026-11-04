from dataclasses import dataclass


@dataclass(frozen=True)
class CharSpan:
    doc_id: int
    char_begin: int
    char_end: int
