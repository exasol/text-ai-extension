from dataclasses import dataclass


@dataclass
class TextTestCase:
    text_doc_id: int
    text_char_begin: int
    text: str
