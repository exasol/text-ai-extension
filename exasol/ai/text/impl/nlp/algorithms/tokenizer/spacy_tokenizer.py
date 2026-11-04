from typing import (
    Any,
    Literal,
)

import pandas as pd
from spacy import Language
from spacy.tokens import Doc

from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import (
    Token,
    TokenContent,
    Tokenizer,
)
from exasol.ai.text.impl.utils.char_span import CharSpan
from exasol.ai.text.impl.utils.constants import DEFAULT_SPACY_PIPELINE
from exasol.ai.text.impl.utils.spacy_utils import get_spacy_model


class SpacyTokenizer(Tokenizer):
    """
    Tokenizer algorithm that uses the Spacy Tokenizer to extract the tokens from text.
    See also, https://spacy.io/api/tokenizer.

    algorithm discriminator: spacy-tokenizer

    Parameters:
        model:
            The model name passed to the Spacy pipeline
    """

    algorithm_name: Literal["spacy-tokenizer"] = "spacy-tokenizer"
    model: str = DEFAULT_SPACY_PIPELINE

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self._nlp: Language | None = None

    def _lazy_init(self):
        if self._nlp is None:
            # We only need the tokenizer in the language pipeline.
            self._nlp = get_spacy_model(
                pipeline=self.model, exclude=["parser", "ner", "textcat"]
            )

    def tokenize_texts(self, texts: pd.Series) -> list[list[Token]]:
        self._lazy_init()
        if self._nlp is None:
            raise RuntimeError("Spacy Pipeline not initialized.")
        docs = self._nlp.pipe(texts)
        tokenized_texts = [self.spacy_doc_to_tokens(doc) for doc in docs]
        return tokenized_texts

    @staticmethod
    def spacy_doc_to_tokens(doc: Doc) -> list[Token]:
        tokens = [
            Token(
                span=CharSpan(
                    doc_id=-1, char_begin=token.idx, char_end=token.idx + len(token)
                ),
                content=TokenContent(
                    token=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    shape=token.shape_,
                    is_stop=token.is_stop,
                    is_alpha=token.is_alpha,
                    is_ascii=token.is_ascii,
                    is_digit=token.is_digit,
                    is_lower=token.is_lower,
                    is_upper=token.is_upper,
                    is_punct=token.is_punct,
                    is_left_punct=token.is_left_punct,
                    is_right_punct=token.is_right_punct,
                    is_quote=token.is_quote,
                    is_title=token.is_title,
                    is_bracket=token.is_bracket,
                    is_currency=token.is_currency,
                    like_num=token.like_num,
                    like_url=token.like_url,
                    like_email=token.like_email,
                ),
            )
            for token in doc
            if not token.is_space
        ]
        return tokens
