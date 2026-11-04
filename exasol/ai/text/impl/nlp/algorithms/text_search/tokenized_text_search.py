from collections.abc import Iterable
from typing import (
    Any,
    Literal,
    cast,
)

from pydantic import Field
from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Token

from exasol.ai.text.impl.nlp.algorithms.text_search.text_search import (
    PhraseLocations,
    TextSearch,
)
from exasol.ai.text.impl.utils.spacy_utils import get_spacy_model


class TokenizedTextSearch(TextSearch):
    """
    A text search algorith using the Spacy TokenMatcher.

    algorithm discriminator: token-match

    Parameters:
        fuzziness:
            A number in the range [0..9] defining the fuzziness of the token matching.
            0 means the exact match is required. The fuzziness can help finding
            misspelled tokens and maybe also different inflections of the token.
            However, it should be used with caution. Values higher than 3 are not
            recommended.
            The fuzziness is only applicable for word tokens. All other types of tokens
            will be matched precisely. This includes tokens with digits.
        ignore_spaces:
            If True (default) the space characters (e.g. a newline) between the tokens
            of the searched phrase are ignored.
    """

    algorithm_name: Literal["token-match"] = Field("token-match", repr=False)
    fuzziness: int = 0
    ignore_spaces: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nlp: Language | None = None

    def _lazy_init(self):
        assert 0 <= self.fuzziness <= 9, (
            "The token match fuzziness " "must be within the range [0..9]"
        )
        if self._nlp is None:
            # We only need the tokenizer in the language pipeline.
            self._nlp = get_spacy_model(
                exclude=["tagger", "parser", "ner", "lemmatizer", "textcat"]
            )

    def _token_rule(self, token: Token) -> dict[str, Any]:
        """
        Creates a rule for matching a single token.
        """
        token_expr: str | dict[str, Any] = token.text.lower()
        if self.fuzziness and token.is_alpha:
            token_expr = {f"FUZZY{self.fuzziness}": token_expr}
        return {"LOWER": token_expr}

    def locate_phrases(self, text: str, phrases: Iterable[str]) -> PhraseLocations:

        self._lazy_init()
        language = cast(Language, self._nlp)
        matcher = Matcher(language.vocab)

        phrase_list = list(phrases)
        for phrase_seq, phrase in enumerate(phrase_list):
            keyword_pattern: list[dict[str, Any]] = []
            for token in language(phrase):
                # If spaces are to be ignored the token matching rules must be
                # interspersed with the rules matching zero or more space tokens.
                if self.ignore_spaces:
                    if token.is_space:
                        continue
                    if keyword_pattern:
                        keyword_pattern.append({"IS_SPACE": True, "OP": "*"})
                keyword_pattern.append(self._token_rule(token))
            if not keyword_pattern:
                raise ValueError(
                    "The Tokenized Text Searcher cannot search for an "
                    f"empty string. The phrase number {phrase_seq} "
                    "appears to be empty."
                )
            matcher.add(phrase_seq, [keyword_pattern])

        doc = language(text)
        matches = matcher(doc)

        rows = []
        for phrase_seq, token_begin, token_end in matches:
            phrase_match = doc[token_begin:token_end]
            rows.append((phrase_seq, phrase_match.start_char, phrase_match.end_char))
        return rows
