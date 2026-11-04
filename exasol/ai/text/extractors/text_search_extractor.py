from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.base_model import TextAIBase


class TokenizedTextSearchSettings(TextAIBase):
    """
    A text search extractor finding locations of given phrases in a text. The extractor
    uses the Spacy TokenMatcher algorithm.

    Parameters:
        setting_type:
            This is an earmark of the algorithm or a particular setup for the algorithm
            where the settings are applicable. This field allows switching between
            different algorithms doing a similar job.
        fuzziness:
            A number in the range [0..9] defining the fuzziness of the token matching.
            0 means the exact match is required. The fuzziness can help finding
            misspelled tokens and maybe also different inflections of the token.
            However, it should be used with caution. Values higher than 3 are not
            recommended.
            The fuzziness is only applicable for word tokens. All other types of tokens
            shall be matched precisely. This includes tokens with digits.
        ignore_spaces:
            If True (default) the space characters (e.g. a newline) between the tokens
            of the searched phrase are ignored.
    """

    setting_type: Literal["tokenized-text-search-settings"] = (
        "tokenized-text-search-settings"
    )
    fuzziness: int = 0
    ignore_spaces: bool = True


class TextSearchBase(TextAIBase):

    # In case of multiple arguments the `text_search_settings` should be a
    # discriminating union.
    text_search_settings: TokenizedTextSearchSettings = TokenizedTextSearchSettings()


class TextSearchExtractor(NLPExtractor, TextSearchBase):
    extractor_type: Literal["TextSearchExtractor"] = Field(
        default="TextSearchExtractor", repr=False
    )
