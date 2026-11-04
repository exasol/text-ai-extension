from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.text_search.text_search import (
    TextSearch as TextSearch,
)
from exasol.ai.text.impl.nlp.algorithms.text_search.tokenized_text_search import (
    TokenizedTextSearch,
)

# All text search algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
TextSearchModel = Annotated[TokenizedTextSearch, Field(discriminator="algorithm_name")]
