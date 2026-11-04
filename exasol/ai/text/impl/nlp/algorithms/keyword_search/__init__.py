from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.keyword_search.keyword_search import (
    KeywordSearch,
)

# All keyword search algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
KeywordSearchModel = Annotated[KeywordSearch, Field(discriminator="algorithm_name")]
