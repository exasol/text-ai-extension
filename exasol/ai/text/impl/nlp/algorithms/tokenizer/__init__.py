from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.tokenizer.spacy_tokenizer import SpacyTokenizer

# All tokenizer algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
TokenizerModel = Annotated[SpacyTokenizer, Field(discriminator="algorithm_name")]
