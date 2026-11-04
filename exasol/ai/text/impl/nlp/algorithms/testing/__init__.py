from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.testing.batch_size_logger import BatchSizeLogger
from exasol.ai.text.impl.nlp.algorithms.testing.dummy_word_counter import (
    DummyWordCounter,
)
from exasol.ai.text.impl.nlp.algorithms.testing.span_size_categorizer import (
    SpanSizeCategorizer,
)
from exasol.ai.text.impl.nlp.algorithms.testing.span_size_logger import SpanSizeLogger

# All testing algorithms should be listed here.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
TestingModel = Annotated[
    (DummyWordCounter | BatchSizeLogger | SpanSizeLogger | SpanSizeCategorizer),
    Field(discriminator="algorithm_name"),
]
