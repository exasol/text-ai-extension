from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.named_entity.hft_named_entity_recognition import (
    HftNamedEntityRecognition,
)
from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityRecognition as NamedEntityRecognition,
)

# All named entity recognition algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
NamedEntityRecognitionModel = Annotated[
    HftNamedEntityRecognition, Field(discriminator="algorithm_name")
]
