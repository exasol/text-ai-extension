from typing import Annotated

from pydantic import Field

from exasol.ai.text.impl.nlp.algorithms.topics.hft_topic_classification import (
    HftTopicClassification,
)
from exasol.ai.text.impl.nlp.algorithms.topics.topic_classification import (
    TopicClassification as TopicClassification,
)

# All topic classification algorithms should be listed here in a union.
# For details on using the Annotated type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
TopicClassificationModel = Annotated[
    HftTopicClassification, Field(discriminator="algorithm_name")
]
