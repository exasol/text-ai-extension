from dataclasses import dataclass
from typing import Annotated

from pydantic import AfterValidator

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
)
from exasol.ai.text.impl.nlp.algorithms.udf_text_base import (
    TextOutput,
    UDFTextBase,
)
from exasol.ai.text.impl.utils.sql_utils import (
    double_column_spec,
    integer_column_spec,
    varchar_column_spec,
)

_default_topic = varchar_column_spec("TOPIC", flags=IOColumnFlags.Normalisation)


def topics_are_unique(topics: list[str]) -> list[str]:
    if len(topics) != len(set(topics)):
        raise ValueError("All topics in the list must be unique")
    return topics


@dataclass(frozen=True)
class TopicClassifierOutput(TextOutput):
    name: str = "TOPIC"

    """
    The output columns of a topic classification algorithm.
    """
    topic: ColumnSpec = _default_topic
    """
    The name of the topic, or NULL in case of an error.
    """

    topic_score: ColumnSpec = double_column_spec(f"{_default_topic.name}_SCORE")
    """
    The topic score, or NULL in case of an error.
    """

    topic_rank: ColumnSpec = integer_column_spec(f"{_default_topic.name}_RANK")
    """
    The topic rank, or NULL in case of an error. The topics are ranked in the
    ascending order, 1 to n-topics, where 1 is the rank of the topic with the
    highest score.
    """


# noinspection PyAbstractClass
class TopicClassification(UDFTextBase):
    """
    The base class for topic classification algorithms.

    Parameters:
        topics:
            List of topics to look for.
    """

    outputs: TopicClassifierOutput = TopicClassifierOutput()
    topics: Annotated[list[str], AfterValidator(topics_are_unique)]

    @property
    def topic_column(self) -> str:
        return self.outputs.topic.name

    @property
    def score_column(self) -> str:
        return self.outputs.topic_score.name

    @property
    def rank_column(self) -> str:
        return self.outputs.topic_rank.name
