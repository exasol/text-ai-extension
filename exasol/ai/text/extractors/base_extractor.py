import re
from abc import ABC
from dataclasses import dataclass

from pydantic import Field

from exasol.ai.text.extractors.base_model import TextAIBase


@dataclass(frozen=True)
class NoSourceExtractorError(ValueError):
    message: str


def format_extractor_name(name: str) -> str:
    """
    Removes the word "extractor" from the end of name and replaces camel case with
    upper snake_case.
    """
    formatted_name = re.sub(r"[_\s]*extractor$", "", name, flags=re.IGNORECASE)
    pattern = r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
    return re.sub(pattern, "_", formatted_name).upper()


class BaseExtractor(TextAIBase, ABC):

    name: str = Field(default_factory=lambda self: self.__class__.__name__)

    def __init__(self, **kwargs):
        # Unless provided explicitly, derive the extractor name from its class name.
        if "name" not in kwargs:
            proposed_name = format_extractor_name(self.__class__.__name__)
            kwargs["name"] = proposed_name
        super().__init__(**kwargs)

    def validate_source(self) -> None:
        raise NoSourceExtractorError(f"a {type(self).__name__} ({self})")


class SourceExtractor(BaseExtractor, ABC):
    """
    Abstract Pydantic Type for source extractors.

    Source extractors are special as their corresponding StageGraph only uses
    the configuration of the Extractor as input but doesn't need any output of
    other Extractors as input.
    """

    def validate_source(self) -> None:
        pass


class OutputExtractor(BaseExtractor, ABC):
    """
    Abstract Pydantic Type for an extractor that creates an output table or a view.

    Parameters:
        db_output:
            The name of a table or a view the extractor should create and write data
            to (in case of a table). If not provided, the name should be derived from
            the name or type of the extractor. The exact details of that are TBD.
    """

    db_output: str | None = None


class NLPExtractor(OutputExtractor, ABC):
    """
    Abstract Pydantic Type for NLP extractors.

    Parameters:
        batch_size:
            Number of rows processed by the algorithm in one call. This parameter has
            an impact on UDF performance. The optimal number depends on the hardware
            the database node is running on.
            Default value is 100.
        parallelism_per_node:
            Number of UDF clients created at each node. This is another performance
            tuning parameter which optimal value depends on the hardware.
            Default value is 4.

    """

    batch_size: int | None = None
    parallelism_per_node: int | None = None
