"""
Defines the Extractor Discriminator Union Type for all supported Extractors.

For successful validation with pydantic and mypy, all types with a circular
dependency must be defined in the same file. If Extractor has a forward
reference to other Extractors, then Extractor Annotated does not work.
"""

import typing
from typing import Annotated

from annotated_types import Len
from pydantic import Field

from exasol.ai.text.extractors.base_extractor import (
    BaseExtractor,
    NoSourceExtractorError,
)
from exasol.ai.text.extractors.branch_extractor_helper import ordinal
from exasol.ai.text.extractors.co_occurrence_extractor import CoOccurrenceExtractor
from exasol.ai.text.extractors.filter_extractor import FilterExtractor
from exasol.ai.text.extractors.keyword_extractor import (
    KeywordExtractor,
)
from exasol.ai.text.extractors.keyword_search_extractor import KeywordSearchExtractor
from exasol.ai.text.extractors.named_entity_extractor import (
    NamedEntityExtractor,
)
from exasol.ai.text.extractors.source_table_extractor import SourceTableExtractor
from exasol.ai.text.extractors.splitter_extractor import SplitterExtractor
from exasol.ai.text.extractors.standard_extractor import StandardExtractor
from exasol.ai.text.extractors.text_search_extractor import (
    TextSearchExtractor,
)
from exasol.ai.text.extractors.token_position_extractor import TokenPositionExtractor
from exasol.ai.text.extractors.tokenizer_extractor import TokenizerExtractor
from exasol.ai.text.extractors.topic_classifier_extractor import (
    TopicClassifierExtractor,
)


class BranchExtractor(BaseExtractor):
    """
    This extractor represents a branch of 1 or multiple sub-extractors,
    called paths.

    The BranchExtractor feeds its input into each of the paths and its output
    is a list of the outputs of all the paths.
    """

    extractor_type: typing.Literal["BranchExtractor"] = Field(
        default="BranchExtractor", repr=False
    )
    paths: Annotated[list["Extractor"], Len(min_length=1)]

    def validate_source(self) -> None:
        for i, p in enumerate(self.paths):
            try:
                p.validate_source()
            except NoSourceExtractorError as ex:
                raise NoSourceExtractorError(
                    f"a BranchExtractor that is no Source Extractor, because"
                    f"\n  - its {ordinal(i + 1)} path is {ex.message}"
                )


class PipelineExtractor(BaseExtractor):
    extractor_type: typing.Literal["PipelineExtractor"] = Field(
        default="PipelineExtractor", repr=False
    )
    steps: Annotated[list["Extractor"], Len(min_length=1)]

    def validate_source(self) -> None:
        try:
            self.steps[0].validate_source()
        except NoSourceExtractorError as ex:
            raise NoSourceExtractorError(
                f"a PipelineExtractor that is no Source Extractor, because"
                f"\n  - its first step is {ex.message}"
            )


# All extractors should be listed here.  For details on using the Annotated
# type hint with pydantic discriminated union see
# https://docs.pydantic.dev/latest/concepts/fields/#using-annotated
Extractor = Annotated[
    (
        KeywordExtractor
        | TextSearchExtractor
        | KeywordSearchExtractor
        | NamedEntityExtractor
        | TopicClassifierExtractor
        | TokenizerExtractor
        | SourceTableExtractor
        | StandardExtractor
        | PipelineExtractor
        | BranchExtractor
        | TokenPositionExtractor
        | CoOccurrenceExtractor
        | FilterExtractor
        | SplitterExtractor
    ),
    Field(discriminator="extractor_type"),
]

# This is needed, because PipelineExtractor and BranchExtractor use a forward reference to the Extractor
PipelineExtractor.model_rebuild()
BranchExtractor.model_rebuild()
