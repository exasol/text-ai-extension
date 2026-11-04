from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import OutputExtractor


class CoOccurrenceExtractor(OutputExtractor):
    extractor_type: Literal["CoOccurrenceExtractor"] = Field(
        default="CoOccurrenceExtractor", repr=False
    )
