from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import OutputExtractor


class TokenPositionExtractor(OutputExtractor):
    extractor_type: Literal["TokenPositionExtractor"] = Field(
        default="TokenPositionExtractor", repr=False
    )
