from typing import (
    Annotated,
    Literal,
)

from annotated_types import Len
from pydantic import Field

from exasol.ai.text.extractors.base_extractor import BaseExtractor


class FilterExtractor(BaseExtractor):

    extractor_type: Literal["FilterExtractor"] = Field(
        default="FilterExtractor", repr=False
    )

    allowed_names: Annotated[set[str], Len(min_length=1)]
    """
    Names of the inputs that will be passed to the outputs. The number of
    elements in the set determines the maximum number of outputs. The relative
    positions of the elements in the output should be the same as in the intput.
    """
