from typing import (
    Annotated,
    Literal,
)

from annotated_types import Len
from pydantic import Field

from exasol.ai.text.extractors.base_extractor import BaseExtractor


class SplitterExtractor(BaseExtractor):

    extractor_type: Literal["SplitterExtractor"] = Field(
        default="SplitterExtractor", repr=False
    )

    new_names: Annotated[set[str], Len(min_length=1)]
    """
    Names to be given to the outputs. The number of elements in the set determines the
    number of outputs. The order of outputs is undetermined. 
    """
