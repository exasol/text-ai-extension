from typing import (
    Literal,
)

from pydantic import Field

from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository


class HuggingFaceModel(TextAIBase):
    model_type: Literal["HuggingFaceModel"] = Field(
        default="HuggingFaceModel", repr=False
    )
    repository: BucketFSRepository | None = None
    name: str
