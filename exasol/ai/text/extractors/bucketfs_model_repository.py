from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_model import TextAIBase


class BucketFSRepository(TextAIBase):
    repository_type: Literal["BucketFSRepository"] = Field(
        default="BucketFSRepository", repr=False
    )
    connection_name: str | None = None
    sub_dir: str | None = None
