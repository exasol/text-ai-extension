from typing import Annotated

from pydantic import Field

from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository

AIModelRepository = Annotated[
    (BucketFSRepository), Field(discriminator="repository_type")
]
