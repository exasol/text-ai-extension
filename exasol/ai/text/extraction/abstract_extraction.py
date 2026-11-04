from typing import Annotated

from exasol.analytics.schema import (
    SchemaName,
    TableNameImpl,
)
from pydantic import AfterValidator

from exasol.ai.text.extractors.base_extractor import NoSourceExtractorError
from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.extractors.bucketfs_model_repository import BucketFSRepository
from exasol.ai.text.extractors.extractor import Extractor

MAX_PREFIX_LENGTH = 50


def is_db_schema_valid(name: str) -> str:
    SchemaName(name)
    return name


def is_table_prefix_valid(prefix: str) -> str:
    TableNameImpl(prefix)
    if len(prefix) > MAX_PREFIX_LENGTH:
        raise ValueError(
            f'{TableNameImpl.__name__} name "{prefix}" is too long: '
            f"{len(prefix)} > max {MAX_PREFIX_LENGTH}"
        )
    return prefix


def validate_source(extractor: Extractor) -> Extractor:
    try:
        extractor.validate_source()
        return extractor
    except NoSourceExtractorError as ex:
        raise ValueError(
            f"the specified extraction is invalid, because no Source Extractor was found:"
            f"\n  - its top-level is {ex.message}\n"
        ) from ex


class Output(TextAIBase):
    db_schema: Annotated[str, AfterValidator(is_db_schema_valid)]
    table_prefix: Annotated[str, AfterValidator(is_table_prefix_valid)] | None = None


class Defaults(TextAIBase):
    model_repository: BucketFSRepository | None = None
    batch_size: int = 100
    parallelism_per_node: int = 4


class AbstractExtraction(TextAIBase):
    extractor: Annotated[Extractor, AfterValidator(validate_source)]
    output: Output
    defaults: Defaults = Defaults()
