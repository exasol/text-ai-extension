from typing import Annotated

from pydantic import Field

from exasol.ai.text.extractors.hf_model import HuggingFaceModel

AIModel = Annotated[(HuggingFaceModel), Field(discriminator="model_type")]
