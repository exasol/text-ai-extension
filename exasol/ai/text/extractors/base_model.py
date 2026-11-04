from abc import ABC

from pydantic import BaseModel


class TextAIBase(BaseModel, ABC):
    model_config = {"frozen": True}
