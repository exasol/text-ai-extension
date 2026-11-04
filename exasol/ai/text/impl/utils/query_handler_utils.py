from typing import Union

from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)

ContinueOrFinish = Union[Continue, Finish]
