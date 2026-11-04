from typing import Union

from exasol.analytics.query_handler.graph.stage.sql.input_output import (
    SQLStageInputOutput,
)
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)

ContinueOrFinish = Union[Continue, Finish[SQLStageInputOutput]]


class IllegalMethodCallError(RuntimeError):
    """
    The application framework logic called a method that was not intended
    to be called, e.g. ``IdentityQueryHandler.handle_query_result()``.
    """
