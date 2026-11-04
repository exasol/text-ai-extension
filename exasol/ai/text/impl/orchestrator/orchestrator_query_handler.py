from dataclasses import dataclass

from exasol.analytics.query_handler.context.scope import ScopeQueryHandlerContext
from exasol.analytics.query_handler.query.result.interface import QueryResult
from exasol.analytics.query_handler.query_handler import QueryHandler
from exasol.analytics.query_handler.result import (
    Continue,
    Finish,
)

from exasol.ai.text.extraction import AbstractExtraction
from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.impl.orchestrator.extraction_query_handler import (
    ExtractionQueryHandler,
)
from exasol.ai.text.impl.utils.query_handler_utils import ContinueOrFinish


@dataclass(frozen=True)
class Child:
    context: ScopeQueryHandlerContext
    query_handler: QueryHandler


class OrchestratorParameter(TextAIBase):
    extraction: AbstractExtraction
    language_alias: str


class OrchestratorOutput(TextAIBase):
    pass


class OrchestratorQueryHandler(QueryHandler[OrchestratorParameter, OrchestratorOutput]):
    """
    Orchestrate the top-level extraction process by controlling the child
    query handler performing the actual extraction.
    """

    def __init__(
        self,
        parameter: OrchestratorParameter,
        context: ScopeQueryHandlerContext,
    ):
        super().__init__(parameter, context)
        self._parameter = parameter

    def _create_extraction_job(self) -> Child:
        context = self._query_handler_context.get_child_query_handler_context()
        query_handler = ExtractionQueryHandler.create(
            extraction=self._parameter.extraction,
            context=context,
            language_alias=self._parameter.language_alias,
        )
        return Child(context, query_handler)

    def _handle_finish(self, action: ContinueOrFinish) -> ContinueOrFinish:
        self._child.context.release()
        return action

    def _handle_action(self, action: ContinueOrFinish) -> ContinueOrFinish:
        while True:
            if isinstance(action, Continue):
                return action
            elif isinstance(action, Finish):
                action = self._handle_finish(action)
                return action
            else:
                raise RuntimeError(f"Unknown action type {type(action)}")

    def start(self) -> ContinueOrFinish:
        self._child = self._create_extraction_job()
        action = self._child.query_handler.start()
        return self._handle_action(action)

    def handle_query_result(self, query_result: QueryResult) -> ContinueOrFinish:
        action = self._child.query_handler.handle_query_result(query_result)
        return self._handle_action(action)
