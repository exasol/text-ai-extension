from dataclasses import dataclass
from typing import cast

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors import (
    BranchExtractor,
    CoOccurrenceExtractor,
    FilterExtractor,
    HftNamedEntitySettings,
    HftTopicSettings,
    KeywordSearchExtractor,
    NamedEntityExtractor,
    PatternRankKeywordSettings,
    PipelineExtractor,
    SpacyTokenizerSettings,
    StandardExtractor,
    TokenizerExtractor,
    TokenPositionExtractor,
    TopicClassifierExtractor,
)
from exasol.ai.text.extractors.base_extractor import OutputExtractor
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder

# The names given to the Token Position extractors.
# They need to be different and provide meaningful prefixes for the column names
# in the CoOccurrence view.
TOKENIZED_ENTITY_NAME = "ENTITY"
TOKENIZED_KEYWORD_NAME = "KEYWORD"


@dataclass
class _NLPX:
    extractor: Extractor
    has_span: bool
    tokenized_name: str | None = None
    """Name of the TokenPosition extractor"""
    tokenized_db_output: str | None = None
    """Name of the view the TokenPosition extractor will create"""

    @property
    def is_tokenizer(self):
        return isinstance(self.extractor, TokenizerExtractor)


def _create_entity_extractor(std_extractor: StandardExtractor) -> _NLPX | None:
    if std_extractor.named_entity_recognition_model:
        extractor = NamedEntityExtractor(
            db_output=std_extractor.entity_db_output,
            named_entity_settings=HftNamedEntitySettings(
                model=std_extractor.named_entity_recognition_model
            ),
            parallelism_per_node=std_extractor.parallelism_per_node,
            batch_size=std_extractor.batch_size,
        )
        return _NLPX(
            extractor=extractor,
            has_span=True,
            tokenized_name=TOKENIZED_ENTITY_NAME,
            tokenized_db_output=std_extractor.entity_tokenized_db_output,
        )
    return None


def _create_topic_extractor(std_extractor: StandardExtractor) -> _NLPX | None:
    if std_extractor.topic_classification_model:
        if not std_extractor.topics:
            raise RuntimeError(
                "Topic Classification selected, but no topics are specified."
            )
        extractor = TopicClassifierExtractor(
            db_output=std_extractor.topic_db_output,
            topic_settings=HftTopicSettings(
                ranking_model=std_extractor.topic_classification_model,
                topics=std_extractor.topics,
            ),
            parallelism_per_node=std_extractor.parallelism_per_node,
            batch_size=std_extractor.batch_size,
        )
        return _NLPX(extractor=extractor, has_span=False)
    return None


def _create_keyword_extractor(std_extractor: StandardExtractor) -> _NLPX | None:
    if std_extractor.keyword_search_model:
        extractor = KeywordSearchExtractor(
            db_output=std_extractor.keyword_db_output,
            keyword_settings=PatternRankKeywordSettings(
                ranking_model=std_extractor.keyword_search_model
            ),
            parallelism_per_node=std_extractor.parallelism_per_node,
            batch_size=std_extractor.batch_size,
        )
        return _NLPX(
            extractor=extractor,
            has_span=True,
            tokenized_name=TOKENIZED_KEYWORD_NAME,
            tokenized_db_output=std_extractor.keyword_tokenized_db_output,
        )
    return None


def _create_tokenizer_extractor(std_extractor: StandardExtractor) -> _NLPX:
    extractor = TokenizerExtractor(
        db_output=std_extractor.tokenizer_db_output,
        tokenizer_settings=SpacyTokenizerSettings(),
        parallelism_per_node=std_extractor.parallelism_per_node,
        batch_size=std_extractor.batch_size,
    )
    # Technically, the tokenizer extractor does have a span, but for the purpose of this
    # workflow it only provides tokens for other extractors, hence has_span=False.
    return _NLPX(extractor=extractor, has_span=False)


def _create_nlp_extractors(std_extractor: StandardExtractor) -> list[_NLPX]:
    """
    Creates the required NLP extractors.
    Adds the TokenizerExtractor if one or more of the created extractors have an
    output span.
    """
    nlp_extractors = [
        nlp_extractor
        for nlp_extractor in [
            _create_entity_extractor(std_extractor),
            _create_topic_extractor(std_extractor),
            _create_keyword_extractor(std_extractor),
        ]
        if nlp_extractor is not None
    ]
    if any(x.has_span for x in nlp_extractors):
        nlp_extractors.append(_create_tokenizer_extractor(std_extractor))
    return nlp_extractors


def _get_tokenizer_name(nlp_extractors: list[_NLPX]) -> str:
    """
    Returns the name of the tokenizer extractor or raises an exception if
    the extractor has not been created.
    """
    for nlp_extractor in nlp_extractors:
        if nlp_extractor.is_tokenizer:
            return nlp_extractor.extractor.name
    else:
        raise RuntimeError("Tokenizer extractor not found.")


def _create_token_pipelines(nlp_extractors: list[_NLPX]) -> list[Extractor]:
    """
    Creates pipelines where token spans are attached to the outputs of the provided
    extractors. If the extractor does not create a span the pipeline is reduced to
    a filter.
    """
    pipelines: list[Extractor] = []
    for nlp_extractor in nlp_extractors:
        if nlp_extractor.has_span:
            pipelines.append(
                PipelineExtractor(
                    steps=[
                        FilterExtractor(
                            allowed_names={
                                nlp_extractor.extractor.name,
                                _get_tokenizer_name(nlp_extractors),
                            }
                        ),
                        TokenPositionExtractor(
                            name=nlp_extractor.tokenized_name,
                            db_output=nlp_extractor.tokenized_db_output,
                        ),
                    ]
                )
            )
        elif not nlp_extractor.is_tokenizer:
            pipelines.append(
                FilterExtractor(allowed_names={nlp_extractor.extractor.name})
            )
    return pipelines


def _set_db_output(std_extractor: StandardExtractor, pipeline: Extractor) -> Extractor:
    """
    Sets the `db_output` of the last extractor in the created pipeline to the
    `db_output` of the standard extractor if one is defined.
    """
    if std_extractor.db_output:
        parent: PipelineExtractor | None = None
        node = pipeline
        while isinstance(node, PipelineExtractor):
            parent = node
            node = parent.steps[-1]
        if isinstance(node, OutputExtractor):
            new_node = node.model_copy(update={"db_output": std_extractor.db_output})
            if parent is None:
                return new_node
            parent.steps[-1] = new_node
    return pipeline


class StandardExtractorGraphBuilder:
    """
    Builds an execution graph for a DefaultExtractor.

    Below is the schematic of the full graph. Note that some of the algorithms may not
    be included in the graph if their models are not defined. This will affect the
    structure of the graph, i.e. the pipeline may have fewer steps if the TokenPosition
    or Co-Occurrence views are not needed. Also, the Tokenizer is added only if some
    of the enabled algorithms have an output span.

    Pipeline(
        Branch(
            Entity(),
            Topic(),
            Keyword(),
            Tokenizer()
        ),
        Branch(
            Pipeline(
                Filter(["Entity", "Tokenizer"]),
                TokenPosition()
            ),
            Filter(["Topic"]),
            Pipeline(
                Filter(["Keyword", "Tokenizer"])
                TokenPosition()
            )
        ),
        CoOccurrence()
    )
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:
        if not isinstance(extractor, StandardExtractor):
            raise ValueError(f"Expected a StandardExtractor, got {extractor}")
        if downstream_builder is None:
            raise ValueError(
                "A valid downstream graph builder must be provided "
                f"in a call to {self.__class__.__name__}.build()."
            )

        std_extractor = cast(StandardExtractor, extractor)
        nlp_extractors = _create_nlp_extractors(std_extractor)

        num_extractors = len(nlp_extractors)
        if num_extractors == 0:
            raise RuntimeError("At least one model needs to be specified.")
        if num_extractors == 1:
            # Nothing else to do if there is only one extractor.
            pipeline = nlp_extractors[0].extractor
        else:
            branch_1_extractors = [
                nlp_extractor.extractor for nlp_extractor in nlp_extractors
            ]
            steps: list[Extractor] = [BranchExtractor(paths=branch_1_extractors)]

            # Add the token position computation step only if there are some spans.
            # At this step we may or may not create a Branch, depending on how many
            # paths we get. Note, that the number of paths can be bigger than the
            # number of extractors with spans.
            num_with_spans = sum(
                1 for extractor in nlp_extractors if extractor.has_span
            )
            if num_with_spans > 0:
                token_pipelines = _create_token_pipelines(nlp_extractors)
                if len(token_pipelines) == 1:
                    steps.append(token_pipelines[0])
                else:
                    steps.append(BranchExtractor(paths=token_pipelines))

            # Add the co-occurrence view in case there are more than one algorithm.
            if (num_extractors > 2) or (num_with_spans == 0):
                steps.append(CoOccurrenceExtractor())

            pipeline = PipelineExtractor(steps=steps)

        pipeline = _set_db_output(std_extractor, pipeline)
        sql_stage_graph = downstream_builder.build(
            pipeline, context, downstream_builder
        )
        return sql_stage_graph
