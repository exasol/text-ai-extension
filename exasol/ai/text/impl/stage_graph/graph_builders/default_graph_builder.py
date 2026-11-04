from exasol.ai.text.extractors import (
    BranchExtractor,
    CoOccurrenceExtractor,
    FilterExtractor,
    KeywordExtractor,
    KeywordSearchExtractor,
    NamedEntityExtractor,
    PipelineExtractor,
    SourceTableExtractor,
    SplitterExtractor,
    StandardExtractor,
    TextSearchExtractor,
    TokenizerExtractor,
    TokenPositionExtractor,
    TopicClassifierExtractor,
)
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.impl.stage_graph.graph_builders import (
    BranchGraphBuilder,
    CoOccurrenceGraphBuilder,
    DispatchingGraphBuilder,
    FilterGraphBuilder,
    GraphBuilder,
    KeywordGraphBuilder,
    KeywordSearchGraphBuilder,
    NamedEntityGraphBuilder,
    PipelineGraphBuilder,
    SourceTableGraphBuilder,
    SplitterGraphBuilder,
    TextSearchGraphBuilder,
    TokenizerGraphBuilder,
    TokenPositionGraphBuilder,
    TopicClassifierGraphBuilder,
)
from exasol.ai.text.impl.stage_graph.graph_builders.standard_extractor_graph_builder import (
    StandardExtractorGraphBuilder,
)


def default_graph_builder() -> DispatchingGraphBuilder:
    """
    Create a general purpose graph builder based on the
    :class:`DispatchingGraphBuilder`.

    The graph builder converts the user-defined extraction into a stage graph
    that can be executed by query handlers.

    The top-level extraction is defined by a hierarchy of extractors.

    TXAIE contains a dedicated extractor for each of the supported AI
    algorithms, e.g. NamedEntityExtractor, or KeywordExtractor.  With
    additional topological extractors such as BranchExtractor and
    PipelineExtractor other extractors can be combined and nested.

    For each of the extractors the corresponding graph builder is registered
    in order to make the DispatchingGraphBuilder support building the sub
    graph.
    """
    dispatcher = DispatchingGraphBuilder()
    pairs: list[tuple[type[Extractor], type[GraphBuilder]]] = [
        (BranchExtractor, BranchGraphBuilder),
        (KeywordExtractor, KeywordGraphBuilder),
        (KeywordSearchExtractor, KeywordSearchGraphBuilder),
        (NamedEntityExtractor, NamedEntityGraphBuilder),
        (PipelineExtractor, PipelineGraphBuilder),
        (TextSearchExtractor, TextSearchGraphBuilder),
        (TopicClassifierExtractor, TopicClassifierGraphBuilder),
        (TokenizerExtractor, TokenizerGraphBuilder),
        (SourceTableExtractor, SourceTableGraphBuilder),
        (TokenPositionExtractor, TokenPositionGraphBuilder),
        (CoOccurrenceExtractor, CoOccurrenceGraphBuilder),
        (FilterExtractor, FilterGraphBuilder),
        (SplitterExtractor, SplitterGraphBuilder),
        (StandardExtractor, StandardExtractorGraphBuilder),
    ]
    for extractor, builder in pairs:
        dispatcher.register(extractor, builder())
    return dispatcher
