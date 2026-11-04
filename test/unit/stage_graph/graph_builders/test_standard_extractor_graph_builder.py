from typing import cast
from unittest.mock import (
    create_autospec,
)

import pytest

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
from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.stage_graph.graph_builders import (
    GraphBuilder,
)
from exasol.ai.text.impl.stage_graph.graph_builders.standard_extractor_graph_builder import (
    TOKENIZED_ENTITY_NAME,
    TOKENIZED_KEYWORD_NAME,
    StandardExtractorGraphBuilder,
)
from test.unit.sql_stage_mock import sql_stage_graph

_topics = ["test", "abc"]

_test_ner_model = HuggingFaceModel(name="test_ner")
_test_topics_model = HuggingFaceModel(name="test_topics")
_test_keyword_model = HuggingFaceModel(name="test_keyword")

ENTITY_OUTPUT = "NER"
TOPIC_OUTPUT = "TOPIC"
KEYWORD_OUTPUT = "KWD"
TOKENIZER_OUTPUT = "TOKENIZER"
ENTITY_TOK_OUTPUT = "TOK_NER"
KEYWORD_TOK_OUTPUT = "TOK_KWD"
FINAL_OUTPUT = "MY_STDX"


@pytest.fixture
def mock_stage_graph():
    return sql_stage_graph(1, 2)


@pytest.fixture
def downstream_builder(mock_stage_graph):
    builder = create_autospec(GraphBuilder)
    builder.build.side_effect = [mock_stage_graph]
    return builder


def define_extractor(**kwargs) -> StandardExtractor:
    """
    Builds the definition of the standard extractor.
    """
    return StandardExtractor(
        **kwargs,
        entity_db_output=ENTITY_OUTPUT,
        topic_db_output=TOPIC_OUTPUT,
        keyword_db_output=KEYWORD_OUTPUT,
        tokenizer_db_output=TOKENIZER_OUTPUT,
        entity_tokenized_db_output=ENTITY_TOK_OUTPUT,
        keyword_tokenized_db_output=KEYWORD_TOK_OUTPUT,
        db_output=FINAL_OUTPUT,
    )


def build_extractor(
    input_extractor: StandardExtractor, downstream_builder, extractor_context
) -> Extractor:
    """
    Builds the implementation of the standard extractor, by calling the graph builder for
    the provided standard extractor definition.
    """
    StandardExtractorGraphBuilder().build(
        input_extractor, extractor_context, downstream_builder
    )
    return downstream_builder.build.call_args_list[0].args[0]


def test_downstream_builder_call(
    mock_stage_graph, downstream_builder, extractor_context
):
    """
    This test validates that the downstream builder is called with expected parameters
    and returns the expected result.
    """
    input_extractor = StandardExtractor(topics=_topics)
    actual_graph = StandardExtractorGraphBuilder().build(
        input_extractor, extractor_context, downstream_builder
    )
    assert actual_graph == mock_stage_graph
    assert len(downstream_builder.build.call_args_list) == 1
    _, actual_context, actual_downstream_builder = (
        downstream_builder.build.call_args_list[0].args
    )
    assert actual_context == extractor_context
    assert actual_downstream_builder == downstream_builder


def test_standard_graph_structure(downstream_builder, extractor_context):
    """
    This test validates the structure of the created standard extractor graph
    in the case when all models are present (the full standard extractor).
    """
    input_extractor = StandardExtractor(topics=_topics)
    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    assert isinstance(actual_extractor, PipelineExtractor)
    assert len(actual_extractor.steps) == 3
    for step, expected_type in zip(
        actual_extractor.steps,
        [BranchExtractor, BranchExtractor, CoOccurrenceExtractor],
    ):
        assert isinstance(step, expected_type)
    for extractor in actual_extractor.steps[0].paths:
        assert isinstance(extractor, NLPExtractor)
    for extractor in actual_extractor.steps[1].paths:
        assert isinstance(extractor, PipelineExtractor) or isinstance(
            extractor, FilterExtractor
        )
        if isinstance(extractor, PipelineExtractor):
            assert len(extractor.steps) == 2
            assert isinstance(extractor.steps[0], FilterExtractor)
            assert isinstance(extractor.steps[1], TokenPositionExtractor)


@pytest.mark.parametrize(
    "input_extractor, expected_outputs",
    [
        (
            define_extractor(topics=_topics),
            {ENTITY_OUTPUT, TOPIC_OUTPUT, KEYWORD_OUTPUT, TOKENIZER_OUTPUT},
        ),
        (
            define_extractor(topics=_topics, named_entity_recognition_model=None),
            {TOPIC_OUTPUT, KEYWORD_OUTPUT, TOKENIZER_OUTPUT},
        ),
        (
            define_extractor(topic_classification_model=None),
            {ENTITY_OUTPUT, KEYWORD_OUTPUT, TOKENIZER_OUTPUT},
        ),
        (
            define_extractor(topics=_topics, keyword_search_model=None),
            {ENTITY_OUTPUT, TOPIC_OUTPUT, TOKENIZER_OUTPUT},
        ),
    ],
    ids=["full", "no entity", "no topic", "no keyword"],
)
def test_algorithm_options(
    input_extractor, expected_outputs, downstream_builder, extractor_context
):
    """
    This test validates that by setting the model of a certain algorithm to None we
    exclude this algorithm from the standard graph.
    All scenarios have more than one algorithm, therefore the graph should still be
    a Pipeline with three steps, ending with the Co-Occurrence view.
    """
    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    assert isinstance(actual_extractor, PipelineExtractor)
    steps = actual_extractor.steps
    assert len(steps) == 3
    assert isinstance(steps[0], BranchExtractor)
    branch_1 = steps[0]
    actual_outputs = {extractor.db_output for extractor in branch_1.paths}
    assert actual_outputs == expected_outputs
    assert isinstance(steps[1], BranchExtractor)
    assert isinstance(steps[2], CoOccurrenceExtractor)
    assert steps[2].db_output == input_extractor.db_output


def test_trivial_case(downstream_builder, extractor_context):
    """
    Validates that when only the Topic Classification is left, while other NLP
    algorithms disabled, the graph is reduced to the Topic extractor. This is
    because the Topic extractor doesn't produce any span and therefore doesn't
    require a Tokenizer.
    """
    input_extractor = define_extractor(
        topics=_topics, named_entity_recognition_model=None, keyword_search_model=None
    )
    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    assert isinstance(actual_extractor, TopicClassifierExtractor)
    assert actual_extractor.db_output == input_extractor.db_output


def test_single_algo_with_span(downstream_builder, extractor_context):
    """
    Validates the case with only one NLP algorithm with a span, e.g. NamedEntity.
    In such a case the pipeline should have two steps - we need to attach the token
    span, but don't need the Co-Occurrence view, since it's just one algorithm.
    """
    input_extractor = define_extractor(
        topic_classification_model=None, keyword_search_model=None
    )
    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    assert isinstance(actual_extractor, PipelineExtractor)
    steps = actual_extractor.steps
    assert len(steps) == 2
    assert isinstance(steps[1], PipelineExtractor)
    steps_2 = steps[1].steps
    assert isinstance(steps_2[0], FilterExtractor)
    assert isinstance(steps_2[1], TokenPositionExtractor)
    assert steps_2[1].db_output == input_extractor.db_output


def test_model_settings(downstream_builder, extractor_context):
    """
    This test validates that the algorithm settings in the standard extractor are
    passed to the created algorithm extractors.
    """
    input_extractor = define_extractor(
        topics=_topics,
        named_entity_recognition_model=_test_ner_model,
        topic_classification_model=_test_topics_model,
        keyword_search_model=_test_keyword_model,
        parallelism_per_node=10,
        batch_size=200,
    )
    expected_content = {
        ENTITY_OUTPUT: NamedEntityExtractor(
            db_output=ENTITY_OUTPUT,
            named_entity_settings=HftNamedEntitySettings(model=_test_ner_model),
            parallelism_per_node=10,
            batch_size=200,
        ),
        TOPIC_OUTPUT: TopicClassifierExtractor(
            db_output=TOPIC_OUTPUT,
            topic_settings=HftTopicSettings(
                ranking_model=_test_topics_model, topics=_topics
            ),
            parallelism_per_node=10,
            batch_size=200,
        ),
        KEYWORD_OUTPUT: KeywordSearchExtractor(
            db_output=KEYWORD_OUTPUT,
            keyword_settings=PatternRankKeywordSettings(
                ranking_model=_test_keyword_model
            ),
            parallelism_per_node=10,
            batch_size=200,
        ),
        TOKENIZER_OUTPUT: TokenizerExtractor(
            db_output=TOKENIZER_OUTPUT,
            tokenizer_settings=SpacyTokenizerSettings(),
            parallelism_per_node=10,
            batch_size=200,
        ),
    }

    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    actual_branch_1 = cast(
        BranchExtractor, cast(PipelineExtractor, actual_extractor).steps[0]
    )
    actual_content = {
        extractor.db_output: extractor for extractor in actual_branch_1.paths
    }
    assert actual_content == expected_content


def test_token_position_outputs(downstream_builder, extractor_context):
    """
    The test validates setting the correct extractor names and DB output names for the
    token position views.
    """
    input_extractor = define_extractor(topic_classification_model=None)
    actual_extractor = build_extractor(
        input_extractor, downstream_builder, extractor_context
    )
    assert isinstance(actual_extractor, PipelineExtractor)
    steps = actual_extractor.steps
    assert len(steps) > 2
    step_2 = steps[1]
    assert isinstance(step_2, BranchExtractor)
    tokenized_names: set[str] = set()
    tokenized_db_outputs: set[str] = set()
    for branch_path in step_2.paths:
        assert isinstance(branch_path, PipelineExtractor)
        token_position_extractor = branch_path.steps[-1]
        assert isinstance(token_position_extractor, TokenPositionExtractor)
        tokenized_names.add(token_position_extractor.name)
        tokenized_db_outputs.add(token_position_extractor.db_output)
    assert tokenized_names == {TOKENIZED_ENTITY_NAME, TOKENIZED_KEYWORD_NAME}
    assert tokenized_db_outputs == {ENTITY_TOK_OUTPUT, KEYWORD_TOK_OUTPUT}


def test_no_topics_fails(downstream_builder, extractor_context):
    """
    Validates the error in case the extractor defines no topics but the topic
    classification model is not set to None.
    """
    input_extractor = StandardExtractor()
    with pytest.raises(
        RuntimeError,
        match="Topic Classification selected, but no topics are specified.",
    ):
        StandardExtractorGraphBuilder().build(
            input_extractor, extractor_context, downstream_builder
        )


def test_no_models_fails(downstream_builder, extractor_context):
    """
    Validates the error in case all models are set to None.
    """
    input_extractor = StandardExtractor(
        keyword_search_model=None,
        topic_classification_model=None,
        named_entity_recognition_model=None,
    )
    with pytest.raises(RuntimeError, match="At least one model needs to be specified."):
        StandardExtractorGraphBuilder().build(
            input_extractor, extractor_context, downstream_builder
        )
