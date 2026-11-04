from unittest.mock import (
    call,
    create_autospec,
)

import pytest

from exasol.ai.text.extractors import (
    NamedEntityExtractor,
)
from exasol.ai.text.extractors.base_extractor import BaseExtractor
from exasol.ai.text.impl.stage_graph.graph_builders import (
    DispatchingGraphBuilder,
    GraphBuilder,
    NoGraphBuilderRegisteredError,
)
from test.utils.helpers import named_entity_extractor


@pytest.fixture
def sample_dispatcher() -> DispatchingGraphBuilder:
    dc = DispatchingGraphBuilder()
    dc.register(NamedEntityExtractor, create_autospec(GraphBuilder))
    return dc


def test_no_builder_registered(sample_dispatcher, extractor_context):
    class UnsupportedExtractor(BaseExtractor):
        pass

    with pytest.raises(NoGraphBuilderRegisteredError):
        sample_dispatcher.build(UnsupportedExtractor(), context=extractor_context)


def test_builder_called(extractor_context):
    extractor = named_entity_extractor()
    dummy_builder = create_autospec(GraphBuilder)
    downstream_builder = create_autospec(GraphBuilder)
    dispatcher = DispatchingGraphBuilder()
    dispatcher.register(type(extractor), dummy_builder)
    dispatcher.build(
        extractor=extractor,
        context=extractor_context,
        downstream_builder=downstream_builder,
    )
    assert dummy_builder.build.call_args_list == [
        call(extractor, extractor_context, downstream_builder)
    ]
