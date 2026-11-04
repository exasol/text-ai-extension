from typing import cast

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.tokenizer_extractor import TokenizerBase
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.algorithms.tokenizer.spacy_tokenizer import (
    SpacyTokenizer as SpacyTokenizerAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import (
    Tokenizer as TokenizerAlgorithm,
)
from exasol.ai.text.impl.nlp.algorithms.tokenizer.tokenizer import (
    TokenizerOutput,
)
from exasol.ai.text.impl.stage_graph.graph_builders import ExtractorContext
from exasol.ai.text.impl.stage_graph.graph_builders.nlp_graph_builder import (
    NLPGraphBuilder,
)


def algorithm_from_extractor(extractor: NLPExtractor) -> TokenizerAlgorithm:
    if isinstance(extractor, TokenizerBase):
        # At the moment there is only one tokenizer algorithm
        # in use. Once there is a choice of such algorithms this code should
        # switch on the discriminator of the `tokenizer_settings`.
        return SpacyTokenizerAlgorithm(outputs=TokenizerOutput(name=extractor.name))
    raise TypeError(
        f"The extractor of type {type(extractor)} does not "
        f"inherit from {TokenizerBase.__name__}."
    )


class TokenizerGraphBuilder(NLPGraphBuilder):
    """
    Builds a single-node graph for a tokenizer extractor.
    """

    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        return cast(AlgorithmModel, algorithm_from_extractor(extractor))
