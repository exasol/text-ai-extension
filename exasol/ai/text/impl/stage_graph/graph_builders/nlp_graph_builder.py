from abc import (
    ABC,
    abstractmethod,
)

from exasol.analytics.query_handler.graph.stage.sql.sql_stage_graph import SQLStageGraph

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.extractor import Extractor
from exasol.ai.text.extractors.hf_model import HuggingFaceModel
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel
from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.stage_graph.graph_builders.extractor_context import (
    ExtractorContext,
)
from exasol.ai.text.impl.stage_graph.graph_builders.graph_builder import GraphBuilder
from exasol.ai.text.impl.stage_graph.graph_builders.utils import (
    create_log_span_for_extractor,
)
from exasol.ai.text.impl.stage_graph.query_handlers.udf_algo_sql_stage import (
    UDFAlgoSQLStage,
)
from exasol.ai.text.impl.utils.stage_graph_utils import get_output_table


def get_parallelism_per_node(context: ExtractorContext, extractor: NLPExtractor):
    return extractor.parallelism_per_node or context.defaults.parallelism_per_node


def get_batch_size(context: ExtractorContext, extractor: NLPExtractor):
    return extractor.batch_size or context.defaults.batch_size


def get_bucketfs_connection_name(context: ExtractorContext, model: HuggingFaceModel):
    default_model_repository = context.defaults.model_repository
    default_connection_name = (
        default_model_repository.connection_name if default_model_repository else None
    )
    model_connection_name = (
        model.repository.connection_name if model.repository else None
    )
    connection_name = model_connection_name or default_connection_name
    if connection_name:
        return connection_name
    else:
        raise ValueError(
            f"No BucketFSConnection specified in the default model repository {default_model_repository} "
            f"or in the model {model}"
        )


def get_bucketfs_subdir(context: ExtractorContext, model: HuggingFaceModel):
    default_model_repository = context.defaults.model_repository
    default_subdir = (
        default_model_repository.sub_dir if default_model_repository else None
    )
    model_subdir = model.repository.sub_dir if model.repository else None
    subdir = model_subdir or default_subdir
    if subdir:
        return subdir
    else:
        raise ValueError(
            f"No subdir specified in the default model repository {default_model_repository} "
            f"or in the model {model}"
        )


class NLPGraphBuilder(ABC):
    """
    A base class for an NLP algorithm graph builder.
    """

    def build(
        self,
        extractor: Extractor,
        context: ExtractorContext,
        downstream_builder: GraphBuilder | None = None,
    ) -> SQLStageGraph:
        if not isinstance(extractor, NLPExtractor):
            raise ValueError(
                "The type of extractor expected by an NLP Graph Builder "
                f"should be a subclass of {NLPExtractor.__name__}. "
                f"Got {type(extractor).__name__}."
            )

        algorithm = self.build_algorithm(extractor, context)
        log_span = create_log_span_for_extractor(extractor)
        stage = UDFAlgoSQLStage(
            udf_runner=UDFRunner(
                algorithm=algorithm,
                batch_size=get_parallelism_per_node(context, extractor),
            ),
            output_table=get_output_table(extractor, context),
            language_alias=context.language_alias,
            parallelism_per_node=get_parallelism_per_node(context, extractor),
            log_span=log_span,
        )
        return SQLStageGraph(stage, stage, set())

    @abstractmethod
    def build_algorithm(
        self, extractor: NLPExtractor, context: ExtractorContext
    ) -> AlgorithmModel:
        """
        Converts the extractor configuration into the corresponding algorithm
        configuration.
        """
