from dataclasses import dataclass

from exasol.analytics.schema import TableLikeName

from exasol.ai.text.impl.nlp.algorithms.udf_text_base import TextInput
from exasol.ai.text.impl.stage_graph.input_output.algo_input_output import AlgoIOData


@dataclass(frozen=True)
class SourceTableOutput(AlgoIOData):
    """
    The output of a Source Table Query Handler.

    The main goal of the Source Table Extractor is collecting texts from various
    database sources into the single Document table. Hence, the query handler output
    inherits from the `TextInput` - the input class representing the Document table.

    Additionally, the query handler provides the names of the mapping tables that
    allow tracing the texts in the Document table back to the source tables where they
    were taken from.
    """

    columns: TextInput
    source_mapping: dict[TableLikeName, TableLikeName]
