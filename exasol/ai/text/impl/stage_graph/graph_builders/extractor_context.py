from dataclasses import dataclass

from exasol.ai.text.extraction.abstract_extraction import (
    Defaults,
    Output,
)


@dataclass
class ExtractorContext:
    """
    A collection of "static" parameters to be injected to the extractor graph_builders.
    They are static in a sense that they are know at the time of graph creation and
    do not change during the graph execution.

    language_alias:
        The UDF language alias.
    schema:
        The database schema.
    db_prefix:
        An optional prefix to bed used when making a name for output tables. Setting
        the prefix can prevent name collision with existing tables.
    """

    language_alias: str
    output: Output
    defaults: Defaults
