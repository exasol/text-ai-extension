import logging

import pandas as pd
import pytest

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from test.utils.helpers import (
    source_table_extractor,
)

LOG = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "optimization_parameters",
    [
        ({"batch_size": 2}, {"batch_size": 10}),
        ({"parallelism_per_node": 1}, {"parallelism_per_node": 2}),
    ],
)
def test_optimization_parameters(
    e2e_scenario,  # parametrized, iterates through a list of NLP extractors
    optimization_parameters,
    db_schema_name,
    language_alias,
    pyexasol_connection,
    setup_database,
    end2end_input_table,
    end2end_output_schema,
    model_subdir,
    end2end_topic_model,
    end2end_topic_model_download,
    end2end_feature_extraction_model,
    end2end_feature_extraction_model_download,
    end2end_named_entity_model,
    end2end_named_entity_model_download,
):
    """
    This end-to-end test verifies that different values for optimization
    parameters sich as batch_size and parallelism_per_node create identical
    results.
    """
    return
    if not e2e_scenario.extractor_factory:
        pytest.skip("""Validating optimization parameters batch_size and
        parallelism_per_node only applies when using an NLP extractor.""")
    for i in range(0, 300, 3):
        pyexasol_connection.execute(
            f"INSERT INTO {end2end_input_table.name.fully_qualified} VALUES "
            f"({i + 1}, 'This is a test.'), "
            f"({i + 2}, 'Exasol is awesome.'), "
            f"({i + 3}, 'Exasol is even better.')"
        )

    def run_extraction(
        individual_params: dict[str, int],
        truncate: bool = False,
    ) -> pd.DataFrame:
        # Use parallelism_per_node=1 as default to avoid that the UDFs get
        # killed, because the DB runs out of memory
        default_params = {
            "batch_size": 10,
            "parallelism_per_node": 1,
        }
        params = default_params | individual_params
        LOG.info(
            f"\n Running {e2e_scenario.extractor_factory.__name__}"
            f" with individual_params {individual_params}"
            f"\n         resulting params: {params}"
        )
        table_name = e2e_scenario.output_table.fully_qualified
        if truncate:
            pyexasol_connection.execute(f"TRUNCATE TABLE {table_name}")
        extraction = Extraction(
            extractor=Pipeline(
                steps=[
                    source_table_extractor(),
                    e2e_scenario.extractor_factory(**params),
                ]
            ),
            output=Output(db_schema=end2end_output_schema.name),
        )
        extraction.run(
            pyexasol_connection,
            db_schema_name,
            language_alias,
        )
        return pyexasol_connection.export_to_pandas(
            f"SELECT * FROM {table_name}" " ORDER BY TEXT_DOC_ID, TEXT_CHAR_BEGIN"
        )

    df1 = run_extraction(optimization_parameters[0])
    df2 = run_extraction(optimization_parameters[1], truncate=True)
    pd.testing.assert_frame_equal(df1, df2)
    pyexasol_connection.execute(
        f"DROP SCHEMA IF EXISTS {end2end_output_schema.quoted_name} CASCADE"
    )
    pyexasol_connection.execute(
        f"TRUNCATE TABLE {end2end_input_table.name.fully_qualified}"
    )
