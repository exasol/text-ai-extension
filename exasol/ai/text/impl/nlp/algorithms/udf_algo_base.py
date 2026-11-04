import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from exasol.ai.text.impl.db_schema import (
    ColumnSpec,
    IOColumnFlags,
    OrderedColumns,
)
from exasol.ai.text.impl.utils.sql_utils import varchar_column_spec


@dataclass(frozen=True)
class AlgoInput(OrderedColumns):
    """
    Base class for the definition of algorithm input columns.
    """

    name: str = "ALGO"


@dataclass(frozen=True)
class AlgoOutput(OrderedColumns):
    """
    Base class for the definition of algorithm output columns.
    """

    name: str = "ALGO"

    error_message: ColumnSpec = varchar_column_spec("ERROR_MESSAGE")
    """
    An error message if the operation has failed, otherwise NULL.
    """

    setup: ColumnSpec = varchar_column_spec("SETUP", flags=IOColumnFlags.Normalisation)
    """
    Algorithm setup description. This information will be included in the UDF output
    in order to identify the conditions under which the result was computed. This would
    normally be a json string of the algorithm configuration.
    """

    @property
    def order(self) -> list[ColumnSpec]:
        return [self.error_message, self.setup]


class UDFAlgoBase(BaseModel):
    """
    The base class for a UDF enabled algorithm.

    IMPORTANT NOTE: The algorithm should be a pydantic model. As such, it will be a part
    of the UDF configuration. The instantiation of a class object should be lightweight,
    i.e. the constructor should not perform any time or resource consuming tasks, like
    loading a language model.

    Parameters:
        inputs:
            Definition of the input columns.
        outputs:
            Definition of the output columns.
        setup:
            Explicit algorithm setup description. This will override the algorithm json.
    """

    inputs: AlgoInput = AlgoInput()
    outputs: AlgoOutput = AlgoOutput()
    setup: str | None = None

    @property
    def error_message_column(self) -> str:
        return self.outputs.error_message.name

    @property
    def setup_column(self) -> str:
        return self.outputs.setup.name

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        """
        An entry point for the algorithm implementation.

        The input is provided as a DataFrame with rows collected from the UDF context
        using the `ctx.get_dataframe` functions. The dataframe is expected to have all
        the mandatory columns defined by the `input_decl`. However, it may also contain
        some additional columns, in a scenario when this algorithm is called by another
        algorithm.

        The returned data is also a DataFrame. The output should be a union of the
        columns defined by the `output_decl` and the input columns.

        The caller shall assume that the data will be written to the provided DataFrame.
        A copy should be made when it is desirable to keep the original DataFrame
        intact.

        The method can throw an exception, which will result in a call to `set_empty`
        with the provided exception stack as the error message.
        """
        batch_df[self.outputs.setup.name] = self.algo_setup
        return batch_df

    def set_empty(
        self, output_df: pd.DataFrame, error_message: str | None = None
    ) -> pd.DataFrame:
        """
        This function is used for setting default, i.e. empty, values for the columns
        the algorithm creates. Optionally, an error message can be specified, otherwise
        it will be set as empty.

        The caller shall assume that the data will be written to the provided DataFrame.
        A copy should be made when it is desirable to keep the original DataFrame
        intact.

        The method is not expected to throw an exception.
        """
        output_df[self.error_message_column] = error_message
        output_df[self.outputs.setup.name] = self.algo_setup
        return output_df

    @property
    def exclude_from_setup(self) -> set[str]:
        """
        A set of parameters to be ignored when creating a json with the algorithm
        configuration. These are the parameters that considered "static" in some sense,
        or parameters that should not affect the output of the algorithm.
        """
        return {"inputs", "outputs", "setup", "model_config"}

    @property
    def algo_dict(self) -> dict[str, Any]:
        """
        Algorithm configuration dictionary.
        """
        return {
            type(self).__name__: self.model_dump(
                exclude_unset=True, exclude=self.exclude_from_setup
            )
        }

    @property
    def algo_setup(self) -> str:
        """
        Algorithm setup, which is the algorithm json unless provided externally.
        """
        if self.setup:
            return self.setup
        return json.dumps(self.algo_dict)
