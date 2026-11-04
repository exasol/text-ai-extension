import traceback

import pandas as pd
from pydantic import BaseModel

from exasol.ai.text.impl.db_schema import ColumnDecl
from exasol.ai.text.impl.nlp.algorithms import AlgorithmModel

MAX_ERROR_MESSAGE_LENGTH = 2000000


def truncate_string(string: str, max_length: int, truncation_ending: str):
    if len(truncation_ending) > max_length:
        raise ValueError(
            f"Truncation ending length {len(truncation_ending)} is larger than "
            f"max_length {max_length} characters"
        )
    if len(string) > max_length:
        return string[: max_length - len(truncation_ending)] + truncation_ending
    else:
        return string


class UDFRunner(BaseModel):
    """
    The class providing a wrapper around an algorithm or a pipeline, enabling them to
    be used in a UDF. This is also a root pydantic Model.

    The class provides some convenience properties, e.g. `setup` for accessing the
    namesake properties of its algorithm.

    Parameters:
        algorithm:
            An algorith is a Discriminated Union of algorithm classes. The
            discriminating tag is called "algorithm_name".
            All algorithms in the union must derive from the `UDFAlgoBase` class.
        batch_size:
            Number of rows processed by the algorithm in one call. The UDF performance
            is generally improved by reading and emitting batches, compared to
            row-by-row processing. Some algorithm can also take advantage from batch
            processing.
            Default value is 100.
    """

    algorithm: AlgorithmModel
    batch_size: int = 100

    def run(self, ctx, exa) -> None:
        """
        Runs the configured algorithm from a UDF.
        """
        while (batch_df := ctx.get_dataframe(num_rows=self.batch_size)) is not None:
            batch_df = self._run_algorithm(batch_df, exa)
            self._emit(batch_df, ctx)

    def _emit(self, batch_df: pd.DataFrame, ctx) -> None:
        try:
            # Need to replace nan with None for strings otherwise the UDF Client
            batch_df.where(pd.notna(batch_df), None, inplace=True)
            self._emit_batch(batch_df, ctx)
        except Exception:  # pylint: disable=broad-exception-caught
            self._handle_emit_exception(batch_df, ctx)

    def _emit_batch(self, batch_df: pd.DataFrame, ctx):
        emit_batch_df = batch_df[self.algorithm.outputs.column_names]
        ctx.emit(emit_batch_df)

    def _run_algorithm(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        try:
            batch_df = self.algorithm.run_impl(batch_df, exa)
        except Exception:  # pylint: disable=broad-exception-caught
            batch_df = self._handle_algorithm_run_exception(batch_df)
        return batch_df

    def _handle_algorithm_run_exception(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        error_message = traceback.format_exc()
        truncated_error_message = truncate_string(
            error_message, MAX_ERROR_MESSAGE_LENGTH, "..."
        )
        batch_df = self.algorithm.set_empty(
            batch_df, error_message=truncated_error_message
        )
        return batch_df

    def _handle_emit_exception(self, batch_df: pd.DataFrame, ctx) -> None:
        error_message = (
            f"Emit failed with error:\n"
            f"{traceback.format_exc()}\n"
            f"Outut Columns: {self.algorithm.outputs.column_names}\n"
            f"Dtypes: {batch_df.dtypes}\n"
            f"Batch:\n"
            f"{batch_df.to_string()}\n"
        )
        truncated_error_message = truncate_string(
            error_message, MAX_ERROR_MESSAGE_LENGTH, "..."
        )
        batch_df = self.algorithm.set_empty(
            batch_df, error_message=truncated_error_message
        )
        self._emit_batch(batch_df, ctx)

    @property
    def setup(self) -> str | None:
        return self.algorithm.setup

    @setup.setter
    def setup(self, value: str) -> None:
        self.algorithm.setup = value

    @property
    def algo_setup(self) -> str:
        return self.algorithm.algo_setup

    @property
    def input_decl(self) -> ColumnDecl:
        return self.algorithm.inputs.order

    @property
    def output_decl(self) -> ColumnDecl:
        return self.algorithm.outputs.order
