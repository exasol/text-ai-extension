from dataclasses import fields
from typing import (
    Any,
    TypeVar,
)

import pandas as pd

from exasol.ai.text.impl.nlp.algorithms.udf_algo_base import UDFAlgoBase

T = TypeVar("T")


def map_fields(data_instance: T, field_map: dict[str, Any]) -> T:
    """
    Creates a copy of the provided arbitrary dataclass replacing the values of some
    known fields.
    """
    if not hasattr(data_instance, "__dataclass_fields__"):
        raise ValueError("Input to the `map_fields` must be a dataclass instance")

    data = {
        fld.name: getattr(data_instance, fld.name)
        for fld in fields(data_instance)  # type: ignore
    }

    for field_name, new_value in field_map.items():
        if field_name not in data_instance.__dataclass_fields__:
            raise ValueError(f"'{field_name}' is not a field in a dataclass")
        data[field_name] = new_value

    return data_instance.__class__(**data)


class UDFAlgoPipeline(UDFAlgoBase):
    """
    The base of an algorithm that is a pipeline of other algorithms ran sequentially.

    The output columns of each intermediate step should match the input columns of the
    subsequent step. Also, the declaration of the inputs/outputs of the pipeline itself
    should be compatible with the cumulative inputs and outputs of the algorithms. These
    conditions are validated and a ValueError is raised in case they are not satisfied.

    A derived pipeline class can use the `map_fields` function for wiring algorithms'
    inputs and outputs and making them compatible with the inputs and outputs of the
    pipeline.
    """

    algorithms: tuple[UDFAlgoBase, ...]

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.map_io()
        self._validate_io()

    def map_io(self) -> None:
        """
        A derived class can do the i/o wiring in this function.
        """

    def _validate_io(self) -> None:

        # Get the columns available at each step and make sure they cover the
        # input requirements of the algorithms.
        available_columns = set(self.inputs.order)
        for algo in self.algorithms:
            if not available_columns.issuperset(algo.inputs.order):
                raise ValueError(
                    "The following input columns are not available "
                    f"for the algorithm {type(algo).__name__}: "
                    f"{set(algo.inputs.order) - available_columns}"
                )
            available_columns.update(algo.outputs.order)

        # Make sure the collection of columns we end up with satisfies the output
        # requirements of the pipeline.
        if not available_columns.issuperset(self.outputs.order):
            raise ValueError(
                "The following output columns are not available "
                f"for the pipeline algorithm {type(self).__name__}: "
                f"{set(self.outputs.order) - available_columns}"
            )

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:
        for algorithm in self.algorithms:
            batch_df = algorithm.run_impl(batch_df, exa)
        return super().run_impl(batch_df, exa)

    def set_empty(
        self, output_df: pd.DataFrame, error_message: str | None = None
    ) -> pd.DataFrame:
        for algorithm in self.algorithms:
            output_df = algorithm.set_empty(output_df, error_message)
        return super().set_empty(output_df, error_message)

    @property
    def algo_dict(self) -> dict[str, Any]:
        step_dicts = [algo.algo_dict for algo in self.algorithms]
        return {type(self).__name__: step_dicts}
