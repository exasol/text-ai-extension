import pandas as pd


class DataFrameMatcher:
    """
    A Matcher class that compares two DataFrames and can produce a diff on cell-level.
    This is class is very useful when comparing dataframes with many columns.
    """

    def __init__(self, expected: pd.DataFrame, result: pd.DataFrame):
        self.columns_diff = result.columns.symmetric_difference(expected.columns)
        self.index_diff = result.index.symmetric_difference(expected.index)
        self.compare: pd.DataFrame | None = None
        if self.columns_diff.empty and self.index_diff.empty:
            self.compare = result.compare(
                expected[result.columns], result_names=("result", "expected")
            )

    def __bool__(self) -> bool:
        return (
            self.columns_diff.empty
            and self.index_diff.empty
            and self.compare is not None
            and self.compare.empty
        )

    def diff(self) -> str:
        if not self.columns_diff.empty:
            return f"Column Diff: {self.columns_diff}"
        if not self.index_diff.empty:
            return f"Index Diff: {self.index_diff}"
        return self.compare.to_string()
