import random
import string
from collections.abc import Iterable

import pandas as pd

from exasol.ai.text.impl.db_schema import SpanColumns


def get_random_object_name(name_length: int = 20) -> str:
    """
    Generates a random name for a database object, e.g. a UDF.

    The name should be long enough to make the probability of the name collision,
    when generating a sequence of such names, infinitesimally small. For a sequence
    of length N this probability approximately equals (N * (N - 1)) / (2 * A ** L),
    where A is the size of the alphabet, i.e. 26, and L is the name length.
    For example, if the name is 20 characters long, and we generate a sequence of 100
    names every second then the average collision waiting time is 4e24 seconds. This
    is about 10 million times the age of our universe.
    """
    return "".join(random.choice(string.ascii_uppercase) for _ in range(name_length))


def generate_random_name(other_than: Iterable[str], name_length: int = 10) -> str:
    result = None
    existing = list(other_than)
    while result is None or result in existing:
        result = get_random_object_name(name_length)
    return result


def ensure_unique_name(proposed: str, other_than: Iterable[str]) -> str:
    while proposed in other_than:
        proposed = proposed + "_"
    return proposed


def copy_span_columns(
    df: pd.DataFrame, from_span: SpanColumns, to_span: SpanColumns
) -> pd.DataFrame:
    for in_col, out_col in zip(from_span.column_names, to_span.column_names):
        df[out_col] = df[in_col]
    return df


def merge_result_with_input(
    input_df: pd.DataFrame, result_df: pd.DataFrame, merge_on_column: str
) -> pd.DataFrame:
    """
    Merges two DataFrames - the input and the result with 1->n relationship. The
    `result_df` should have a column with the name `merge_on_column` that contains
    indices of the corresponding rows in the `input_df`.
    There could be some intersection of the columns. Before merging, the columns present
    in both DataFrames are removed from `input_df`.
    """

    # Make sure the input and the result do not have any columns in common.
    input_df = input_df[
        [col for col in input_df.columns if col not in result_df.columns]
    ]
    # Create the "merge_on" column in the input.
    input_df[merge_on_column] = range(len(input_df))
    # Merge the DataSets and remove the "merge_on" column.
    output_df = pd.merge(input_df, result_df, on=merge_on_column, how="left")
    output_df.drop(merge_on_column, axis=1, inplace=True)
    return output_df


def adjust_result_span(
    output_df: pd.DataFrame, text_span: SpanColumns, result_span: SpanColumns
) -> pd.DataFrame:
    """
    Performs three operations:
    - Makes the result span relative to the start of the document.
    - Checks that the result span fits within the text span and raises and error
      if it doesn't.
    - Copies the text_doc_id to the result_doc_id.
    """

    text_span_columns = text_span.column_names
    text_doc_id_column, text_begin_column, text_end_column = text_span_columns
    result_span_columns = result_span.column_names
    result_doc_id_column, result_begin_column, result_end_column = result_span_columns

    output_df[result_doc_id_column] = output_df[text_doc_id_column]
    not_na = output_df[result_begin_column].notna()
    if not not_na.all():
        output_df.loc[~not_na, result_doc_id_column] = None
    output_df.loc[not_na, result_doc_id_column] = output_df.loc[
        not_na, text_doc_id_column
    ]
    output_df.loc[not_na, result_begin_column] += output_df.loc[
        not_na, text_begin_column
    ]
    output_df.loc[not_na, result_end_column] += output_df.loc[not_na, text_begin_column]
    span_end_df = output_df[[text_end_column, result_end_column]].dropna()
    result_does_not_fit_into_original_text = ~(
        span_end_df[text_end_column] >= span_end_df[result_end_column]
    )
    if result_does_not_fit_into_original_text.any():
        with pd.option_context("display.max_columns", 6):
            wrong_spans = output_df[
                list(text_span_columns) + list(result_span_columns)
            ][result_does_not_fit_into_original_text]
            raise RuntimeError(
                "The span of a text operation result does not fit within "
                "the span of the original text.\n"
                f"{wrong_spans}"
            )
    return output_df
