from typing import (
    Any,
    Literal,
    cast,
)

import pandas as pd
from pydantic import Field
from transformers import (
    AutoModelForTokenClassification,
    TokenClassificationPipeline,
)

from exasol.ai.text.impl.nlp.algorithms.named_entity.named_entity_recognition import (
    NamedEntityRecognition,
)
from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import (
    DEVICE_CPU,
    UDFHftBase,
)
from exasol.ai.text.impl.utils.pandas_utils import (
    adjust_result_span,
    generate_random_name,
    merge_result_with_input,
)
from exasol.ai.text.impl.utils.transformers_utils import load_transformers_pipline


class HftNamedEntityRecognition(NamedEntityRecognition, UDFHftBase):
    """
    A named entity recognition algorithm using the TokenClassificationPipeline from the
    Huggingface transformers library.

    algorithm discriminator: huggingface-token-classification

    Parameters:
        The descriptions of some of the parameters are copied from the Huggingface
        Transformers documentation.

        ignore_labels:
            A list of labels to ignore.
        aggregation_strategy:
            The strategy to fuse (or not) tokens based on the model prediction.
            “none”:
                Will simply not do any aggregation and simply return raw results from
                the model
            “simple”:
                Will attempt to group entities following the default schema. (A, B-TAG),
                (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being
                [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”},
                {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will
                end up as different entities. On word based languages, we might end up
                splitting words undesirably : Imagine Microsoft being tagged as
                [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}].
                Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate
                words (on languages that support that meaning, which is basically tokens
                separated by a space). These mitigations will only work on real words,
                “New york” might still be tagged with two different entities.
            “first”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. Words will simply use the
                tag of the first token of the word when there is ambiguity.
            “average”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. scores will be averaged
                first across tokens, and then the maximum label is applied.
            “max”:
                Works only on word based models. Will use the SIMPLE strategy except
                that words, cannot end up with different tags. Word entity will simply
                be the token with the maximum score.

            Defaults to "simple".
    """

    algorithm_name: Literal["huggingface-token-classification"] = Field(
        "huggingface-token-classification", repr=False
    )
    ignore_labels: list[str] | None = None
    aggregation_strategy: str = "simple"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._token_classifier: TokenClassificationPipeline | None = None

    def _lazy_init(self, exa):
        if self._token_classifier is None:
            # Going forward, the device should be set to the ID of the allocated GPU.
            # For now, we will run the model on a CPU.
            classifier = load_transformers_pipline(
                exa,
                bucketfs_conn_name=self.bucketfs_conn_name,
                sub_dir=self.sub_dir,
                device=DEVICE_CPU,
                model_name=self.model_name,
                task_type="token-classification",
                model_factory=AutoModelForTokenClassification,
            )
            self._token_classifier = cast(TokenClassificationPipeline, classifier)

    def _rename_classifier_columns(self, tokens_df: pd.DataFrame) -> pd.DataFrame:

        _, entity_begin_column, entity_end_column = (
            self.outputs.entity_span.column_names
        )
        # The output of the classifier may be slightly different depending on the
        # aggregation strategy. The entity name can be called either "entity" or
        # "entity_group".
        old_entity = "entity_group" if "entity_group" in tokens_df.columns else "entity"
        column_mapping = {
            "start": entity_begin_column,
            "end": entity_end_column,
            "word": self.entity_column,
            old_entity: self.entity_type_column,
            "score": self.score_column,
        }

        # Occasionally, some expected columns can be missing in the prediction output.
        # Make sure this is not the case.
        if any(col not in tokens_df.columns for col in column_mapping):
            raise KeyError(
                "Some of the expected columns were not found in prediction "
                f"results. The expected columns are: {column_mapping.keys()}. "
                f"Prediction results contain columns: {tokens_df.columns}"
            )

        tokens_df.rename(columns=column_mapping, inplace=True)
        return tokens_df[column_mapping.values()]

    @property
    def exclude_from_setup(self) -> set[str]:
        return {*super().exclude_from_setup, *self.bucketfs_param_names}

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:

        # Run the classifier on the text column
        tokens = self.run_classifier(exa, batch_df[self.text_column].tolist())

        # It's possible that no tokens are found in any of the texts.
        if sum(len(txt_tokens) for txt_tokens in tokens) == 0:
            return self.set_empty(batch_df)

        # Put the results in a single DataFrame, keeping track of which tokens belong
        # to which text.
        concat_tokens: list[dict[str, Any]] = []
        concat_text_seq: list[int] = []
        for text_seq, text_tokens in enumerate(tokens):
            concat_tokens.extend(text_tokens)
            concat_text_seq.extend([text_seq] * len(text_tokens))
        tokens_df = pd.DataFrame(concat_tokens)
        tokens_df = self._rename_classifier_columns(tokens_df)

        # Set the sequential number of the input text in the result.
        text_seq_column = generate_random_name([*batch_df.columns, *tokens_df.columns])
        tokens_df[text_seq_column] = concat_text_seq

        # Merge the result with the input
        output_df = merge_result_with_input(
            input_df=batch_df, result_df=tokens_df, merge_on_column=text_seq_column
        )
        output_df[self.error_message_column] = None
        output_df = adjust_result_span(
            output_df=output_df,
            text_span=self.inputs.text_span,
            result_span=self.outputs.entity_span,
        )
        return super().run_impl(output_df, exa)

    def run_classifier(self, exa, texts: list[str]) -> list[list[dict[str, Any]]]:
        """
        Each returned token is represented by a dictionary with fields like 'start',
        'end', etc. The function return a list of tokens for each of the provided
        texts.
        """
        self._lazy_init(exa)
        ignore_labels_kwargs = (
            {"ignore_labels": self.ignore_labels} if self.ignore_labels else {}
        )
        token_classifier = cast(TokenClassificationPipeline, self._token_classifier)
        return token_classifier(
            texts,
            aggregation_strategy=self.aggregation_strategy,
            **ignore_labels_kwargs,
        )
