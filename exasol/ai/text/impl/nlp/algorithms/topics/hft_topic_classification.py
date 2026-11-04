from typing import (
    Any,
    Literal,
    cast,
)

import pandas as pd
from pydantic import Field
from transformers import (
    AutoModelForSequenceClassification,
    ZeroShotClassificationPipeline,
)

from exasol.ai.text.impl.nlp.algorithms.topics.topic_classification import (
    TopicClassification,
)
from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import (
    DEVICE_CPU,
    UDFHftBase,
)
from exasol.ai.text.impl.utils.pandas_utils import (
    generate_random_name,
    merge_result_with_input,
)
from exasol.ai.text.impl.utils.transformers_utils import load_transformers_pipline


class HftTopicClassification(TopicClassification, UDFHftBase):
    """
    A topic classification algorithm using the ZeroShotClassificationPipeline from the
    Huggingface transformers library.

    algorithm discriminator: huggingface-zero-shot-classification

    Parameters:
        The descriptions of the parameters are copied from the Huggingface Transformers
        documentation.

        hypothesis_template:
            The template used to turn each label into an NLI-style hypothesis. This
            template must include a {} or similar syntax for the candidate label to be
            inserted into the template. The default template is "This example is {}.".
            With the candidate label "sports", this would be fed into the model like
            "<cls> sequence to classify <sep> This example is sports . <sep>". The
            default template works well in many cases, but it may be worthwhile to
            experiment with different templates depending on the task setting.
        multi_label:
            Indicates if multiple candidate labels can be true. If False, the scores
            are normalized such that the sum of the label likelihoods for each sequence
            is 1. If True, the labels are considered independent and probabilities are
            normalized for each candidate by doing a softmax of the entailment score vs.
            the contradiction score. The default value is False.
    """

    algorithm_name: Literal["huggingface-zero-shot-classification"] = Field(
        "huggingface-zero-shot-classification", repr=False
    )
    hypothesis_template: str | None = None
    multi_label: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._topic_classifier: ZeroShotClassificationPipeline | None = None

    def _lazy_init(self, exa):
        if self._topic_classifier is None:
            # Going forward, the device should be set to the ID of the allocated GPU.
            # For now, we will run the model on a CPU.
            classifier = load_transformers_pipline(
                exa,
                bucketfs_conn_name=self.bucketfs_conn_name,
                sub_dir=self.sub_dir,
                device=DEVICE_CPU,
                model_name=self.model_name,
                task_type="zero-shot-classification",
                model_factory=AutoModelForSequenceClassification,
            )
            self._topic_classifier = cast(ZeroShotClassificationPipeline, classifier)

    @property
    def exclude_from_setup(self) -> set[str]:
        return {*super().exclude_from_setup, *self.bucketfs_param_names}

    def run_impl(self, batch_df: pd.DataFrame, exa) -> pd.DataFrame:

        # Run the classifier on the text column
        results = self.run_classifier(exa, batch_df[self.text_column].tolist())

        # Put the results in a single DataFrame.
        concat_labels: list[str] = []
        concat_scores: list[float] = []
        concat_text_seq: list[int] = []
        n_topics = len(self.topics)
        n_texts = len(batch_df)
        for text_seq, text_result in enumerate(results):
            concat_labels.extend(text_result["labels"])
            concat_scores.extend(text_result["scores"])
            concat_text_seq.extend([text_seq] * n_topics)

        # Validate the results. All column vectors should be the same length,
        # as the pipeline should always return scores for all suggested labels.
        assert (
            len(concat_labels)
            == len(concat_scores)
            == len(concat_text_seq)
            == n_texts * n_topics
        ), "Unexpected total of the topic predictions."

        # Temporary column for merging the results with the input.
        text_seq_column = generate_random_name(
            [*batch_df.columns, self.topic_column, self.score_column, self.rank_column]
        )

        # When creating the rank column we can take advantage of the fact that the
        # output of the pipeline is sorted by the score in the descending order (see
        # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline).
        topic_df = pd.DataFrame(
            {
                self.topic_column: concat_labels,
                self.score_column: concat_scores,
                self.rank_column: list(range(1, n_topics + 1)) * n_texts,
                text_seq_column: concat_text_seq,
                self.error_message_column: None,
            }
        )

        output_df = merge_result_with_input(batch_df, topic_df, text_seq_column)
        return super().run_impl(output_df, exa)

    def run_classifier(self, exa, texts: list[str]) -> list[dict[str, Any]]:
        self._lazy_init(exa)
        hypothesis_template_kwargs = (
            {"hypothesis_template": self.hypothesis_template}
            if self.hypothesis_template
            else {}
        )
        topic_classifier = cast(ZeroShotClassificationPipeline, self._topic_classifier)
        return topic_classifier(
            texts,
            candidate_labels=self.topics,
            multi_label=self.multi_label,
            **hypothesis_template_kwargs,
        )
