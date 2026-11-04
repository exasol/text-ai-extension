from collections.abc import Iterable
from typing import (
    Any,
    Literal,
    cast,
)

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from pydantic import Field
from transformers import AutoModel

from exasol.ai.text.impl.nlp.algorithms.keywords.keyword_extractor import (
    KeywordExtractor,
    KeywordExtractorOutput,
)
from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import (
    DEVICE_CPU,
    UDFHftBase,
)
from exasol.ai.text.impl.utils.constants import DEFAULT_SPACY_PIPELINE
from exasol.ai.text.impl.utils.transformers_utils import load_transformers_pipline

DEFAULT_STOP_WORDS = None


class PatternRankKeywordExtractor(KeywordExtractor, UDFHftBase):
    """
    A keyword extractor algorithm using the KeyphraseVectorizers package. The algorithm
    prepares key-phrases based on the part-of-speech tags. The key-phrases subsequently
    get ranked by a KeyBERT model. For details see
    https://github.com/TimSchopf/KeyphraseVectorizers.
    The model should be a Huggingface transformers model. It should support the
    feature-extraction task.

    algorithm discriminator: pattern-rank-keyword-extractor

    Parameters:
        vec_kwargs:
            These parameters will be passed to the KeyphraseCountVectorizer. See
            https://github.com/TimSchopf/KeyphraseVectorizers/blob/77fdbc7504d8a4fc61714b83ce368087097c2b01/keyphrase_vectorizers/keyphrase_count_vectorizer.py#L26
            Please use only serializable types.
        kbx_kwargs:
            These parameters will be passed to the KeyBERT extrac_keywords method. See
            https://github.com/MaartenGr/KeyBERT/blob/38f036defa8130642ca4cdebf13382f4a1d8c0d1/keybert/_model.py#L64
            Please use only serializable types.
    """

    algorithm_name: Literal["pattern-rank-keyword-extractor"] = Field(
        "pattern-rank-keyword-extractor", repr=False
    )
    vec_kwargs: dict[str, Any] = {}
    kbx_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorizer: KeyphraseCountVectorizer | None = None
        self._extractor: KeyBERT | None = None

    def _lazy_init(self, exa):
        if self._vectorizer is None:
            self._vectorizer = self.create_vectorizer()
        if self._extractor is None:
            self._extractor = self.create_extractor(exa)

    def create_extractor(self, exa) -> KeyBERT:
        # Going forward, the device should be set to the ID of the allocated GPU.
        # For now, we will run the model on a CPU.
        # We will use the basic transformers AutoModel class for loading the
        # model, as we don't know much about the model except that it supports
        # the feature-extraction task. This should be sufficient because in this
        # instance we only need the embeddings from the model.
        transformers_model = load_transformers_pipline(
            exa,
            bucketfs_conn_name=self.bucketfs_conn_name,
            sub_dir=self.sub_dir,
            device=DEVICE_CPU,
            model_name=self.model_name,
            task_type="feature-extraction",
            model_factory=AutoModel,
        )
        return KeyBERT(model=transformers_model)

    def create_vectorizer(self) -> KeyphraseCountVectorizer:
        pipeline_kwargs: dict[str, Any] = {}
        if (
            "spacy_pipeline" not in self.vec_kwargs
            or self.vec_kwargs["spacy_pipeline"] is None
        ):
            self.vec_kwargs["spacy_pipeline"] = DEFAULT_SPACY_PIPELINE
        if "stop_words" not in self.vec_kwargs:
            # Anything, different from None or a List[str] for stop_words will require a NLTK stopword list
            # to be present inside the UDF. If it not inside the SLC, NLTK will try to download which is
            # unwanted behavior inside a UDF, because it might not be possible, depending on the DB configuration
            # and deployment.
            self.vec_kwargs["stop_words"] = DEFAULT_STOP_WORDS
        return KeyphraseCountVectorizer(**self.vec_kwargs, **pipeline_kwargs)

    @property
    def exclude_from_setup(self) -> set[str]:
        return {*super().exclude_from_setup, *self.bucketfs_param_names}

    def extract_keywords(self, exa, texts: Iterable[str]) -> KeywordExtractorOutput:
        self._lazy_init(exa)
        result = cast(KeyBERT, self._extractor).extract_keywords(
            list(texts), vectorizer=self._vectorizer, **self.kbx_kwargs
        )
        # Due to its own perception of "beautiful" the KeyBERT flattens the output in
        # case there is only one document.
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            return result
        return [result]
