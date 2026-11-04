# pylint: disable=line-too-long
from collections.abc import Generator

import pytest
from transformers import AutoModelForSequenceClassification

from test.integration.utils.model_utils import download_model_to_bucketfs
from test.utils.helpers import DEFAULT_NLI_MODEL


@pytest.fixture(scope="session")
def end2end_topic_model():
    return DEFAULT_NLI_MODEL


@pytest.fixture(scope="session")
def end2end_topic_model_revision():
    return "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"


@pytest.fixture(scope="session")
def end2end_topic_model_download(
    end2end_topic_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    """
    Downloads a zero-shot-classification model into the BucketFS, to be used from a UDF.
    """
    with download_model_to_bucketfs(
        end2end_topic_model,
        model_subdir,
        "zero-shot-classification",
        backend_aware_bucketfs_params,
        AutoModelForSequenceClassification,
    ):
        yield
