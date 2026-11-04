# pylint: disable=line-too-long
from collections.abc import Generator

import pytest
from transformers import AutoModelForTokenClassification

from test.integration.utils.model_utils import download_model_to_bucketfs
from test.utils.helpers import DEFAULT_NAMED_ENTITY_MODEL


@pytest.fixture(scope="session")
def end2end_named_entity_model():
    return DEFAULT_NAMED_ENTITY_MODEL


@pytest.fixture(scope="session")
def end2end_named_entity_model_revision():
    return "cb3c67c57e2091e4f65894e8653c1c9617567116"


@pytest.fixture(scope="session")
def end2end_named_entity_model_download(
    end2end_named_entity_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    """
    Downloads a token classification model into the BucketFS, to be used from a UDF.
    """
    with download_model_to_bucketfs(
        end2end_named_entity_model,
        model_subdir,
        "token-classification",
        backend_aware_bucketfs_params,
        AutoModelForTokenClassification,
    ):
        yield
