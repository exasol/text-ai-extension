# pylint: disable=line-too-long
from collections.abc import Generator

import pytest
from transformers import AutoModel

from test.integration.utils.model_utils import download_model_to_bucketfs
from test.utils.helpers import DEFAULT_FEATURE_EXTRACTION_MODEL


@pytest.fixture(scope="session")
def end2end_feature_extraction_model():
    return DEFAULT_FEATURE_EXTRACTION_MODEL


@pytest.fixture(scope="session")
def end2end_feature_extraction_model_revision():
    return "8949b909ec900327062f0ebf497f51aef5e6f0c8"


@pytest.fixture(scope="session")
def end2end_feature_extraction_model_download(
    end2end_feature_extraction_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    with download_model_to_bucketfs(
        end2end_feature_extraction_model,
        model_subdir,
        "feature-extraction",
        backend_aware_bucketfs_params,
        AutoModel,
    ):
        yield
