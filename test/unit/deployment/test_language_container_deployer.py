import importlib.metadata
from pathlib import Path
from unittest.mock import (
    MagicMock,
    call,
    create_autospec,
)

import exasol.bucketfs as bfs
import pytest
from exasol.python_extension_common.deployment.language_container_deployer import (
    LanguageActivationLevel,
)
from pyexasol import ExaConnection

from exasol.ai.text.deployment.txaie_language_container_deployer import (
    TXAIELanguageContainerDeployer,
)


@pytest.fixture
def txaie_container_deployer() -> TXAIELanguageContainerDeployer:
    deployer = TXAIELanguageContainerDeployer(
        pyexasol_connection=create_autospec(ExaConnection),
        language_alias="PYTHON3_TEST",
        bucketfs_path=create_autospec(bfs.path.PathLike),
    )
    deployer.upload_container = MagicMock()
    deployer.activate_container = MagicMock()
    return deployer


def test_txaie_language_container_deployer(
    txaie_container_deployer: TXAIELanguageContainerDeployer,
):
    file_name = "txaie_container.tar.gz"
    file_path = Path(file_name)
    txaie_container_deployer.run(
        container_file=file_path,
        bucket_file_path=file_name,
        alter_system=True,
        allow_override=True,
        wait_for_completion=False,
    )
    assert txaie_container_deployer.upload_container.mock_calls == [
        call(file_path, file_name)
    ]
    expected_calls = [
        call(file_name, LanguageActivationLevel.System, True),
        call(file_name, LanguageActivationLevel.Session, True),
    ]
    assert txaie_container_deployer.activate_container.mock_calls == expected_calls


@pytest.mark.parametrize("version", ["1.2.3", None])
def test_slc_url(version: str | None):
    expected_version = version or importlib.metadata.version(
        TXAIELanguageContainerDeployer.PROJECT_NAME
    )
    suffix = "/".join(
        [
            TXAIELanguageContainerDeployer.SLC_PATH_IN_S3_BUCKET,
            expected_version,
            TXAIELanguageContainerDeployer.SLC_NAME,
        ]
    )
    s3_upload_path = TXAIELanguageContainerDeployer.slc_s3_upload_path(version)
    assert s3_upload_path == suffix
    download_url = TXAIELanguageContainerDeployer.slc_download_url(version)
    assert (
        download_url == f"{TXAIELanguageContainerDeployer.SLC_CLOUDFRONT_URL}/{suffix}"
    )
