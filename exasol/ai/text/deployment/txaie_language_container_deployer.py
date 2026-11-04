import importlib.metadata
from pathlib import Path

from exasol.python_extension_common.deployment.language_container_deployer import (
    LanguageContainerDeployer,
)

from exasol.ai.text.deployment.language_container import CONTAINER_NAME


class TXAIELanguageContainerDeployer(LanguageContainerDeployer):
    PROJECT_NAME = "exasol-text-ai-extension"
    SLC_CLOUDFRONT_URL = "https://dut5tonqye28.cloudfront.net"
    SLC_PATH_IN_S3_BUCKET = "ai_lab/text_ai/slc"
    SLC_NAME = f"{CONTAINER_NAME}_release_x64.tar.gz"

    @classmethod
    def slc_s3_upload_path(cls, version: str | None = None) -> str:
        """
        Path for uploading the SLC to AWS S3 bucket.
        """
        version = version or importlib.metadata.version(cls.PROJECT_NAME)
        return f"{cls.SLC_PATH_IN_S3_BUCKET}/{version}/{cls.SLC_NAME}"

    @classmethod
    def slc_download_url(cls, version: str | None = None) -> str:
        """
        Path for downloading the SLC from AWS via cloudfront distribution.
        """
        path = cls.slc_s3_upload_path(version)
        return f"{cls.SLC_CLOUDFRONT_URL}/{path}"

    def run(
        self,
        container_file: Path | None = None,
        bucket_file_path: str | None = None,
        alter_system: bool = True,
        allow_override: bool = False,
        wait_for_completion: bool = True,
        print_activation_statements: bool = True,
    ) -> None:
        if not bucket_file_path:
            bucket_file_path = self.SLC_NAME
        super().run(
            container_file,
            bucket_file_path,
            alter_system,
            allow_override,
            wait_for_completion,
            print_activation_statements,
        )
