import os
import sys
from datetime import datetime
from pathlib import Path

import boto3

sys.path.append(str(Path(__file__).parent))
ROOT_PATH = Path(__file__).parent
EXPORT_PATH = ROOT_PATH / "export"

import nox

# imports all nox task provided by the toolbox
# noinspection PyUnresolvedReferences
from exasol.toolbox.nox.tasks import *

from exasol.ai.text.deployment.language_container import language_container_factory
from exasol.ai.text.deployment.txaie_language_container_deployer import (
    TXAIELanguageContainerDeployer,
)

# default actions to be run if nothing is explicitly specified with the -s option
nox.options.sessions = ["format:fix"]


@nox.session(name="slc:export", python=False)
def export_slc(session: nox.Session):
    """
    Builds and exports the Text AI SLC to
    project_root/export/exasol_text_ai_extension_container_release_x64.tar.gz
    """
    with language_container_factory() as container_builder:
        container_builder.export(EXPORT_PATH)


def build_slc(session: nox.Session) -> Path:
    slc_tar_gz = EXPORT_PATH / TXAIELanguageContainerDeployer.SLC_NAME
    if slc_tar_gz.exists():
        session.log(f"Deleting file {slc_tar_gz}")
        slc_tar_gz.unlink()
    export_slc(session)
    if not slc_tar_gz.exists():
        session.error(f"Couldn't find the built SLC: {slc_tar_gz}.")
    return slc_tar_gz


def upload_to_aws(file: Path) -> None:
    # Optional: get version from env variable GITHUB_REF_NAME
    # version = os.environ["GITHUB_REF_NAME"]
    bucket_file = TXAIELanguageContainerDeployer.slc_s3_upload_path()
    s3_client = boto3.client("s3")
    s3_client.upload_file(file, os.environ["AWS_S3_BUCKET"], bucket_file)

    # Create invalidation in the CloudFront distribution.
    # See: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Invalidation.html
    if not bucket_file.startswith("/"):
        bucket_file = "/" + bucket_file
    cloud_front_client = boto3.client("cloudfront")
    cloud_front_client.create_invalidation(
        DistributionId=os.environ["AWS_DISTRIBUTION_ID"],
        InvalidationBatch={
            "Paths": {"Quantity": 1, "Items": [bucket_file]},
            "CallerReference": str(datetime.now().timestamp()),
        },
    )


@nox.session(name="slc:s3-upload", python=False)
def upload_slc(session: nox.Session):
    """
    Performs the following steps:
    1. Build the SLC.
    2. Upload the archive to an S3 bucket.

    Requires the following environment variables:
    - AWS_S3_BUCKET
    - AWS_DISTRIBUTION_ID

    Prior to running this task the AWS CLI must be set up, providing the
    write-access to the bucket in question.
    """
    slc_tar_gz = build_slc(session)
    upload_to_aws(slc_tar_gz)
