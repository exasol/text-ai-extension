import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import exasol.bucketfs as bfs
import pytest
import transformers
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_bucket_fs_model_specification_from_model_specs,
)
from exasol_transformers_extension.utils.model_specification import ModelSpecification

from exasol.ai.text.impl.utils.transformers_utils import download_transformers_model


def download_model_locally(
    model_specification: ModelSpecification,
    model_revision: str,
    tmp_path_factory: pytest.TempPathFactory,
    sub_dir: str,
    model_factory,
) -> bfs.path.PathLike:
    """
    Downloads the specified model to the given location in a local file system and
    returns the BucketFS PathLike object pointing to this location.
    """
    current_model_specs = get_bucket_fs_model_specification_from_model_specs(
        model_specification, "", Path(sub_dir)
    )
    model_rel_path = current_model_specs.get_bucketfs_model_save_path()
    tmp_path = tmp_path_factory.mktemp(current_model_specs.task_type)
    model_path = tmp_path / model_rel_path
    model_name = model_specification.model_name
    cache_dir = tmp_path / "cache" / model_name
    for model_factory in [model_factory, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(
            model_name, cache_dir=cache_dir, revision=model_revision
        )
        model.save_pretrained(model_path)
    return bfs.path.build_path(
        backend=bfs.path.StorageBackend.mounted, base_path=str(tmp_path)
    )


@contextmanager
def download_model_to_bucketfs(
    model_name: str,
    model_subdir: str,
    task_type: str,
    backend_aware_bucketfs_params,
    model_factory,
) -> Iterator[None]:
    """
    Downloads the specified model into the BucketFS, to be used from a UDF.
    Then deletes it.
    """
    bucketfs_location = bfs.path.build_path(**backend_aware_bucketfs_params)
    bfs_path = download_transformers_model(
        bucketfs_location=bucketfs_location,
        sub_dir=model_subdir,
        task_type=task_type,
        model_name=model_name,
        model_factory=model_factory,
    )
    time.sleep(20)
    yield
    cleanup_bucketfs(bfs_path)


def cleanup_bucketfs(bucketfs_path: bfs.path.PathLike):
    if bucketfs_path.is_dir():
        bucketfs_path.rmdir(recursive=True)
    elif bucketfs_path.is_file():
        bucketfs_path.rm()
