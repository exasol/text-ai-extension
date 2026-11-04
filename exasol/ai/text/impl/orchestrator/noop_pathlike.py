from collections.abc import (
    ByteString,
    Iterable,
)
from pathlib import PurePath
from typing import BinaryIO

import exasol.bucketfs as bfs

BUCKET_API = bfs.MountedBucket("svc", "bkt")


class NoOpPathLike(bfs.path.BucketPath):
    """
    Nonfunctional subclass of bfs.path.PathLike failing on read() and
    write() but allowing joinpath() and similar introspective operations.

    Currently, :class:`StageQueryHandler` must neither write to or read from
    the BucketFS as currently QueryHandlers are run in Python while running
    them in SQL by calling a UDF behaves differently.

    In UDFs, the stages used for data processing can perform arbitrary
    BucketFS operations.  Using the BucketFS for data transfer between
    StageQueryHandler and UDFs is only possible when using the real BucketFS
    but not with a path mounted from the file system.
    """

    def __init__(self, path: str | PurePath, bucket_api: bfs.BucketLike):
        super().__init__(path, bucket_api)

    def write(self, data: ByteString | BinaryIO | Iterable[ByteString]) -> None:
        raise NotImplementedError()

    @classmethod
    def default_instance(cls):
        return cls("/dir/file.txt", BUCKET_API)

    # Currently, dependency resolution returns exasol-bucketfs version 0.13.0
    # for which `BucketPath.__eq__()` is not implemented, yet.
    # As soon as dependencies are updated we can use
    # return super().__eq__(other)
    def __eq__(self, other) -> bool:
        if not isinstance(other, NoOpPathLike):
            return False
        return (self._path, self._bucket_api) == (other._path, other._bucket_api)
