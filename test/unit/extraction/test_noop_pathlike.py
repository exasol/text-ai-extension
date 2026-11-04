import exasol.bucketfs as bfs
import pytest

from exasol.ai.text.impl.orchestrator.noop_pathlike import NoOpPathLike
from test.utils.helpers import not_raises


@pytest.fixture
def testee():
    return NoOpPathLike.default_instance()


def test_rm(testee):
    with pytest.raises(FileNotFoundError):
        testee.rm()


def test_rmdir(testee):
    with not_raises(Exception):
        testee.rmdir(recursive=True)


def test_eq(testee):
    other = NoOpPathLike.default_instance()
    assert testee == other
    assert (testee / "a") == (other / "a")


def test_joinpath(testee):
    expected = f"{str(testee)}/a/b"
    actual = str(testee.joinpath("a", "b"))
    assert expected == actual


def test_truediv(testee):
    expected = f"{str(testee)}/a"
    actual = str(testee / "a")
    assert expected == actual


@pytest.mark.parametrize(
    "method",
    [
        "walk",
        "iterdir",
        "read",
    ],
)
def test_generators(testee, method):
    with pytest.raises(FileNotFoundError):
        exec(f"list(testee.{method}())")


def test_write(testee):
    data = bytearray([1, 2, 3, 4])
    with pytest.raises(NotImplementedError):
        testee.write(data)


@pytest.mark.parametrize(
    "property, expected",
    [
        ("name", "file.txt"),
        ("suffix", ".txt"),
        ("root", "/"),
    ],
)
def test_properties(testee, property, expected):
    assert expected == getattr(testee, property)


def test_parent(testee):
    actual = testee.parent
    assert isinstance(actual, bfs.path.BucketPath)
    assert str(actual) == "/dir"


@pytest.mark.parametrize(
    "method",
    [
        "as_uri",
        "as_udf_path",
        "exists",
        "is_dir",
        "is_file",
    ],
)
def test_other_methods(testee, method):
    actual = exec(f"testee.{method}()")
    assert actual is None
