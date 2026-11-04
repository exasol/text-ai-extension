import pytest

from exasol.ai.text.impl.utils.sql_utils import (
    MAX_NAME_LENGTH,
    create_db_object_name,
)


def test_create_db_object_name():
    assert create_db_object_name("abc") == "abc"


def test_create_db_object_name_args():
    assert create_db_object_name("abc", "def", "gh") == "abc_def_gh"


def test_create_db_object_name_hash():
    obj_name = create_db_object_name("abc", "x" * MAX_NAME_LENGTH)
    assert len(obj_name) <= MAX_NAME_LENGTH
    base, h = obj_name.split("_")
    assert base == "abc"
    assert h.isdigit()


def test_create_db_object_name_fail():
    with pytest.raises(ValueError):
        create_db_object_name("a" * (MAX_NAME_LENGTH + 1))


def test_create_db_object_name_args_fail():
    with pytest.raises(ValueError):
        create_db_object_name("a" * (MAX_NAME_LENGTH - 1), "xxx")
