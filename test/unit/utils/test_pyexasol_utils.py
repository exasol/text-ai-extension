from unittest.mock import (
    Mock,
    call,
)

import pytest

from exasol.ai.text.impl.utils.pyexasol_utils import autocommit_off


@pytest.mark.parametrize("initial", [True, False])
def test_autocommit_off(initial):
    mock = Mock(options={"autocommit": initial})
    with autocommit_off(mock):
        assert mock.set_autocommit.call_args == call(False)
    assert mock.set_autocommit.call_args == call(initial)


@pytest.mark.parametrize("initial", [True, False])
def test_autocommit_off_exception(initial):
    mock = Mock(options={"autocommit": initial})
    with pytest.raises(Exception):
        with autocommit_off(mock):
            assert mock.set_autocommit.call_args == call(False)
            raise Exception("")
    assert mock.set_autocommit.call_args == call(initial)
