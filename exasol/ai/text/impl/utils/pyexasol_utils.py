import contextlib

import pyexasol


@contextlib.contextmanager
def autocommit_off(connection: pyexasol.ExaConnection):
    """
    See https://docs.python.org/3/reference/compound_stmts.html#finally

    If an exception occurs in any of the clauses and is not handled, the
    exception is temporarily saved. The finally clause is executed. If there
    is a saved exception it is re-raised at the end of the finally clause.
    """

    old_value = connection.options["autocommit"]
    connection.set_autocommit(False)
    try:
        yield connection
    except Exception:  # pylint: disable=broad-exception-caught
        connection.rollback()
        raise
    finally:
        connection.set_autocommit(old_value)
