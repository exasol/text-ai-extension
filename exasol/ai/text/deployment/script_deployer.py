from collections.abc import Generator

from pyexasol import ExaConnection

from exasol.ai.text.impl.scripts import Scripts


def enum_script_names() -> Generator[str, None, None]:
    """
    Generates a sequence of attribute names each of which is a
    function or a script creation query.
    """
    for attr_name in dir(Scripts):
        if (not attr_name.startswith("_")) and isinstance(
            getattr(Scripts, attr_name), property
        ):
            yield attr_name


def enum_scripts() -> Generator[str, None, None]:
    """
    Generates a sequence of function or script creation queries.
    """
    scripts = Scripts()
    for script_name in enum_script_names():
        yield getattr(scripts, script_name)


def create_scripts(pyexasol_connection: ExaConnection) -> None:
    """
    Creates all pre-defined functions and scripts in the database.
    """
    for create_query in enum_scripts():
        pyexasol_connection.execute(create_query)
