import hashlib

from exasol.ai.text.impl.db_schema import (
    ColumnDecl,
    ColumnSpec,
    IOColumnFlags,
)

MAX_NAME_LENGTH = 128


def sql_text_value(text: str) -> str:
    return f"""'{text.replace("'", "''")}'"""


def boolean_column_spec(name: str) -> ColumnSpec:
    return ColumnSpec(name=name, sql_type="BOOLEAN")


def varchar_column_spec(
    name: str,
    size: int = 2000000,
    is_key: bool = False,
    flags: IOColumnFlags = IOColumnFlags.NONE,
) -> ColumnSpec:
    return ColumnSpec(
        name=name, sql_type=f"VARCHAR({size})", is_key=is_key, flags=flags
    )


def integer_column_spec(
    name: str, is_key: bool = False, flags: IOColumnFlags = IOColumnFlags.NONE
) -> ColumnSpec:
    return ColumnSpec(name=name, sql_type="INTEGER", is_key=is_key, flags=flags)


def double_column_spec(
    name: str, flags: IOColumnFlags = IOColumnFlags.NONE
) -> ColumnSpec:
    return ColumnSpec(name=name, sql_type="DOUBLE PRECISION", flags=flags)


def column_decl_to_str(decl: ColumnDecl) -> str:
    return ", ".join(f"{col.quoted_name} {col.sql_type}" for col in decl)


def create_db_object_name(base: str, *args) -> str:
    """
    Forms a new name for a database object, e.g. a table.
    The name shall start from the base string and preferably also include all other
    parts provided as `args`.
    We should observe the length restriction on object names, currently 128 characters.
    If the entire combined string does not fit into this limit, the name should be
    made from the base and the hash of the extra parts. If that doesn't fit then an
    exception will be raised.
    """
    suffix = "_".join(str(arg) for arg in args)
    if suffix:
        candidate = f"{base}_{suffix}"
        if len(candidate) > MAX_NAME_LENGTH:
            # Try to hash the extra bits
            hash_bytes = hashlib.sha3_256(suffix.encode("utf-8")).digest()
            hash_int = int.from_bytes(hash_bytes, byteorder="big")
            hash_mask = (1 << 64) - 1
            candidate = f"{base}_{hash_int & hash_mask}"
    else:
        candidate = base
    if len(candidate) > MAX_NAME_LENGTH:
        raise ValueError(
            "Unable to create a new database object name "
            f"based on {base}. The base is too long."
        )
    return candidate
