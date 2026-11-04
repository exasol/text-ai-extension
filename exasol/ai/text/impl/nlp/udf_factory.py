import re
from textwrap import dedent

from exasol.analytics.query_handler.context.proxy.drop_udf_query import DropUDFQuery
from exasol.analytics.schema import (
    UDFName,
    UDFNameImpl,
)

from exasol.ai.text.impl.nlp.udf_runner import UDFRunner
from exasol.ai.text.impl.utils.constants import DEFAULT_LANGUAGE_ALIAS
from exasol.ai.text.impl.utils.pandas_utils import get_random_object_name
from exasol.ai.text.impl.utils.sql_utils import column_decl_to_str


class UDFFactory:
    """
    The class is a generic UDF factory, in a sense that it generates SQL queries for
    the UDF creation and deletion. However, it doesn't execute these queries, leaving
    it for the user.

    Parameters:
        udf_name:
            The name to be assigned to the UDF. The name can optionally include the
            schema. If not provided the UDF will be created with a random name in the
            current schema.
        language_alias:
            The language alias to use in the UDF.
    """

    def __init__(
        self,
        udf_name: UDFName | None = None,
        language_alias: str = DEFAULT_LANGUAGE_ALIAS,
    ):
        self._udf_name = udf_name or UDFNameImpl(get_random_object_name())
        self._language_alias = language_alias

    @property
    def udf_name(self) -> UDFName:
        """
        UDF name, including the schema if it was specified.
        """
        return self._udf_name

    def create_udf_query(self, runner: UDFRunner) -> str:
        """
        Builds an SQL query creating a UDF for the provided algorithm.
        """
        if runner.setup:
            # The double quotes in the `setup` must be escaped, to make sure the
            # field is treated as a string, not an enclosed json.
            escaped_setup = re.sub(r'(?<!\\)"', r"\"", runner.setup)
            runner = runner.model_copy(deep=True)
            runner.setup = escaped_setup
        runner_json = f'"""{runner.model_dump_json(serialize_as_any=True)}"""'

        return dedent(f"""
            CREATE OR REPLACE {self._language_alias} SET SCRIPT {self._udf_name.fully_qualified}(
                {column_decl_to_str(runner.input_decl)}
            ) EMITS (
                {column_decl_to_str(runner.output_decl)}
            ) AS

            from {UDFRunner.__module__} import UDFRunner

            runner_json = {runner_json}
            runner = UDFRunner.model_validate_json(runner_json)

            def run(ctx):
                runner.run(ctx, exa)
            /
        """)

    def delete_udf_query(self) -> str:
        """
        Builds an SQL query deleting the UDF if it was created.
        """
        return DropUDFQuery(self._udf_name).query_string
