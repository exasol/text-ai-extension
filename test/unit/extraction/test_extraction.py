import pytest
from pydantic import ValidationError

from exasol.ai.text.extraction.abstract_extraction import Output
from exasol.ai.text.extraction.extraction import Extraction
from exasol.ai.text.extractors import BranchExtractor as Branch
from exasol.ai.text.extractors import PipelineExtractor as Pipeline
from exasol.ai.text.extractors.branch_extractor_helper import ordinal
from test.utils.helpers import (
    keyword_search_extractor,
    named_entity_extractor,
    not_raises,
    source_table_extractor,
    topic_classifier_extractor,
)


@pytest.mark.parametrize(
    "extractor,expected_messages",
    [
        (
            named_entity_extractor(),
            [
                "  - its top-level is a NamedEntityExtractor (named_entity_settings=",
            ],
        ),
        (
            Pipeline(steps=[named_entity_extractor()]),
            [
                "  - its top-level is a PipelineExtractor that is no Source Extractor, because",
                "  - its first step is a NamedEntityExtractor (named_entity_settings=",
            ],
        ),
        (
            Branch(paths=[source_table_extractor(), named_entity_extractor()]),
            [
                "  - its top-level is a BranchExtractor that is no Source Extractor, because",
                "  - its second path is a NamedEntityExtractor (named_entity_settings=",
            ],
        ),
        (
            Branch(
                paths=[
                    source_table_extractor(),
                    Pipeline(steps=[named_entity_extractor()]),
                ]
            ),
            [
                "  - its top-level is a BranchExtractor that is no Source Extractor, because",
                "  - its second path is a PipelineExtractor that is no Source Extractor, because",
                "  - its first step is a NamedEntityExtractor (named_entity_settings=",
            ],
        ),
        (
            Pipeline(
                steps=[
                    Branch(paths=[source_table_extractor(), named_entity_extractor()])
                ]
            ),
            [
                "  - its top-level is a PipelineExtractor that is no Source Extractor, because",
                "  - its first step is a BranchExtractor that is no Source Extractor, because",
                "  - its second path is a NamedEntityExtractor (named_entity_settings=",
            ],
        ),
    ],
)
def test_invalid_extraction(extractor, expected_messages, output):
    expected = "\n".join(
        [
            "Value error, the specified extraction is invalid, because no Source Extractor was found:"
        ]
        + expected_messages
    )
    with pytest.raises(ValidationError) as ex:
        Extraction(extractor=extractor, output=output)
    if expected not in str(ex.value):
        print(f"\n\nexpected: {expected}\nactual: {ex.value}")
    assert expected in str(ex.value), f"\n\nexpected: {expected}\nactual: {ex.value}"


@pytest.mark.parametrize(
    "extractor",
    [
        source_table_extractor(),
        Pipeline(steps=[source_table_extractor(), named_entity_extractor()]),
        Branch(paths=[source_table_extractor(), source_table_extractor()]),
        Pipeline(
            steps=[
                source_table_extractor(),
                Branch(
                    paths=[
                        named_entity_extractor(),
                        topic_classifier_extractor(),
                        keyword_search_extractor(),
                    ]
                ),
            ]
        ),
    ],
)
def test_valid_extraction(extractor, output):
    with not_raises(ValidationError):
        extraction = Extraction(extractor=extractor, output=output)
        assert extraction.extractor == extractor
        assert extraction.output == output


@pytest.mark.parametrize(
    "schema",
    [
        "",
        ":",
        ".",
        "_",
    ],
)
def test_output_invalid_schema(schema):
    with pytest.raises(ValidationError):
        Output(db_schema=schema)


@pytest.mark.parametrize(
    "schema",
    [
        "A",
        "1",
        "A_",
    ],
)
def test_output_valid_schema(schema):
    with not_raises(ValidationError):
        Output(db_schema=schema)


LONG_PREFIX = "x" * 50
TOO_LONG_PREFIX = "A" + LONG_PREFIX


@pytest.mark.parametrize("prefix", ["A", "1", "A_", LONG_PREFIX, None])
def test_output_valid_prefix(prefix):
    with not_raises(ValidationError):
        Output(db_schema="x", table_prefix=prefix)


@pytest.mark.parametrize("prefix", ["", ":", ".", "_", TOO_LONG_PREFIX])
def test_output_invalid_table_prefix(prefix):
    with pytest.raises(ValidationError):
        Output(db_schema="x", table_prefix=prefix)


def test_ordinal_zero():
    with pytest.raises(ValueError):
        ordinal(0)


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (20, "20th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (24, "24th"),
        (101, "101st"),
        (102, "102nd"),
        (103, "103rd"),
        (104, "104th"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
        (120, "120th"),
        (121, "121st"),
        (122, "122nd"),
        (123, "123rd"),
        (124, "124th"),
    ],
)
def test_ordinals(n, expected):
    assert ordinal(n) == expected
