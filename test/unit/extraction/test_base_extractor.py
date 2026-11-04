import pytest
from pydantic import (
    BaseModel,
    ValidationError,
)

from exasol.ai.text.extractors import NamedEntityExtractor
from exasol.ai.text.extractors.base_extractor import format_extractor_name
from exasol.ai.text.extractors.extractor import (
    BaseExtractor,
    Extractor,
)
from test.utils.helpers import (
    named_entity_extractor,
    named_entity_settings,
)


class DummyExtractor(BaseExtractor):
    pass


class ExtractorWrapper(BaseModel):
    attribute: Extractor


@pytest.mark.parametrize(
    ["input_name", "expected_output"],
    [
        ("TextSearchExtractor", "TEXT_SEARCH"),
        ("textsearchExtractor", "TEXTSEARCH"),
        ("Text_Search", "TEXT_SEARCH"),
        ("text_search", "TEXT_SEARCH"),
        ("TEXT_SEARCH", "TEXT_SEARCH"),
        ("TEXTSEARCH", "TEXTSEARCH"),
    ],
)
def test_format_extractor_name(input_name, expected_output):
    assert format_extractor_name(input_name) == expected_output


def test_generic_extractor():
    """
    This test verifies the instantiation of an Extractor attribute using
    the pydantic discriminator "extractor_type".
    """
    wrapper = ExtractorWrapper(
        attribute={
            "extractor_type": "NamedEntityExtractor",
            "named_entity_settings": named_entity_settings(),
        }
    )
    assert isinstance(wrapper.attribute, NamedEntityExtractor)


def test_unsupported_extractor_type():
    with pytest.raises(ValidationError):
        ExtractorWrapper(attribute={"extractor_type": "NonExistingTestExtractor"})


def test_hidden_field():
    """
    Instantiate an extractor and verify that attribute ``extractor_type`` is
    missing in its string representation but still present when dumping the
    model. This also applies to the model's Json representation.
    """
    extractor = named_entity_extractor()
    str_repr = str(extractor)
    dump = extractor.model_dump()
    assert "extractor_type" not in str_repr
    assert dump["extractor_type"] == "NamedEntityExtractor"


def test_extractor_implicit_name():
    extractor = DummyExtractor()
    assert extractor.name == "DUMMY"


def test_extractor_explicit_name():
    extractor = DummyExtractor(name="MY_DUMMY")
    assert extractor.name == "MY_DUMMY"
