# pylint: disable=line-too-long
from collections.abc import Generator
from dataclasses import dataclass
from textwrap import dedent

import exasol.bucketfs as bfs
import pytest
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from transformers import AutoModelForSequenceClassification

from test.integration.utils.model_utils import (
    download_model_locally,
    download_model_to_bucketfs,
)
from test.integration.utils.text_test_case import TextTestCase


@pytest.fixture(scope="session")
def topic_model():
    return "cross-encoder/nli-deberta-base"


@pytest.fixture(scope="session")
def topic_model_revision():
    return "c962d60ec3063a58de0b3c8851a98c90f7255a36"


@pytest.fixture(scope="session")
def topic_model_location(
    tmp_path_factory, topic_model, topic_model_revision, model_subdir
) -> bfs.path.PathLike:
    model_specification = ModelSpecification(topic_model, "zero-shot-classification")
    return download_model_locally(
        model_specification,
        topic_model_revision,
        tmp_path_factory,
        model_subdir,
        AutoModelForSequenceClassification,
    )


@pytest.fixture(scope="session")
def topic_model_download(
    topic_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    """
    Downloads a zero-shot-classification model into the BucketFS, to be used from a UDF.
    """
    with download_model_to_bucketfs(
        topic_model,
        model_subdir,
        "zero-shot-classification",
        backend_aware_bucketfs_params,
        AutoModelForSequenceClassification,
    ):
        yield


@dataclass
class TopicTestCase(TextTestCase):
    topic: str


@pytest.fixture
def topic_list() -> list[str]:
    return [
        "psychology",
        "microbiology",
        "software development",
        "archeology",
        "agriculture",
    ]


@pytest.fixture
def topic_test_cases() -> list[TopicTestCase]:
    return [
        TopicTestCase(
            text_doc_id=3498,
            text_char_begin=200,
            text=dedent("""
                The accuracy of perception of facial emotion expressions was studied in individuals with low and high psychopathic personality characteristics in a sample of 21 male and 39 female university students. Participants completed the Psychopathic Personality Inventory (PPI), and the Behavioural Inhibition Scale and the Behavioural Activation Scale (BIS/BAS) as measures of psychopathy. Participants completed a computerised emotion recognition task containing six emotions of facial expressions (each emotion had five different intensities). The results showed that participants scoring low on the BIS and high on the BAS scores showed significant impairments in the recognition of both sad and fearful expressions. On the other hand, group scoring high on the PPI, showed significant impairment in the recognition of angry, but not fearful or sad expressions in the total sample. Males with high psychopathic personality characteristics showed significant impairments in the recognition of sad, fearful and angry expressions compared to males with low psychopathic personality characteristics. On the other hand females with high psychopathic personality characteristics showed significant impairment in recognising the expression of disgust only compared to females with low psychopathic personality characteristics. The PPI and the BIS/BAS scales showed reasonable alpha reliabilities with some exceptions for one subscale in each measure. Correlations between the PPI and the BIS/BAS scales were weak to moderate. The current findings suggest that different dimensions of psychopathy may be associated with selective impairments in recognising unpleasant emotion expressions in others.
            """),
            topic="psychology",
        ),
        TopicTestCase(
            text_doc_id=5123,
            text_char_begin=300,
            text=dedent("""
                Six hens responded under an increasing Fixed Ratio schedule of reinforcement to assess demand separately for two different food types: wheat and puffed wheat. Demand curves generated showed the least preferred food, puffed wheat, yielded a higher initial (ln L) demand than the more preferred food, wheat. While responding for the more preferred food, wheat, produced lower initial (ln L) demand functions, responding for wheat was maintained to higher increasing FR schedules of reinforcement than was that for puffed wheat. This phenomenon occurred across all six hens. To assess preference between the two food types the hens responded under a two-link concurrent-chain schedule of reinforcement. Under the concurrent-chain schedule of reinforcement there were three conditions, each consisted of a initial link with VI 90-s VI 90-s in effect, and terminal links of FR1, FR8 and FR32. The concurrent-chain schedule was used to examine if or how preference may relate to demand. Preference measures obtained showed wheat was generally preferred to puffed wheat across all prices throughout the preference assessment. As price increased in the terminal link during the preference assessment, preference for wheat became more extreme as did the hens responding. The results suggest that while there is a systematic relation between preference and demand, in that at higher FR values food with higher demand levels is preferred. This does not seem to hold, however, at FR1.
            """),
            topic="agriculture",
        ),
        TopicTestCase(
            text_doc_id=7264,
            text_char_begin=400,
            text=dedent("""
                Program slicing is applied to the software maintenance problem by extending the notion of a program slice (that originally required both a variable and line number) to a decomposition slice, one that captures all computation on a given variable, i.e., is independent of line numbers. Using the lattice of single variable decomposition slices ordered by set inclusion, it is shown how a slice-based decomposition for programs can be formed. One can then delineate the effects of a proposed change by isolating those effects in a single component of the decomposition. This gives maintainers a straightforward technique for determining those statements and variables which may be modified in a component and those which may not. Using the decomposition, a set of principles to prohibit changes which will interfere with unmodified components is provided. These semantically consistent changes can then be merged back into the original program in linear time.
            """),
            topic="software development",
        ),
    ]
