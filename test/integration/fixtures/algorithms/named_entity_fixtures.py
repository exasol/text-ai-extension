# pylint: disable=line-too-long
from collections.abc import Generator
from dataclasses import dataclass
from textwrap import dedent

import exasol.bucketfs as bfs
import pytest
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from transformers import AutoModelForTokenClassification

from test.integration.utils.model_utils import (
    download_model_locally,
    download_model_to_bucketfs,
)
from test.integration.utils.text_test_case import TextTestCase


@pytest.fixture(scope="session")
def named_entity_model():
    return "sschet/biomedical-ner-all"


@pytest.fixture(scope="session")
def named_entity_model_revision():
    return "ecf82ed8e29281e14da9e72a5e3a729fce2c73a3"


@pytest.fixture(scope="session")
def named_entity_model_location(
    tmp_path_factory, named_entity_model, named_entity_model_revision, model_subdir
) -> bfs.path.PathLike:
    model_specification = ModelSpecification(named_entity_model, "token-classification")
    return download_model_locally(
        model_specification,
        named_entity_model_revision,
        tmp_path_factory,
        model_subdir,
        AutoModelForTokenClassification,
    )


@pytest.fixture(scope="session")
def named_entity_model_download(
    named_entity_model, model_subdir, backend_aware_bucketfs_params
) -> Generator[None, None, None]:
    """
    Downloads a token classification model into the BucketFS, to be used from a UDF.
    """
    with download_model_to_bucketfs(
        named_entity_model,
        model_subdir,
        "token-classification",
        backend_aware_bucketfs_params,
        AutoModelForTokenClassification,
    ):
        yield


@dataclass
class RecognizedNamedEntity:
    entity_type: str
    entity_text: str
    position: int


@dataclass
class NamedEntitiesWithStrategies:
    strategies: list[str]
    entities: list[RecognizedNamedEntity]


@dataclass
class NamedEntityTestCase(TextTestCase):
    expected_entities: list[NamedEntitiesWithStrategies]


@pytest.fixture
def named_entity_test_cases() -> list[NamedEntityTestCase]:
    return [
        NamedEntityTestCase(
            text_doc_id=4568,
            text_char_begin=200,
            text=dedent("""
                Presenting Complaint:
                The patient was brought to the Emergency Department (ED) following an out-of-hospital cardiac arrest (OHCA). Bystanders reported sudden collapse with loss of consciousness and absence of pulse. Cardiopulmonary resuscitation (CPR) was initiated immediately by abystander, and automated external defibrillator (AED) shocks were administered by emergency medical services (EMS) en route to the hospital.

                History of Present Illness:
                The patient had no prior complaints of chest pain, palpitations, or shortness of breath prior to the event. According to the family, he was in his usual state of health until the sudden collapse. No recent history of fever, cough, or other systemic symptoms. No known trauma or drug overdose.

                Past Medical History:
                1. Hypertension: Diagnosed 10 years ago, on antihypertensive medications (lisinopril 10 mg daily).
                2. Hyperlipidemia: Diagnosed 8 years ago, on atorvastatin 20 mg daily.
                3. Type 2 Diabetes Mellitus: Diagnosed 5 years ago, managed with metformin 500 mg twice daily.
                4. Coronary Artery Disease (CAD): History of myocardial infarction (MI) 3 years ago, treated with percutaneous coronary intervention (PCI) and drug-eluting stent placement in the left anterior descending (LAD) artery.
                5. Obesity: Body mass index (BMI) of 32 kg/m².
                6. Smoking History: 20 pack-year history, quit smoking 5 years ago.
                7. Family History: Father died of myocardial infarction at age 62; mother has hypertension and diabetes.

                Medications:
                - Lisinopril 10 mg daily
                - Atorvastatin 20 mg daily
                - Metformin 500 mg twice daily
                - Aspirin 81 mg daily
                - Clopidogrel 75 mg daily (post-PCI)

                Allergies:
                - No known drug allergies.
            """),
            expected_entities=[
                NamedEntitiesWithStrategies(
                    ["simple", "max"],
                    [
                        RecognizedNamedEntity(
                            "Nonbiological_location", "emergency department", 54
                        ),
                        RecognizedNamedEntity(
                            "Nonbiological_location", "hospital", 100
                        ),
                        RecognizedNamedEntity(
                            "Disease_disorder", "cardiac arrest", 109
                        ),
                        RecognizedNamedEntity("Sign_symptom", "sudden collapse", 152),
                        RecognizedNamedEntity(
                            "Sign_symptom", "loss of consciousness", 173
                        ),
                        RecognizedNamedEntity(
                            "Therapeutic_procedure",
                            "cardiopulmonary resuscitation",
                            217,
                        ),
                        RecognizedNamedEntity(
                            "Sign_symptom", "shortness of breath", 521
                        ),
                        RecognizedNamedEntity("Dosage", "10 mg", 854),
                        RecognizedNamedEntity(
                            "Disease_disorder", "hyperlipidemia", 871
                        ),
                        RecognizedNamedEntity("Medication", "metformin", 1004),
                        RecognizedNamedEntity("Dosage", "500 mg twice daily", 1014),
                        RecognizedNamedEntity(
                            "Biological_structure", "left anterior", 1213
                        ),
                        RecognizedNamedEntity("Subject", "father", 1386),
                        RecognizedNamedEntity("Outcome", "died", 1393),
                        RecognizedNamedEntity("Subject", "mother", 1434),
                        RecognizedNamedEntity("Dosage", "10 mg daily", 1499),
                        RecognizedNamedEntity("Dosage", "20 mg daily", 1526),
                        RecognizedNamedEntity("Dosage", "500 mg", 1550),
                        RecognizedNamedEntity("Dosage", "twice daily", 1557),
                        RecognizedNamedEntity("Dosage", "81 mg daily", 1579),
                        RecognizedNamedEntity("Dosage", "75 mg daily", 1605),
                    ],
                ),
                NamedEntitiesWithStrategies(
                    ["none"],
                    [
                        RecognizedNamedEntity(
                            "B-Nonbiological_location", "emergency", 54
                        ),
                        RecognizedNamedEntity(
                            "I-Nonbiological_location", "department", 64
                        ),
                        RecognizedNamedEntity(
                            "I-Nonbiological_location", "hospital", 100
                        ),
                        RecognizedNamedEntity("B-Disease_disorder", "cardiac", 109),
                        RecognizedNamedEntity("I-Disease_disorder", "arrest", 117),
                        RecognizedNamedEntity("B-Sign_symptom", "sudden", 152),
                        RecognizedNamedEntity("B-Sign_symptom", "loss", 173),
                        RecognizedNamedEntity("I-Sign_symptom", "of", 178),
                        RecognizedNamedEntity("I-Sign_symptom", "consciousness", 181),
                        RecognizedNamedEntity(
                            "I-Therapeutic_procedure", "medical", 375
                        ),
                        RecognizedNamedEntity(
                            "I-Therapeutic_procedure", "services", 383
                        ),
                        RecognizedNamedEntity("B-Dosage", "20", 926),
                        RecognizedNamedEntity("I-Dosage", "mg", 929),
                        RecognizedNamedEntity("I-Dosage", "daily", 932),
                        RecognizedNamedEntity("B-Biological_structure", "left", 1213),
                        RecognizedNamedEntity(
                            "I-Biological_structure", "anterior", 1218
                        ),
                    ],
                ),
            ],
        ),
        NamedEntityTestCase(
            text_doc_id=8745,
            text_char_begin=300,
            text=dedent("""
                Presenting Complaint:
                The patient was brought to the clinic by her daughter due to progressive memory loss, confusion, and difficulty performing daily activities over the past 2 years. The family reports increasing episodes of disorientation, repetitive questioning, and occasional agitation.

                History of Present Illness:
                The patient’s symptoms began approximately 2 years ago with mild forgetfulness, such as misplacing items and forgetting appointments. Over time, her memory deficits worsened, and she began to struggle with tasks like cooking, managing finances, and recognizing familiar faces. She has become increasingly dependent on her family for daily activities.

                Recent Changes:
                - Increased confusion, especially in the evenings (sundowning).
                - Difficulty recalling recent events, though long-term memory remains relatively intact.
                - Occasional episodes of agitation and irritability, particularly when routines are disrupted.
                - No history of falls, seizures, or focal neurological deficits.

                Past Medical History:
                1. Hypertension: Diagnosed 15 years ago, on amlodipine 5 mg daily.
                2. Hyperlipidemia: Diagnosed 10 years ago, on atorvastatin 20 mg daily.
                3. Type 2 Diabetes Mellitus: Diagnosed 12 years ago, managed with metformin 500 mg twice daily.
                4. Osteoarthritis: Chronic knee pain, managed with acetaminophen as needed.
                5. Depression: Treated with sertraline 50 mg daily for the past 5 years.

                Medications:
                - Amlodipine 5 mg daily
                - Atorvastatin 20 mg daily
                - Metformin 500 mg twice daily
                - Sertraline 50 mg daily
                - Acetaminophen 500 mg as needed for pain

                Allergies:
                - No known drug allergies.

                Family History:
                - Mother had Alzheimer’s dementia, diagnosed at age 78.
                - Father had a history of stroke at age 75.
                - No family history of Parkinson’s disease or other neurodegenerative disorders.
            """),
            expected_entities=[
                NamedEntitiesWithStrategies(
                    ["simple", "max"],
                    [
                        RecognizedNamedEntity("Nonbiological_location", "clinic", 54),
                        RecognizedNamedEntity("Subject", "daughter", 68),
                        RecognizedNamedEntity("Sign_symptom", "agitation", 869),
                        RecognizedNamedEntity("Dosage", "5 mg daily", 1082),
                        RecognizedNamedEntity(
                            "Disease_disorder", "hyperlipidemia", 1097
                        ),
                        RecognizedNamedEntity("Dosage", "20 mg daily", 1153),
                        RecognizedNamedEntity(
                            "Disease_disorder", "type 2 diabetes mellitus", 1169
                        ),
                        RecognizedNamedEntity("Medication", "metformin", 1232),
                        RecognizedNamedEntity("Dosage", "500 mg", 1242),
                        RecognizedNamedEntity("Dosage", "twice daily", 1249),
                        RecognizedNamedEntity("Biological_structure", "knee", 1289),
                        RecognizedNamedEntity("Medication", "acetaminophen", 1313),
                        RecognizedNamedEntity("Medication", "sertraline", 1366),
                        RecognizedNamedEntity("Dosage", "50 mg daily", 1377),
                        RecognizedNamedEntity("Dosage", "5 mg daily", 1438),
                        RecognizedNamedEntity("Dosage", "20 mg daily", 1464),
                        RecognizedNamedEntity("Medication", "metformin", 1478),
                        RecognizedNamedEntity("Dosage", "500 mg", 1488),
                        RecognizedNamedEntity("Dosage", "twice daily", 1495),
                        RecognizedNamedEntity("Medication", "sertraline", 1509),
                        RecognizedNamedEntity("Dosage", "50 mg daily", 1520),
                        RecognizedNamedEntity("Medication", "acetaminophen", 1534),
                        RecognizedNamedEntity("Dosage", "500 mg", 1548),
                        RecognizedNamedEntity("Sign_symptom", "pain", 1569),
                        RecognizedNamedEntity("Subject", "father", 1688),
                    ],
                ),
                NamedEntitiesWithStrategies(
                    ["none"],
                    [
                        RecognizedNamedEntity("B-Nonbiological_location", "clinic", 54),
                        RecognizedNamedEntity("B-Subject", "daughter", 68),
                        RecognizedNamedEntity("B-Sign_symptom", "memory", 472),
                        RecognizedNamedEntity("I-Sign_symptom", "deficit", 479),
                        RecognizedNamedEntity("B-Sign_symptom", "agitation", 869),
                        RecognizedNamedEntity("B-Dosage", "5", 1082),
                        RecognizedNamedEntity("I-Dosage", "mg", 1084),
                        RecognizedNamedEntity("I-Dosage", "daily", 1087),
                        RecognizedNamedEntity("B-Disease_disorder", "type", 1169),
                        RecognizedNamedEntity("I-Disease_disorder", "diabetes", 1176),
                        RecognizedNamedEntity("B-Dosage", "500", 1242),
                        RecognizedNamedEntity("I-Dosage", "mg", 1246),
                        RecognizedNamedEntity("I-Dosage", "daily", 1255),
                        RecognizedNamedEntity("B-Biological_structure", "knee", 1289),
                        RecognizedNamedEntity("B-Sign_symptom", "pain", 1569),
                        RecognizedNamedEntity("B-Subject", "father", 1688),
                    ],
                ),
            ],
        ),
        NamedEntityTestCase(
            text_doc_id=6666,
            text_char_begin=400,
            text="This is a test",  # deliberatly not using any named entity here
            expected_entities=[
                NamedEntitiesWithStrategies(
                    ["simple", "max", "none"],
                    [],
                ),
            ],
        ),
    ]


@pytest.fixture
def named_entity_ignore_labels(named_entity_test_cases) -> dict[str, list[str]]:
    """
    For each strategy construct a list of the labels to ignore. It should consist of
    approximately half of all distinct labels expected in all test cases when using
    this aggregation strategy.
    """
    common_labels: dict[str, set[str]] = {}
    for text_case in named_entity_test_cases:
        for strategy_entities in text_case.expected_entities:
            case_labels = {
                expected_entity.entity_type
                for expected_entity in strategy_entities.entities
            }
            for strategy in strategy_entities.strategies:
                if strategy in common_labels:
                    common_labels[strategy] = case_labels.intersection(
                        common_labels[strategy]
                    )
                else:
                    common_labels[strategy] = case_labels
    return {
        strategy: sorted(labels)[: len(labels) // 2]
        for strategy, labels in common_labels.items()
    }
