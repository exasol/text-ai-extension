import json

from exasol.ai.text.impl.nlp.algorithms.testing import DummyWordCounter
from exasol.ai.text.impl.nlp.algorithms.topics.hft_topic_classification import (
    HftTopicClassification,
)


def test_word_counter_setup():
    algo = DummyWordCounter()
    algo_setup = json.loads(algo.algo_setup)
    expected_setup = {"DummyWordCounter": {}}
    assert algo_setup == expected_setup


def test_topic_classification_setup():
    """
    Tests that the setup json doesn't include the parameters deemed as constant
    and only includes explicitly set parameters.
    """
    algo = HftTopicClassification(
        model_name="my_model",
        bucketfs_conn_name="my_bfs_connection",
        sub_dir="my_model_subdir",
        topics=["Age", "Occupation"],
    )
    algo_setup = json.loads(algo.algo_setup)
    expected_setup = {
        "HftTopicClassification": {
            "model_name": "my_model",
            "topics": ["Age", "Occupation"],
        }
    }
    assert algo_setup == expected_setup
