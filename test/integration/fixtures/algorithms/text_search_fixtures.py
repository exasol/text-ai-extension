# pylint: disable=line-too-long
from dataclasses import dataclass
from textwrap import dedent

import pytest

from test.integration.utils.text_test_case import TextTestCase


@dataclass
class PhraseOccurrences:
    phrase: str
    occurrences: list[int]


@dataclass
class TextSearchTestCase(TextTestCase):
    phrases_occurrences: list[PhraseOccurrences]


@pytest.fixture
def text_search_test_cases() -> list[TextSearchTestCase]:
    return [
        TextSearchTestCase(
            text_doc_id=1234,
            text_char_begin=200,
            text=dedent("""
                The forest was quiet, with only the sound of rustling leaves. The sound
                of rustling leaves filled the air, blending with the occasional chirp
                of unseen birds. As Sarah walked deeper into the forest, the sound of
                rustling leaves followed her like a gentle melody.
                She paused for a moment, listening to the sound of rustling leaves.
                "It's peaceful here," she thought, but the sound of rustling leaves
                seemed to grow louder as she moved forward.
                The deeper she went, the more the sound of rustling leaves dominated
                her senses. It was no longer just a sound—it was a presence, a rhythm,
                a constant reminder of the forest's life.
            """),
            phrases_occurrences=[
                PhraseOccurrences(
                    phrase="sound of rustling leaves",
                    occurrences=[37, 67, 204, 306, 375, 478],
                )
            ],
        ),
        TextSearchTestCase(
            text_doc_id=5678,
            text_char_begin=400,
            text=dedent("""
                The rain began as a soft drizzle, the kind of drizzle that whispers
                secrets to the earth. The kind of drizzle that makes the air thick with
                anticipation. Under the umbrella of gray clouds, Lily walked slowly, her
                boots splashing through shallow puddles. The sound of splashing puddles
                echoed softly, a rhythm matching the drizzle that whispered to the
                earth.
                She pulled her coat tighter, the wind carrying the scent of wet earth.
                The scent of wet earth was intoxicating, grounding her in the present
                moment. It was the scent of wet earth that reminded her of childhood
                afternoons spent in the garden, her hands covered in mud.
                The drizzle that whispered secrets to the earth soon turned into a
                steady rain. A steady rain that blurred the edges of the world, turning
                familiar streets into dreamy reflections. The sound of splashing puddles
                grew louder, louder still as the rain danced on rooftops and slid down
                windows.
                As Lily reached the corner of the park, she stopped, letting the scent
                of wet earth surround her. She thought of the days when rain felt like
                freedom, when the drizzle that whispered secrets was a call to
                adventure. The steady rain now felt different—less of a call, more of a
                song. A song that blended with the sound of splashing puddles, with the
                scent of wet earth, with memories that refused to fade.
            """),
            phrases_occurrences=[
                PhraseOccurrences(
                    phrase="sound of splashing puddles", occurrences=[259, 813, 1232]
                ),
                PhraseOccurrences(
                    phrase="scent of wet earth", occurrences=[411, 435, 520, 985, 1269]
                ),
                PhraseOccurrences(phrase="whispered secrets", occurrences=[645, 1093]),
                PhraseOccurrences(phrase="steady rain", occurrences=[695, 710, 1140]),
            ],
        ),
    ]
