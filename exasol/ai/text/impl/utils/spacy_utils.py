import spacy

from exasol.ai.text.impl.utils.constants import DEFAULT_SPACY_PIPELINE


def get_spacy_model(
    pipeline: str = DEFAULT_SPACY_PIPELINE, exclude: list[str] | None = None
):
    """
    Loads a spaCy language processing pipline.

    Parameters:
        pipeline:
            Pipeline name.
        exclude:
            The list of components to exclude.
            See https://spacy.io/usage/processing-pipelines
    """
    exclude = exclude or []
    return spacy.load(pipeline, exclude=exclude)
