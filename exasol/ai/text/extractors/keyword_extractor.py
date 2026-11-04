from typing import Literal

from pydantic import Field

from exasol.ai.text.extractors.base_extractor import NLPExtractor
from exasol.ai.text.extractors.base_model import TextAIBase
from exasol.ai.text.extractors.default_models import DEFAULT_FEATURE_EXTRACTION_MODEL
from exasol.ai.text.extractors.hf_model import HuggingFaceModel


class PatternRankKeywordSettings(TextAIBase):
    """
    A keyword extractor using the KeyphraseVectorizers and the KeyBert packages. The
    algorithm prepares keyphrases based on the part-of-speech tags. The keyphrases
    subsequently get ranked by a KeyBERT model. For details see
    https://github.com/TimSchopf/KeyphraseVectorizers.

    Parameters:
        setting_type:
            This is an earmark of the algorithm or a particular setup for the algorithm
            where the settings are applicable. This field allows switching between
            different algorithms doing a similar job.
        ranking_model:
            The embedding model used for the similarity computation between keyword candidates and the text.
        pos_pattern:
            The regex pattern of POS-tags to extract candidates for the keyphrases.
            Note: That doesn't mean all of these are selected as keywords.
            The default pattern selects keyphrases that are noun or adjective phrases.
            For a more detailed explanation of the default pattern, please check the docstring of the attribute.
        max_df:
            If provided, sets the upper limit for the frequency of the keyphrases in a
            document. Keyphrases with the frequency higher than the given threshold will
            be ignored.
        min_df:
            If provided, sets the lower limit for the frequency of the keyphrases in a
            document. Keyphrases with the frequency lower than the given threshold will
            be ignored. This value is also called cut-off in the literature.
        top_n:
            Sets the number of returned keywords. The default value is 5.
        use_maxsum:
            If True the algorithm will use Max Sum Distance (MSD) for the selection of
            keyphrases. The default is False.
        use_mmr:
            If True the algorithm will use Maximal Marginal Relevance (MMR) for the
            selection of keyphrases.The default is False.
        diversity:
            The diversity of the results between 0 and 1, to use with MMR. Ignored if
            `use_mmr` is False. The default value is 0.5.
        nr_candidates:
            The number of candidates to consider when using MSD. Ignored if `use_maxsum`
            is False. The default value is 20.
    """

    setting_type: Literal["pattern-rank-keyword-settings"] = (
        "pattern-rank-keyword-settings"
    )
    ranking_model: HuggingFaceModel = HuggingFaceModel(
        name=DEFAULT_FEATURE_EXTRACTION_MODEL
    )
    pos_pattern: str = (
        "(<J.*><HYPH.*>?)*((<N.*>|<CD>)<HYPH.*>?)*(<RB.*><HYPH.*>?(<VBN><IN>?)?(<N.*>|<CD>)+|<VBN><IN>?<N.*>+)?((<IN>|<N.*>|<CD>)?<J.*>?)+"
    )
    """
    The POS pattern to extract candidates for the keywords.
    See also: https://keyphrase-vectorizers.readthedocs.io/en/latest/api.html#module-keyphrase_vectorizers.keyphrase_count_vectorizer

    We are using the following default pattern: 
    
    (<J.*><HYPH.*>?)*((<N.*>|<CD>)<HYPH.*>?)*(<RB.*><HYPH.*>?(<VBN><IN>?)?(<N.*>|<CD>)+|<VBN><IN>?<N.*>+)?((<IN>|<N.*>|<CD>)?<J.*>?)+
    .
    It extracts noun phrases and adjective phrases.
    
    1. (<J.*><HYPH.*>?)*
    
        <J.*> matches any adjective tag (e.g., JJ, JJR, JJS).
        <HYPH.*>? matches an optional hyphen tag (if such tags like HYPH are used to indicate a hyphen/punctuation).
        The * outside means this can repeat zero or more times.
        Interpretation: Zero or more sequences of (adjective possibly followed by a hyphen).
    
    2. ((<N.*>|<CD>)<HYPH.*>?)*
    
        <N.*> matches any noun tag (NN, NNS, NNP, etc).
        <CD> matches a cardinal number (like 42).
        <HYPH.*>? is again an optional hyphen.
        The whole group repeats * (zero or more times).
        Interpretation: Zero or more sequences of a noun or number, possibly followed by a hyphen.
    
    3. (<RB.*><HYPH.*>?(<VBN><IN>?)?(<N.*>|<CD>)+|<VBN><IN>?<N.*>+)?
    
        This is wrapped in a ?, so it’s optional.
    
    First alternative:
    
        <RB.*>: Adverb (e.g., RB, RBR, RBS)
        <HYPH.*>?: Optional hyphen.
        (<VBN><IN>?)?: Optional: past participle verb (VBN) possibly followed by preposition/subordinating conjunction (IN).
        (<N.*>|<CD>)+: One or more noun(s) or numbers.
        So,: adverb (maybe hyphen), maybe VBN/IN, then one or more noun/number.
    
    Or:
    
        <VBN><IN>?<N.*>+: Past participle verb, optional preposition, one or more nouns.
    
    4. ((<IN>|<N.*>|<CD>)?<J.*>?)+
    
        <IN>: Preposition or subordinating conjunction.
        <N.*>: Any noun.
        <CD>: Number.
        This part: "maybe a preposition or noun or number", then "optional adjective".
        The whole group repeats + (one or more times).
    
    Summing up
    
        The pattern is designed to match stretches of words that can occur in noun phrases and modifiers, potentially including sequences like:
            (adjective--)* (noun/number--)*... (adverbs, VBN/IN, nouns/numbers, ...), possibly ending with phrases that contain nouns, numbers, adjectives, and/or prepositions.
        Common structures it can match:
            "well-known New York-based financial-services institutions"
            "recently-published long-term economic indicators"
    
    Examples Sentences with POS Tags and phrases detected in brackets:
    
        This/DT was/VBD a/DT [eye/NN -/HYPH opening/NN presentation/NN] ./.
        My/PRP$ [yummy/JJ peanut/NN -/HYPH caramel/NN -/HYPH covered/VBN pizza/NN] is/VBZ [delicious/JJ] ./.
        My/PRP$ [yummy/JJ peanut/NN -/HYPH jelly/RB -/HYPH covered/VBN pizza/NN] is/VBZ [delicious/JJ] ./.
        My/PRP$ [jelly/JJ sandwich/NN] is/VBZ [delicious/JJ] ./.
        My/PRP$ [wi/JJ -/HYPH fi/NN] does/VBZ n't/RB work/VB ./.
        My/PRP$ [wi/JJ -/HYPH fi/NN] is/VBZ broken/VBN ./.
        I/PRP have/VBP [wi/JJ -/HYPH fi/NN problems/NNS] ./.
        I/PRP have/VBP a/DT [slow/JJ wi/NN -/HYPH fi/NN] ./.
        I/PRP think/VBP this/DT is/VBZ a/DT [hardware/NN issue/NN] ./.
        I/PRP covered/VBD our/PRP$ [base/NN] ./.
        The/DT [movie/NN] was/VBD [watched/VBN by/IN millions/NNS] ./.
        The/DT [Samsung/NNP Galaxy/NNP 21/CD] is/VBZ [cool/JJ] ./.
        [Angela/NNP Merkel/NNP] was/VBD [chancelor/NN of/IN Germany/NNP] ./.
    """
    max_df: int | None = None
    min_df: int | None = None
    top_n: int = 5
    use_maxsum: bool = False
    use_mmr: bool = False
    diversity: float = 0.5
    nr_candidates: int = 20


class KeywordBase(TextAIBase):
    # In case of multiple arguments the `keyword_settings` should be a
    # discriminating union.
    keyword_settings: PatternRankKeywordSettings


class KeywordExtractor(NLPExtractor, KeywordBase):
    extractor_type: Literal["KeywordExtractor"] = Field(
        default="KeywordExtractor", repr=False
    )
