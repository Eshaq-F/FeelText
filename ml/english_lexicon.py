"""
English sentiment lexicon scorer.

Designed to work on tokens that have already been preprocessed by
TextPreprocessor (i.e. lowercased, stopwords removed, negation-prefixed).
The NOT_ prefix produced by _apply_negation() is handled here to flip
the polarity of negated sentiment words automatically.

Used as a semantic complement to TF-IDF: it provides reliable signal for
short texts where the statistical model has few vocabulary matches.
"""

_POSITIVE_WORDS = frozenset({
    # Strong positive
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'awesome', 'love', 'loved', 'adore', 'adored', 'brilliant', 'perfect',
    'outstanding', 'superb', 'magnificent', 'exceptional', 'extraordinary',
    'masterpiece', 'flawless', 'breathtaking', 'phenomenal', 'spectacular',
    # Moderate positive
    'like', 'liked', 'enjoy', 'enjoyed', 'enjoyable', 'pleasant', 'nice',
    'fine', 'solid', 'decent', 'okay', 'ok', 'alright', 'fair', 'good',
    'fun', 'funny', 'hilarious', 'entertaining', 'engaging', 'captivating',
    'compelling', 'fascinating', 'interesting', 'charming', 'delightful',
    'heartwarming', 'touching', 'moving', 'powerful', 'beautiful', 'gorgeous',
    'stunning', 'thrilling', 'exciting', 'impressive', 'clever', 'witty',
    # Emotion / outcome
    'happy', 'glad', 'pleased', 'satisfied', 'joy', 'joyful', 'cheerful',
    'delight', 'delighted', 'thrilled', 'excited', 'enthusiastic', 'proud',
    'hope', 'hopeful', 'optimistic', 'inspired', 'inspiring', 'uplifting',
    'recommend', 'recommended', 'worth', 'worthy', 'valuable', 'rewarding',
    'success', 'successful', 'triumph', 'win', 'winner', 'winning', 'best',
    'top', 'favorite', 'favourite', 'gem', 'treasure', 'refreshing',
    'creative', 'original', 'unique', 'innovative', 'authentic', 'genuine',
    'smart', 'intelligent', 'skillful', 'talented', 'gifted', 'superb',
})

_NEGATIVE_WORDS = frozenset({
    # Strong negative
    'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'atrocious',
    'appalling', 'abysmal', 'disgusting', 'revolting', 'hideous', 'vile',
    'hate', 'hated', 'despise', 'loathe', 'detest', 'abhor',
    'worst', 'pathetic', 'garbage', 'trash', 'rubbish', 'junk',
    # Moderate negative
    'boring', 'bored', 'dull', 'bland', 'flat', 'tedious', 'monotonous',
    'disappointing', 'disappointed', 'disappoints', 'poor', 'weak', 'mediocre',
    'fail', 'failed', 'failure', 'flop', 'disaster', 'mess', 'fiasco',
    'stupid', 'dumb', 'silly', 'ridiculous', 'absurd', 'nonsense',
    'pointless', 'useless', 'worthless', 'meaningless', 'empty', 'shallow',
    'predictable', 'cliche', 'generic', 'derivative', 'unoriginal',
    'annoying', 'irritating', 'frustrating', 'infuriating', 'unbearable',
    'ugly', 'offensive', 'disturbing', 'uncomfortable', 'unpleasant',
    'slow', 'drag', 'dragging', 'confusing', 'incoherent', 'implausible',
    'overrated', 'pretentious', 'forced', 'fake', 'contrived',
    # Emotion / outcome
    'sad', 'unhappy', 'miserable', 'depressing', 'hopeless', 'desperate',
    'angry', 'furious', 'outraged', 'bitter', 'resentful', 'regret',
    'waste', 'wasted', 'painful', 'suffer', 'suffering', 'agony',
    'cheap', 'sloppy', 'lazy', 'amateur', 'incompetent',
})

_INTENSIFIERS: dict[str, float] = {
    'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
    'absolutely': 2.0, 'highly': 1.5, 'utterly': 1.8, 'completely': 1.8,
    'totally': 1.6, 'deeply': 1.6, 'truly': 1.4, 'so': 1.3,
    'super': 1.5, 'remarkably': 1.7, 'exceptionally': 1.8,
    'genuinely': 1.4, 'definitely': 1.4, 'clearly': 1.3,
    'undeniably': 1.6, 'outrageously': 1.8, 'insanely': 1.8,
}

_DIMINISHERS: dict[str, float] = {
    'somewhat': 0.6, 'slightly': 0.5, 'fairly': 0.7, 'almost': 0.6,
    'barely': 0.4, 'hardly': 0.4, 'little': 0.5, 'kind': 0.6,
    'sort': 0.6, 'rather': 0.8, 'enough': 0.8, 'quite': 0.9,
}

# Negation words themselves carry no lexicon sentiment value
_SKIP_TOKENS = frozenset({
    'not', 'no', 'nor', 'never', 'dont', 'cant', 'wont', 'isnt', 'arent',
    'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt',
    'wouldnt', 'shouldnt', 'couldnt', 'neednt', 'mightnt', 'mustnt',
    'neither', 'nothing', 'nobody', 'nowhere', 'hardly', 'scarcely',
    'barely', 'without',
})


class EnglishLexiconScorer:
    """
    Scores a list of pre-processed tokens using the sentiment lexicon.

    Handles:
      - NOT_ prefix (from negation preprocessing): flips polarity.
      - Intensifiers preceding a sentiment token: boost weight.
      - Diminishers preceding a sentiment token: reduce weight.

    Returns None when no sentiment words are found (caller falls back
    to the statistical model).
    """

    def score(self, tokens: list[str]) -> dict | None:
        pos = neg = 0.0
        multiplier = 1.0

        for token in tokens:
            if token in _SKIP_TOKENS:
                continue

            # Carry the multiplier from the previous token
            mult = multiplier
            multiplier = 1.0  # reset for next token

            if token in _INTENSIFIERS:
                multiplier = _INTENSIFIERS[token]
                continue
            if token in _DIMINISHERS:
                multiplier = _DIMINISHERS[token]
                continue

            negated = token.startswith('NOT_')
            base = token[4:] if negated else token

            if base in _POSITIVE_WORDS:
                if negated:
                    neg += mult
                else:
                    pos += mult
            elif base in _NEGATIVE_WORDS:
                if negated:
                    pos += mult
                else:
                    neg += mult

        total = pos + neg
        if total == 0.0:
            return None

        pos_ratio = pos / total
        neg_ratio = neg / total
        return {
            'pos_score': pos_ratio,
            'neg_score': neg_ratio,
            'n_matches': total,
        }
