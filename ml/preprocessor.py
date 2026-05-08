import re

# Negation words are intentionally excluded from stopwords so they can
# be used to flip the polarity of the tokens that follow them.
_EN_NEGATION_WORDS = frozenset({
    'not', 'no', 'nor', 'never', 'neither', 'nothing', 'nobody',
    'nowhere', 'hardly', 'scarcely', 'barely', 'without',
    # contracted forms after apostrophe-stripping (clean() removes apostrophes)
    'dont', 'cant', 'wont', 'isnt', 'arent', 'wasnt', 'werent',
    'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt', 'wouldnt',
    'shouldnt', 'couldnt', 'neednt', 'mightnt', 'mustnt',
})

ENGLISH_STOPWORDS = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'ma',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
    'wasn', 'weren', 'won', 'wouldn',
    # note: 'no', 'not', 'nor' intentionally absent — they are negation markers
}) - _EN_NEGATION_WORDS  # safety: guarantee no overlap

PERSIAN_STOPWORDS = frozenset({
    'و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'بر',
    'آن', 'یک', 'یا', 'هم', 'اما', 'تا', 'برای', 'می', 'شد', 'کرد',
    'نیز', 'هر', 'خود', 'اگر', 'بود', 'کند', 'شود', 'داد', 'دارد',
    'چه', 'ها', 'های', 'ام', 'ند', 'اند', 'شده', 'کنم', 'کنیم',
    'دارم', 'داریم', 'بودم', 'بودیم', 'باید', 'باشد', 'باشم',
})

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_MULTI_SPACE_RE = re.compile(r'\s+')
_ENGLISH_KEEP_RE = re.compile(r"[^a-zA-Z\s]")
_PERSIAN_KEEP_RE = re.compile(r'[^\u0600-\u06FF\s]')

_NEGATION_WINDOW = 3


def _apply_negation(tokens: list[str]) -> list[str]:
    """
    Prefix tokens that follow a negation word with 'NOT_'.

    Example: ["the", "movie", "was", "not", "good", "at", "all"]
          -> ["the", "movie", "was", "not", "NOT_good", "NOT_at", "NOT_all"]

    A sliding window of _NEGATION_WINDOW tokens limits the negation scope.
    The negation word itself is kept so its own frequency is also a feature.
    """
    result = []
    countdown = 0
    for token in tokens:
        if token in _EN_NEGATION_WORDS:
            result.append(token)
            countdown = _NEGATION_WINDOW
        elif countdown > 0:
            result.append(f'NOT_{token}')
            countdown -= 1
        else:
            result.append(token)
    return result


class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        self.language = language
        self._stopwords = PERSIAN_STOPWORDS if language == 'persian' else ENGLISH_STOPWORDS

    def clean(self, text: str) -> str:
        text = _HTML_RE.sub(' ', text)
        text = _URL_RE.sub(' ', text)
        if self.language == 'english':
            text = text.lower()
            text = _ENGLISH_KEEP_RE.sub(' ', text)
        else:
            text = _PERSIAN_KEEP_RE.sub(' ', text)
        return _MULTI_SPACE_RE.sub(' ', text).strip()

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def process(self, text: str) -> list[str]:
        tokens = self.tokenize(self.clean(text))
        # Strip function words first so negation only propagates to content words.
        # Example: "not bad at all" → filter → ["not","bad"] → negate → ["not","NOT_bad"]
        # Without this order: ["not","NOT_bad","NOT_at","NOT_all"] — pure noise.
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]
        if self.language == 'english':
            tokens = _apply_negation(tokens)
        return tokens

    def process_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.process(t) for t in texts]
