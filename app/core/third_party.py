import logging
import re

from app.config import settings
from app.core.base import BaseAnalyzer

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_PERSIAN_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
)

_SUPPORTED_LANGUAGES = [
    'Persian (Farsi)', 'English', 'Arabic', 'French', 'German',
    'Spanish', 'Italian', 'Portuguese', 'Dutch', 'Russian',
    'Chinese', 'Japanese', 'Korean', 'Turkish', 'Hindi',
    'Indonesian', 'Vietnamese', 'Thai', 'Polish', 'Ukrainian',
]

_PERSIAN_LABEL_MAP = {
    'recommended': 'positive',
    'not_recommended': 'negative',
    'label_0': 'negative',
    'label_1': 'positive',
    'negative': 'negative',
    'positive': 'positive',
    'neutral': 'neutral',
}

_MAX_TEXT_LENGTH = 512


def _is_persian(text: str) -> bool:
    persian = len(_PERSIAN_RE.findall(text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    total = persian + latin
    return total > 0 and (persian / total) > 0.3


class ThirdPartyAnalyzer(BaseAnalyzer):
    """
    Sentiment analyzer backed by pre-trained HuggingFace transformer models.
    Enabled when USE_CUSTOM_MODEL=false (the default).

    Persian  →  HooshvareLab/bert-fa-base-uncased-sentiment-digikala
    Others   →  lxyuan/distilbert-base-multilingual-cased-sentiments-student
    """

    def __init__(self):
        self._persian_pipeline = None
        self._multilingual_pipeline = None
        self._persian_loaded: bool = False
        self._multilingual_loaded: bool = False

    @property
    def model_name(self) -> str:
        return (
            f'Persian: {settings.PERSIAN_MODEL} | '
            f'Multilingual: {settings.MULTILINGUAL_MODEL}'
        )

    @property
    def is_loaded(self) -> bool:
        return self._persian_loaded and self._multilingual_loaded

    @property
    def supported_languages(self) -> list[str]:
        return _SUPPORTED_LANGUAGES

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _require_transformers(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                'transformers and torch are required for third-party mode. '
                'Install them with:\n'
                '  pip install transformers torch sentencepiece accelerate'
            )

    def _load_persian(self) -> None:
        if self._persian_loaded:
            return
        self._require_transformers()
        logger.info('Loading Persian model: %s', settings.PERSIAN_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(settings.PERSIAN_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(settings.PERSIAN_MODEL)
        self._persian_pipeline = hf_pipeline(
            'sentiment-analysis',
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        self._persian_loaded = True
        logger.info('Persian model loaded.')

    def _load_multilingual(self) -> None:
        if self._multilingual_loaded:
            return
        self._require_transformers()
        logger.info('Loading multilingual model: %s', settings.MULTILINGUAL_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(settings.MULTILINGUAL_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(settings.MULTILINGUAL_MODEL)
        self._multilingual_pipeline = hf_pipeline(
            'sentiment-analysis',
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        self._multilingual_loaded = True
        logger.info('Multilingual model loaded.')

    def load_model(self) -> None:
        self._load_persian()
        self._load_multilingual()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _analyze_persian(self, text: str) -> dict:
        if not self._persian_loaded:
            self._load_persian()

        raw = self._persian_pipeline(text)[0]
        scores: dict[str, float] = {}
        for item in raw:
            label = _PERSIAN_LABEL_MAP.get(item['label'].lower(), 'neutral')
            scores[label] = item['score']

        scores.setdefault('positive', 0.0)
        scores.setdefault('negative', 0.0)
        if 'neutral' not in scores:
            top = max(scores.get('positive', 0), scores.get('negative', 0))
            scores['neutral'] = 1.0 - top if top < 0.6 else 0.0

        best = max(scores, key=scores.get)
        return {
            'sentiment': best,
            'confidence': round(scores[best], 4),
            'scores': {k: round(v, 4) for k, v in scores.items()},
            'language_detected': 'persian',
        }

    def _analyze_multilingual(self, text: str) -> dict:
        if not self._multilingual_loaded:
            self._load_multilingual()

        raw = self._multilingual_pipeline(text)[0]
        scores = {item['label'].lower(): item['score'] for item in raw}
        best = max(scores, key=scores.get)
        return {
            'sentiment': best,
            'confidence': round(scores[best], 4),
            'scores': {k: round(v, 4) for k, v in scores.items()},
            'language_detected': 'other',
        }

    def analyze(self, text: str) -> dict:
        truncated = text[:_MAX_TEXT_LENGTH]
        result = (
            self._analyze_persian(truncated)
            if _is_persian(truncated)
            else self._analyze_multilingual(truncated)
        )
        result['text'] = text
        return result
