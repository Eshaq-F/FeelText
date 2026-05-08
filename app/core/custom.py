import logging
import re
from pathlib import Path

from app.config import settings
from app.core.base import BaseAnalyzer
from ml.pipeline import SentimentPipeline

logger = logging.getLogger(__name__)

_PERSIAN_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
)

_SUPPORTED_LANGUAGES = ['English', 'Persian (Farsi)']
_MAX_TEXT_LENGTH = 1000


def _is_persian(text: str) -> bool:
    persian = len(_PERSIAN_RE.findall(text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    total = persian + latin
    return total > 0 and (persian / total) > 0.3


class CustomAnalyzer(BaseAnalyzer):
    """
    Sentiment analyzer backed by a custom-trained TF-IDF + Naive Bayes model.
    Enabled when USE_CUSTOM_MODEL=true.
    """

    def __init__(self):
        self._pipeline: SentimentPipeline | None = None
        self._loaded: bool = False

    @property
    def model_name(self) -> str:
        return 'Custom TF-IDF + Multinomial Naive Bayes (trained from scratch)'

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supported_languages(self) -> list[str]:
        return _SUPPORTED_LANGUAGES

    def load_model(self) -> None:
        if self._loaded:
            return

        model_path = Path(settings.CUSTOM_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f'Trained model not found at "{model_path}". '
                'Run  python scripts/train.py  to train and save the model first.'
            )

        logger.info('Loading custom model from %s', model_path)
        self._pipeline = SentimentPipeline.load(str(model_path))
        self._loaded = True
        logger.info('Custom model loaded.')

    def analyze(self, text: str) -> dict:
        if not self._loaded:
            self.load_model()

        truncated = text[:_MAX_TEXT_LENGTH]
        language = 'persian' if _is_persian(truncated) else 'english'
        result = self._pipeline.predict(truncated, language=language)
        result['text'] = text
        result['language_detected'] = language
        return result
