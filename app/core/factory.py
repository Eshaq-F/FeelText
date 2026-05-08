import logging
from functools import lru_cache

from app.config import settings
from app.core.base import BaseAnalyzer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_analyzer() -> BaseAnalyzer:
    if settings.USE_CUSTOM_MODEL:
        logger.info('Analyzer mode: custom (TF-IDF + Naive Bayes)')
        from app.core.custom import CustomAnalyzer
        return CustomAnalyzer()

    logger.info('Analyzer mode: third-party (HuggingFace transformers)')
    from app.core.third_party import ThirdPartyAnalyzer
    return ThirdPartyAnalyzer()
