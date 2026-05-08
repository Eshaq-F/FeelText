import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    BatchSentimentResult,
    BatchTextInput,
    HealthResponse,
    LanguagesResponse,
    SentimentResult,
    TextInput,
)
from app.config import settings
from app.core.factory import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health_check():
    analyzer = get_analyzer()
    return HealthResponse(
        status='healthy',
        model_loaded=analyzer.is_loaded,
        model_name=analyzer.model_name,
        mode='custom' if settings.USE_CUSTOM_MODEL else 'third-party',
    )


@router.get('/languages', response_model=LanguagesResponse)
async def get_supported_languages():
    analyzer = get_analyzer()
    languages = analyzer.supported_languages
    return LanguagesResponse(
        languages=languages,
        total=len(languages),
        note=(
            'Custom model supports English and Persian.'
            if settings.USE_CUSTOM_MODEL
            else 'Third-party model supports 100+ languages.'
        ),
    )


@router.post('/analyze', response_model=SentimentResult)
async def analyze_sentiment(body: TextInput):
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze(body.text)
        return SentimentResult(**result)
    except Exception as exc:
        logger.error('Error analyzing sentiment: %s', exc)
        raise HTTPException(status_code=500, detail=f'Analysis failed: {exc}') from exc


@router.post('/analyze/batch', response_model=BatchSentimentResult)
async def analyze_batch(body: BatchTextInput):
    try:
        analyzer = get_analyzer()
        results = analyzer.analyze_batch(body.texts)
        return BatchSentimentResult(
            results=[SentimentResult(**r) for r in results],
            total_analyzed=len(results),
        )
    except Exception as exc:
        logger.error('Error in batch analysis: %s', exc)
        raise HTTPException(status_code=500, detail=f'Batch analysis failed: {exc}') from exc
