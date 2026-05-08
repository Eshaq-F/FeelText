import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.core.factory import get_analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = 'custom' if settings.USE_CUSTOM_MODEL else 'third-party'
    logger.info('Starting FeelText API  [mode: %s] ...', mode)
    analyzer = get_analyzer()
    analyzer.load_model()
    logger.info('FeelText API ready.')
    yield
    logger.info('Shutting down FeelText API.')


app = FastAPI(
    title='FeelText',
    description='Multilingual Sentiment Analysis API',
    version='3.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)


@app.get('/')
async def root():
    mode = 'custom' if settings.USE_CUSTOM_MODEL else 'third-party'
    return {
        'name': 'FeelText',
        'description': 'Multilingual Sentiment Analysis API',
        'version': '3.0.0',
        'mode': mode,
        'docs': '/docs',
        'endpoints': {
            'analyze': '/analyze',
            'batch_analyze': '/analyze/batch',
            'languages': '/languages',
            'health': '/health',
        },
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
