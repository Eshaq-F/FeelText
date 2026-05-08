import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.core.factory import get_analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

_STATIC = Path('static')


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = 'custom' if settings.USE_CUSTOM_MODEL else 'third-party'
    logger.info('Starting FeelText API  [mode: %s] ...', mode)
    get_analyzer().load_model()
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


@app.get('/', include_in_schema=False)
async def serve_ui():
    index = _STATIC / 'index.html'
    if index.exists():
        return FileResponse(str(index), media_type='text/html')
    # Fallback JSON when static files are absent
    return {
        'name': 'FeelText',
        'version': '3.0.0',
        'docs': '/docs',
        'ui': 'Static files not found. Place them in the static/ directory.',
    }


# Mount after all routes so the wildcard doesn't shadow API endpoints
if _STATIC.exists():
    app.mount('/static', StaticFiles(directory=str(_STATIC)), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
