from typing import Literal

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchTextInput(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class SentimentResult(BaseModel):
    text: str
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]
    language_detected: str = 'other'


class BatchSentimentResult(BaseModel):
    results: list[SentimentResult]
    total_analyzed: int


class HealthResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    status: str
    model_loaded: bool
    model_name: str
    mode: str


class LanguagesResponse(BaseModel):
    languages: list[str]
    total: int
    note: str
