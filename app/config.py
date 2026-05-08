import os


class Settings:
    USE_CUSTOM_MODEL: bool = os.getenv('USE_CUSTOM_MODEL', 'true').lower() == 'true'

    PERSIAN_MODEL: str = os.getenv(
        'PERSIAN_MODEL',
        'HooshvareLab/bert-fa-base-uncased-sentiment-digikala',
    )
    MULTILINGUAL_MODEL: str = os.getenv(
        'MULTILINGUAL_MODEL',
        'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
    )

    CUSTOM_MODEL_PATH: str = os.getenv('CUSTOM_MODEL_PATH', 'models/sentiment_pipeline.pkl')


settings = Settings()
