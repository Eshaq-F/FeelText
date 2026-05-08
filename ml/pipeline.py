import pickle

import numpy as np

from ml.classifier import ComplementNaiveBayes
from ml.english_lexicon import EnglishLexiconScorer
from ml.persian_lexicon import PersianLexiconScorer
from ml.preprocessor import TextPreprocessor
from ml.vectorizer import TFIDFVectorizer

# Only predict neutral when the combined score is nearly 50/50.
# 0.52 means a 52 % vs 48 % split already gets a definite label.
_NEUTRAL_THRESHOLD = 0.52


def _lexicon_weight(n_tokens: int) -> float:
    """
    Return how much weight to give the lexicon vs the statistical model.

    Short texts have few TF-IDF vocabulary hits → rely more on lexicon.
    Long texts provide rich statistical evidence → trust the model more.
    """
    if n_tokens <= 5:
        return 0.75
    if n_tokens <= 15:
        return 0.50
    if n_tokens <= 40:
        return 0.25
    return 0.10


class SentimentPipeline:
    """
    End-to-end sentiment analysis pipeline.

    English inference
    -----------------
    1. Preprocessing  : HTML/URL strip · lowercase · stopword removal ·
                        negation marking (NOT_ prefix on content words only)
    2. Vectorisation  : TF-IDF with unigrams + bigrams, sublinear TF scaling
    3. Classification : Complement Naive Bayes
    4. Hybridisation  : Blends statistical model with English lexicon.
                        The shorter the input the more the lexicon contributes,
                        compensating for sparse TF-IDF evidence on short texts.

    Persian inference
    -----------------
    Lexicon-based scoring with negation handling.

    Neutral inference
    -----------------
    Neutral is predicted only when the blended pos/neg score is below
    _NEUTRAL_THRESHOLD (≈50/50 split). This avoids the "everything is
    neutral" trap caused by high-threshold binary confidence scores.
    """

    def __init__(self):
        self._en_preprocessor = TextPreprocessor('english')
        self._fa_preprocessor = TextPreprocessor('persian')
        self._vectorizer = TFIDFVectorizer(
            max_features=30000,
            min_df=3,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._classifier = ComplementNaiveBayes(alpha=0.5)
        self._en_lexicon = EnglishLexiconScorer()
        self._persian_scorer = PersianLexiconScorer()
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str], labels: list[int]) -> None:
        tokenized = self._en_preprocessor.process_batch(texts)
        X = self._vectorizer.fit_transform(tokenized)
        y = np.array(labels, dtype=np.int64)
        self._classifier.fit(X, y)
        self.is_trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_english(self, text: str) -> dict:
        tokens = self._en_preprocessor.process(text)

        # --- Statistical score (TF-IDF + Complement NB) ---
        X = self._vectorizer.transform([tokens])
        proba = self._classifier.predict_proba(X)[0]
        # classes_ = [0, 1]  →  index 0 = negative, index 1 = positive
        nb_pos = float(proba[1])
        nb_neg = float(proba[0])

        # --- Lexicon score ---
        lex = self._en_lexicon.score(tokens)

        if lex is not None:
            lw = _lexicon_weight(len(tokens))
            sw = 1.0 - lw
            raw_pos = sw * nb_pos + lw * lex['pos_score']
            raw_neg = sw * nb_neg + lw * lex['neg_score']
        else:
            raw_pos, raw_neg = nb_pos, nb_neg

        # Renormalise to a proper probability pair
        total = raw_pos + raw_neg
        if total > 0:
            pos_score = raw_pos / total
            neg_score = raw_neg / total
        else:
            pos_score = neg_score = 0.5

        max_score = max(pos_score, neg_score)

        if max_score < _NEUTRAL_THRESHOLD:
            sentiment = 'neutral'
        elif pos_score >= neg_score:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        neutral_score = round(max(0.0, 1.0 - max_score), 4)

        return {
            'sentiment': sentiment,
            'confidence': round(max_score, 4),
            'scores': {
                'positive': round(pos_score, 4),
                'negative': round(neg_score, 4),
                'neutral': neutral_score,
            },
        }

    def predict(self, text: str, language: str = 'english') -> dict:
        if language == 'persian':
            return self._persian_scorer.score(text)
        return self._predict_english(text)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'SentimentPipeline':
        with open(path, 'rb') as f:
            return pickle.load(f)
