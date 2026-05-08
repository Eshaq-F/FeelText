import math
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


class TFIDFVectorizer:
    """
    TF-IDF vectorizer built entirely from scratch.

    TF  = sublinear: 1 + log(count)  |  raw: count / doc_length
    IDF = log((1 + N) / (1 + df(term))) + 1   (sklearn-style smooth IDF)
    TF-IDF = TF * IDF, then L2-normalised per document.

    ngram_range controls unigram-only (1,1) or unigram+bigram (1,2) features.
    Bigrams capture short phrases like "not good", "very bad", "really great".
    """

    def __init__(
        self,
        max_features: int = 30000,
        min_df: int = 3,
        max_df: float = 0.90,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vocabulary: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # N-gram generation
    # ------------------------------------------------------------------

    def _ngrams(self, tokens: list[str]) -> list[str]:
        min_n, max_n = self.ngram_range
        grams: list[str] = []
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            if n == 1:
                grams.extend(tokens)
            else:
                grams.extend(
                    ' '.join(tokens[i:i + n]) for i in range(n_tokens - n + 1)
                )
        return grams

    # ------------------------------------------------------------------
    # TF calculation
    # ------------------------------------------------------------------

    def _tf(self, count: int, doc_len: int) -> float:
        if self.sublinear_tf:
            return 1.0 + math.log(count) if count > 0 else 0.0
        return count / doc_len

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, tokenized_docs: list[list[str]]) -> 'TFIDFVectorizer':
        n_docs = len(tokenized_docs)
        doc_freq: Counter = Counter()

        for tokens in tokenized_docs:
            doc_freq.update(set(self._ngrams(tokens)))

        min_count = self.min_df
        max_count = int(self.max_df * n_docs)

        candidates = sorted(
            [(t, f) for t, f in doc_freq.items() if min_count <= f <= max_count],
            key=lambda x: -x[1],
        )[:self.max_features]

        self.vocabulary = {term: idx for idx, (term, _) in enumerate(candidates)}

        self._idf = np.array(
            [math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term, _ in candidates],
            dtype=np.float32,
        )
        return self

    def transform(self, tokenized_docs: list[list[str]]) -> csr_matrix:
        rows, cols, vals = [], [], []
        vocab = self.vocabulary
        idf = self._idf

        for doc_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            grams = self._ngrams(tokens)
            counts = Counter(grams)
            doc_len = len(grams)

            for gram, count in counts.items():
                feat_idx = vocab.get(gram)
                if feat_idx is not None:
                    rows.append(doc_idx)
                    cols.append(feat_idx)
                    vals.append(self._tf(count, doc_len) * idf[feat_idx])

        X = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(tokenized_docs), len(self.vocabulary)),
            dtype=np.float32,
        )

        # L2 row normalisation — makes cosine similarity the implicit metric
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        norms = np.where(norms == 0, 1.0, norms)
        X = X.multiply(1.0 / norms[:, np.newaxis])

        return X

    def fit_transform(self, tokenized_docs: list[list[str]]) -> csr_matrix:
        return self.fit(tokenized_docs).transform(tokenized_docs)

    @property
    def n_features(self) -> int:
        return len(self.vocabulary)
