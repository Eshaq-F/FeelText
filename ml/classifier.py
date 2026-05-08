import numpy as np
from scipy.sparse import csr_matrix, issparse


def _build_indicator(y: np.ndarray, classes: np.ndarray) -> csr_matrix:
    """Sparse (n_classes, n_samples) class-indicator matrix."""
    n_samples = len(y)
    n_classes = len(classes)
    y_encoded = np.searchsorted(classes, y)
    return csr_matrix(
        (np.ones(n_samples), (y_encoded, np.arange(n_samples))),
        shape=(n_classes, n_samples),
        dtype=np.float64,
    )


class MultinomialNaiveBayes:
    """
    Standard Multinomial Naive Bayes from scratch.

    P(feature j | class c) estimated with Laplace smoothing.
    Kept as a reference baseline; prefer ComplementNaiveBayes for text.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self._class_log_prior: np.ndarray | None = None
        self._feature_log_prob: np.ndarray | None = None

    def fit(self, X, y: np.ndarray) -> 'MultinomialNaiveBayes':
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self._class_log_prior = np.log(class_counts / n_samples)

        feature_counts = (_build_indicator(y, self.classes_) @ X).toarray()
        smoothed = feature_counts + self.alpha
        self._feature_log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def _log_joint(self, X) -> np.ndarray:
        scores = X @ self._feature_log_prob.T
        if issparse(scores):
            scores = scores.toarray()
        return scores + self._class_log_prior

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        lj = self._log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)


NaiveBayesClassifier = MultinomialNaiveBayes


class ComplementNaiveBayes:
    """
    Complement Naive Bayes (CNB) from scratch.

    Instead of modelling P(feature | class c), CNB models
    P(feature | NOT class c) — the complement distribution.
    Weights are estimated from all classes except c, which makes
    CNB less sensitive to class imbalance and generally more accurate
    for text classification than standard Multinomial NB.

    Reference: Rennie et al. (2003) "Tackling the Poor Assumptions of
               Naive Bayes Text Classifiers", ICML.

    Key steps
    ---------
    1. feature_counts[c]     = sum of feature vectors for class c
    2. complement_counts[c]  = total_counts - feature_counts[c]
    3. log P(j | NOT c)      = log((complement_counts[c,j] + α) /
                                   (sum_j complement_counts[c,j] + α·|V|))
    4. Normalise weights by their L1 norm (Rennie et al. §2.3)
    5. Predict: argmax_c  ( log P(c) − X · complement_log_prob[c] )
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self._class_log_prior: np.ndarray | None = None
        self._complement_log_prob: np.ndarray | None = None

    def fit(self, X, y: np.ndarray) -> 'ComplementNaiveBayes':
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self._class_log_prior = np.log(class_counts / n_samples)

        # feature_counts: (n_classes, n_features)
        feature_counts = (_build_indicator(y, self.classes_) @ X).toarray()

        # complement_counts[c] = sum over all OTHER classes
        total_counts = feature_counts.sum(axis=0, keepdims=True)        # (1, n_features)
        complement_counts = total_counts - feature_counts               # (n_classes, n_features)

        # Laplace smoothing
        smoothed = complement_counts + self.alpha
        raw_log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))

        # L1 weight normalisation — reduces bias from unequal class sizes
        l1_norms = np.abs(raw_log_prob).sum(axis=1, keepdims=True)
        self._complement_log_prob = raw_log_prob / l1_norms

        return self

    def _log_joint(self, X) -> np.ndarray:
        # CNB score: log P(c) - X · complement_log_prob[c]
        # Lower complement score = document looks unlike the complement = belongs to c
        scores = -(X @ self._complement_log_prob.T)
        if issparse(scores):
            scores = scores.toarray()
        return scores + self._class_log_prior

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        lj = self._log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)
