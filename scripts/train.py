"""
Training script for the FeelText custom sentiment model.

Dataset : IMDB Large Movie Review Dataset (Maas et al., 2011)
          http://ai.stanford.edu/~amaas/data/sentiment/
Algorithm: TF-IDF (from scratch) + Multinomial Naive Bayes (from scratch)

Usage:
    python scripts/train.py
"""

import logging
import sys
import tarfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.pipeline import SentimentPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

IMDB_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
ARCHIVE_NAME = 'aclImdb_v1.tar.gz'
IMDB_DIR_NAME = 'aclImdb'


# --------------------------------------------------------------------------- #
# Data helpers                                                                 #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> None:
    logger.info('Downloading %s ...', url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(dest, 'wb') as fh:
        for chunk in response.iter_content(chunk_size=65536):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f'\r  {pct:5.1f}% ({downloaded // 1_048_576} MB / {total // 1_048_576} MB)',
                      end='', flush=True)
    print()


def _ensure_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    imdb_dir = data_dir / IMDB_DIR_NAME

    if not archive.exists():
        _download(IMDB_URL, archive)
    else:
        logger.info('Archive already present: %s', archive)

    if not imdb_dir.exists():
        logger.info('Extracting archive ...')
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(data_dir)
        logger.info('Extracted to %s', imdb_dir)
    else:
        logger.info('Dataset already extracted: %s', imdb_dir)

    return imdb_dir


def _load_split(base: Path, split: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for label_name, label_idx in (('pos', 1), ('neg', 0)):
        label_dir = base / split / label_name
        for path in sorted(label_dir.glob('*.txt')):
            texts.append(path.read_text(encoding='utf-8', errors='ignore'))
            labels.append(label_idx)
    return texts, labels


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float((y_true == y_pred).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    imdb_dir = _ensure_dataset(DATA_DIR)

    logger.info('Loading training split ...')
    train_texts, train_labels = _load_split(imdb_dir, 'train')
    logger.info('  %d samples loaded', len(train_texts))

    logger.info('Loading test split ...')
    test_texts, test_labels = _load_split(imdb_dir, 'test')
    logger.info('  %d samples loaded', len(test_texts))

    pipeline = SentimentPipeline()

    logger.info('Training TF-IDF + Naive Bayes ...')
    pipeline.train(train_texts, train_labels)
    logger.info(
        '  Vocabulary size : %d features',
        pipeline._vectorizer.n_features,
    )

    logger.info('Evaluating on test set ...')
    test_tokens = pipeline._en_preprocessor.process_batch(test_texts)
    X_test = pipeline._vectorizer.transform(test_tokens)
    y_pred = pipeline._classifier.predict(X_test)
    y_true = np.array(test_labels, dtype=np.int64)

    metrics = _compute_metrics(y_true, y_pred)

    logger.info('─' * 45)
    logger.info('  Accuracy  : %.4f', metrics['accuracy'])
    logger.info('  Precision : %.4f', metrics['precision'])
    logger.info('  Recall    : %.4f', metrics['recall'])
    logger.info('  F1 Score  : %.4f', metrics['f1'])
    logger.info('─' * 45)

    model_path = MODELS_DIR / 'sentiment_pipeline.pkl'
    pipeline.save(str(model_path))
    logger.info('Model saved → %s', model_path)


if __name__ == '__main__':
    main()
