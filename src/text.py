import re
import hashlib
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import RussianStemmer
from .data import digest, frame_digest

_STEM = RussianStemmer()


def normalize(value):
    return re.sub(r"\s+", " ", str(value).lower().replace("ё", "е")).strip()


@lru_cache(maxsize=400000)
def stem_word(w):
    return _STEM.stem(w)


def stem_text(s):
    return " ".join(stem_word(t) for t in re.findall(r"(?u)\b\w\w+\b", normalize(s)))


def bm25(counts):
    counts = counts.tocsr().astype(np.float32)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    n = counts.shape[0]
    length = np.asarray(counts.sum(axis=1)).ravel()
    df = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    denominator = 1.5 * (0.25 + 0.75 * length / max(float(length.mean()), 1))
    # Обрабатываем по блокам, чтобы не держать временные массивы всего каталога.
    for start in range(0, n, 8192):
        end = min(start + 8192, n)
        lo, hi = counts.indptr[start], counts.indptr[end]
        x = counts.data[lo:hi]
        x *= 2.5 / (
            x
            + np.repeat(denominator[start:end], np.diff(counts.indptr[start : end + 1]))
        )
        x *= idf[counts.indices[lo:hi]]
    return counts.T.tocsr()


def top(scores, k):
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=np.int32)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    candidates = np.flatnonzero(scores >= threshold)
    return candidates[np.lexsort((candidates, -scores[candidates]))[:k]]


def build(channel, items, tag, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    vp = cache / f"{tag}_{channel}_vectorizer.joblib"
    ip = cache / f"{tag}_{channel}_index.npz"
    stamp = cache / f"{tag}_{channel}.key"
    columns = ["item_title_raw", "item_infm_params_text", "item_description_raw"]
    signature = hashlib.sha256(
        (frame_digest(items[columns]) + digest(Path(__file__)) + channel).encode()
    ).hexdigest()
    if channel == "stem":
        build("word", items, tag, cache)
    if (
        vp.exists()
        and ip.exists()
        and stamp.exists()
        and stamp.read_text() == signature
    ):
        return joblib.load(vp), sparse.load_npz(ip)
    if channel == "word":
        vectorizer = CountVectorizer(
            dtype=np.float32, max_features=300000, token_pattern=r"(?u)\b\w\w+\b"
        )
        docs = (
            f"{r.item_title_raw} {r.item_title_raw} {r.item_title_raw} {r.item_infm_params_text} {r.item_description_raw}"
            for r in items.itertuples()
        )
        counts = vectorizer.fit_transform(docs).tocsr()
        sparse.save_npz(cache / f"{tag}_word_counts.npz", counts)
        index = bm25(counts)
    elif channel == "stem":
        raw = joblib.load(cache / f"{tag}_word_vectorizer.joblib")
        tokens = raw.get_feature_names_out()
        stems = [stem_word(w) for w in tokens]
        unique = sorted(set(stems))
        vocab = {w: i for i, w in enumerate(unique)}
        projection = sparse.csr_matrix(
            (
                np.ones(len(tokens), np.float32),
                (np.arange(len(tokens)), [vocab[w] for w in stems]),
            ),
            shape=(len(tokens), len(unique)),
        )
        counts = sparse.load_npz(cache / f"{tag}_word_counts.npz") @ projection
        vectorizer = CountVectorizer(
            dtype=np.float32, vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b"
        )
        index = bm25(counts)
    elif channel == "char":
        # Берём только заголовки: адреса и прайс-листы мешают искать короткие названия услуг.
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=180000,
            dtype=np.float32,
            sublinear_tf=True,
        )
        index = vectorizer.fit_transform(items.item_title_raw).T.tocsr()
    else:
        raise ValueError(channel)
    joblib.dump(vectorizer, vp)
    sparse.save_npz(ip, index)
    stamp.write_text(signature)
    print("built", channel, index.shape, flush=True)
    return vectorizer, index
