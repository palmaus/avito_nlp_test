import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SEARCH = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_location_id",
    "item_microcat_id",
    "item_category_id",
    "item_rating",
    "item_rating_reviews_count",
]
FIELDS = {
    "title": "item_title_raw",
    "params": "item_infm_params_text",
    "description": "item_description_raw",
}
# Настройки приложенных весов. В notebook используются результаты его подборов.
SAVED_SEARCH = {
    "channels": {"stem": 1, "char": 0.5},
    "location": 8,
    "service": 1,
    "prior": 0.5,
    "alpha": 5,
    "beta": 0,
    "geo_strength": 128,
    "scope": "all",
    "e5_weight": 0.5,
    "e5_geo": 0.05,
    "e5_max_length": 80,
}
SEED = 2026092807


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_digest(df):
    h = hashlib.sha256()
    h.update(str(list(df.columns)).encode())
    for start in range(0, len(df), 8192):
        part = df.iloc[start : start + 8192].copy()
        for col in part.select_dtypes("object"):
            part[col] = part[col].map(lambda x: tuple(x) if isinstance(x, list) else x)
        h.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
    return h.hexdigest()


def labels():
    from .text import normalize

    df = pd.read_parquet(
        ROOT / "data/train.parquet",
        columns=SEARCH
        + ["item_id", "item_location_id", "item_microcat_id", "item_category_id"],
    )
    df["query_norm"] = df.search_query.map(normalize)
    return df


def corpus(benchmark=False):
    from .text import normalize

    parts, seen = [], set()
    files = ["benchmark_items.parquet"]
    if not benchmark:
        files.append("train.parquet")
    for name in files:
        for batch in pq.ParquetFile(ROOT / "data" / name).iter_batches(
            batch_size=8192, columns=ITEM
        ):
            df = batch.to_pandas()
            df = df[~df.item_id.isin(seen)].drop_duplicates("item_id").copy()
            seen.update(df.item_id)
            for col in FIELDS.values():
                df[col] = df[col].fillna("").map(normalize)
            df["item_description_raw"] = df.item_description_raw.str.slice(0, 2000)
            for col in ["item_id", *FIELDS.values()]:
                df[col] = df[col].astype("string[pyarrow]")
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def select_contexts(df, texts, split, rng):
    records = []
    for text, group in df[df.query_norm.isin(texts)].groupby("query_norm", sort=True):
        contexts = group[SEARCH].drop_duplicates().sort_values(SEARCH, kind="stable")
        chosen = contexts.iloc[int(rng.integers(len(contexts)))]
        mask = np.ones(len(group), dtype=bool)
        for col in SEARCH:
            value = "" if pd.isna(chosen[col]) else chosen[col]
            mask &= group[col].fillna("").to_numpy() == value
        row = chosen.to_dict()
        row.update(
            query_norm=text,
            split=split,
            query_id=f"{split}_{len(records):08d}",
            gold=sorted(group.loc[mask, "item_id"].unique()),
            known_positives=sorted(group.item_id.unique()),
        )
        records.append(row)
    return pd.DataFrame(records)


def split_data(
    df, seed=SEED, previous_seeds=(20260927, 2026092705, 2026092806), training=True
):
    previous = set()
    for previous_seed in previous_seeds:
        names = np.array(sorted(set(df.query_norm) - previous))
        np.random.default_rng(previous_seed).shuffle(names)
        previous.update(names[:1600])
    texts = np.array(sorted(set(df.query_norm) - previous))
    rng = np.random.default_rng(seed)
    rng.shuffle(texts)
    val, test = set(texts[:800]), set(texts[800:1600])
    evaluation = pd.concat(
        [select_contexts(df, val, "val", rng), select_contexts(df, test, "test", rng)],
        ignore_index=True,
    )
    fit = df[~df.query_norm.isin(previous | val | test)].copy()
    assert not set(fit.query_norm) & set(evaluation.query_norm)
    if not training:
        return evaluation, fit, []
    texts = np.array(sorted(fit.query_norm.unique()))
    rng.shuffle(texts)
    folds = {text: i % 3 for i, text in enumerate(texts)}
    fit["fold"] = fit.query_norm.map(folds)
    blocks = []
    for fold in range(3):
        selected = {t for t in [t for t in texts if folds[t] == fold][:2000]}
        queries = select_contexts(fit, selected, f"train{fold}", rng)
        fitting = fit[fit.fold.ne(fold)]
        assert not set(queries.query_norm) & set(fitting.query_norm)
        blocks.append((queries, fitting))
    assert not set(fit.query_norm) & set(evaluation.query_norm)
    return evaluation, fit, blocks
