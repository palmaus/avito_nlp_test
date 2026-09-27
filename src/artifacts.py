import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, SEARCH, digest, frame_digest


def feature_key(
    items, queries, fit, config, *, semantic=False, training=False, seed=0, fold=0
):
    h = hashlib.sha256()
    columns = [
        c
        for c in [*SEARCH, *(["gold", "known_positives"] if training else [])]
        if c in queries
    ]
    config = {k: v for k, v in config.items() if semantic or not k.startswith("e5_")}
    for value in [
        frame_digest(items),
        frame_digest(queries[columns]),
        frame_digest(fit),
        json.dumps([config, semantic, training, seed, fold], sort_keys=True),
    ]:
        h.update(value.encode())
    for name in [
        "data",
        "text",
        "retrieval",
        "geography",
        "features",
        "pipeline",
        "encoder",
    ]:
        h.update(digest(ROOT / "src" / f"{name}.py").encode())
    if semantic:
        from .encoder import Encoder

        h.update(
            Encoder.fingerprint(max_length=config.get("e5_max_length", 80)).encode()
        )
    return h.hexdigest()


def save_features(folder, features, items, queries, key):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = {
        name: features[name]
        for name in ["X", "ids", "y", "rrf", "text_rrf", "text_mask", "offsets"]
        if name in features
    }
    np.savez_compressed(folder / "core.npz", **values, key=np.array(key))
    np.save(folder / "item_ids.npy", items.item_id.to_numpy(dtype=str))
    queries[["query_id", *SEARCH]].to_parquet(
        folder / "query_rows.parquet", index=False
    )


def load_features(folder, items, queries, key=None):
    folder = Path(folder)
    saved_ids = np.load(folder / "item_ids.npy", mmap_mode="r")
    if not np.array_equal(saved_ids, items.item_id.to_numpy(dtype=str)):
        raise ValueError("Каталог или порядок объявлений изменился. Включите REBUILD.")
    check_queries(folder, queries)
    with np.load(folder / "core.npz") as saved:
        saved_key = str(saved["key"])
        if key is not None and saved_key != key:
            raise ValueError(
                f"Данные или настройки изменились для {folder.name}. Включите REBUILD."
            )
        arrays = {name: saved[name] for name in saved.files if name != "key"}
    assert len(arrays["offsets"]) == len(queries) + 1
    assert arrays["offsets"][0] == 0 and np.all(np.diff(arrays["offsets"]) >= 0)
    assert all(
        len(arrays[k]) == arrays["offsets"][-1]
        for k in ["X", "ids", "rrf", "text_mask"]
    )
    if len(arrays["ids"]):
        assert arrays["ids"].min() >= 0 and arrays["ids"].max() < len(items)
    arrays.update(_folder=folder, _key=saved_key)
    return arrays


def check_queries(folder, queries):
    saved = pd.read_parquet(Path(folder) / "query_rows.parquet")
    try:
        pd.testing.assert_frame_equal(
            saved,
            queries[["query_id", *SEARCH]].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "Запросы или их порядок изменились. Включите REBUILD."
        ) from exc


def add_columns(features, name, values=None):
    path = features["_folder"] / f"{name}.npz"
    if values is not None:
        np.savez_compressed(path, X=values, key=np.array(features["_key"]))
    else:
        with np.load(path) as saved:
            if str(saved["key"]) != features["_key"]:
                raise ValueError(f"Устарели признаки {name}. Включите REBUILD.")
            values = saved["X"]
    assert len(values) == len(features["X"])
    return {**features, "X": np.column_stack([features["X"], values])}
