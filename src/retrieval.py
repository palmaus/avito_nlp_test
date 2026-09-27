import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .geography import location_factor
from .text import build, normalize, stem_text, top
from .data import ROOT, SEARCH, digest, frame_digest
from .artifacts import check_queries


def search_key(items, queries, channel, geo=None, config=None):
    h = hashlib.sha256()
    for value in [frame_digest(items), frame_digest(queries[SEARCH]), channel]:
        h.update(value.encode())
    for name in ["text", "geography", "retrieval"]:
        h.update(digest(ROOT / "src" / f"{name}.py").encode())
    if geo is not None:
        h.update(json.dumps(config, sort_keys=True).encode())
        for a in [
            geo.sources,
            geo.destinations,
            geo.counts.data,
            geo.counts.indices,
            geo.counts.indptr,
            geo.support,
            geo.background,
        ]:
            h.update(a.tobytes())
    return h.hexdigest()


def cached_collect(
    items,
    queries,
    channel,
    folder,
    *,
    bundle=None,
    rebuild=False,
    geo=None,
    config=None,
    suffix="",
    tag="local",
    index_dir=None,
):
    folder = Path(folder)
    filename = f"{channel}{suffix}.npz"
    key = search_key(items, queries, channel, geo, config)
    if not rebuild:
        source = folder if (folder / filename).exists() else Path(bundle or folder)
        check_queries(source, queries)
        with np.load(source / filename) as saved:
            if str(saved["key"]) != key:
                raise ValueError(
                    f"Изменились данные или настройки {filename}. Включите REBUILD."
                )
            return saved["indices"], saved["scores"]
    cache = Path(index_dir) if index_dir is not None else ROOT / "cache/indices"
    ids, scores = collect(items, queries, channel, cache, geo, config, tag)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / filename, indices=ids, scores=scores, key=np.array(key)
    )
    queries[["query_id", *SEARCH]].to_parquet(
        folder / "query_rows.parquet", index=False
    )
    return ids, scores


def collect(items, queries, channel, cache, geo=None, config=None, tag="local"):
    """Сохраняет оценки до настройки географии и смешивания каналов."""
    vectorizer, index = build(channel, items, tag, cache)
    q = vectorizer.transform(
        queries.search_query.map(stem_text if channel == "stem" else normalize)
    ).tocsr()
    if channel != "char":
        q.data[:] = 1
    locations = items.item_location_id.to_numpy()
    service = items.item_category_id.to_numpy() == 114
    if geo is not None:
        positions = geo.destination_positions(locations)
        location_set = set(locations)
    width = 900 if geo is None else 1400
    ids = np.full((len(queries), width), -1, np.int32)
    values = np.zeros(ids.shape, np.float32)
    for start in range(0, len(queries), 8):
        for j, scores in enumerate((q[start : start + 8] @ index).toarray()):
            i = start + j
            source = queries.iloc[i].search_location_id
            local = locations == source
            parts = [top(scores, 500)]
            for is_local in [False, True]:
                for is_service in [False, True]:
                    members = np.flatnonzero(
                        (local == is_local) & (service == is_service)
                    )
                    parts.append(members[top(scores[members], 100)])
            if geo is not None:
                weighted = scores * location_factor(
                    source, locations, positions, geo, config, source in location_set
                )
                weighted *= np.where(service, config.get("service", 1), 1)
                parts.append(top(weighted, 500))
            selected = np.unique(np.concatenate(parts))
            ids[i, : len(selected)] = selected
            values[i, : len(selected)] = scores[selected]
    del index, q, vectorizer
    gc.collect()
    return ids, values


def search(items, queries, candidates, config, indices, prior=None, geo=None):
    locations = items.item_location_id.to_numpy()
    categories = items.item_category_id.to_numpy()
    if geo is not None:
        positions = geo.destination_positions(locations)
        location_set = set(locations)
    if prior is not None:
        probabilities, classes = prior
        mapping = {value: i for i, value in enumerate(classes)}
        microcats = np.array(
            [mapping.get(value, len(classes)) for value in items.item_microcat_id]
        )
    predictions = []
    for i in indices:
        source = queries.iloc[i].search_location_id
        parts, contributions = [], []
        for channel, weight in config["channels"].items():
            ids, raw = candidates[channel]
            valid = ids[i] >= 0
            ids, scores = ids[i, valid], raw[i, valid].copy()
            if geo is None:
                scores *= np.where(locations[ids] == source, config["location"], 1)
            else:
                scores *= location_factor(
                    source,
                    locations[ids],
                    positions[ids],
                    geo,
                    config,
                    source in location_set,
                )
            scores *= np.where(categories[ids] == 114, config.get("service", 1), 1)
            if len(config["channels"]) == 1:
                predictions.append(ids[top(scores, 50)])
                break
            ranks = np.empty(len(scores), np.int32)
            ranks[top(scores, len(scores))] = np.arange(1, len(scores) + 1)
            positive = scores > 0
            parts.append(ids[positive])
            contributions.append(weight / (60 + ranks[positive]))
        else:
            ids, inverse = np.unique(np.concatenate(parts), return_inverse=True)
            scores = np.bincount(inverse, weights=np.concatenate(contributions)).astype(
                np.float64, copy=False
            )
            if prior is not None and config.get("prior", 0):
                p = probabilities[i]
                scores *= 1 + config["prior"] * p[microcats[ids]] / max(p.max(), 1e-12)
            result = ids[top(scores, 50)]
            if len(result) < 50:
                fallback = np.setdiff1d(np.arange(min(len(items), 100)), result)
                result = np.r_[result, fallback[: 50 - len(result)]]
            predictions.append(result.astype(np.int32))
    return predictions
