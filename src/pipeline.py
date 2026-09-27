import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l1norm
from catboost import CatBoostRanker
import re

from .data import ROOT, FIELDS, SEED, digest, frame_digest
from .text import build, bm25, normalize, stem_word, top
from .geography import GeoModel, location_factor
from .features import (
    core_features,
    field_features,
    union_candidates,
    training_selection,
    embedding_features,
    rank_column,
    CORE_NAMES,
    FIELD_NAMES,
    EMBEDDING_NAMES,
)
from .encoder import embeddings, item_text
from .ranking import blend_order, TREES, RANKER_WEIGHT
from .retrieval import collect
from .artifacts import feature_key, save_features, load_features, add_columns


def prior_probabilities(fit, queries):
    texts = sorted(fit.query_norm.unique())
    classes = np.sort(fit.item_microcat_id.unique())
    rows = pd.Categorical(fit.query_norm, categories=texts).codes
    cols = pd.Categorical(fit.item_microcat_id, categories=classes).codes
    y = sparse.csr_matrix(
        (np.ones(len(fit), np.float32), (rows, cols)), shape=(len(texts), len(classes))
    )
    y = l1norm(y, norm="l1", axis=1)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x = vectorizer.fit_transform(texts).tocsr()
    q = vectorizer.transform(queries.search_query.map(normalize))
    probabilities = np.zeros((len(queries), len(classes)), np.float32)
    for start in range(0, len(q.indptr) - 1, 16):
        for j, scores in enumerate((q[start : start + 16] @ x.T).toarray()):
            neighbors = top(scores, 10)
            weights = scores[neighbors] ** 2
            weights[scores[neighbors] < 0.2] = 0
            if weights.sum() > 0:
                probabilities[start + j] = np.asarray(
                    weights @ y[neighbors] / weights.sum()
                ).ravel()
    return np.pad(probabilities, ((0, 0), (0, 1))), classes


def field_indexes(items, tag, cache):
    build("stem", items, tag, cache)
    stamp = cache / f"{tag}_fields.key"
    signature = frame_digest(items[list(FIELDS.values())]) + digest(Path(__file__))
    signature += digest(ROOT / "src/features.py") + digest(ROOT / "src/text.py")
    fresh = stamp.exists() and stamp.read_text() == signature
    raw = joblib.load(cache / f"{tag}_word_vectorizer.joblib")
    vectorizer = joblib.load(cache / f"{tag}_stem_vectorizer.joblib")
    vocabulary = getattr(vectorizer, "vocabulary_", None) or vectorizer.vocabulary
    tokens = raw.get_feature_names_out()
    projection = sparse.csr_matrix(
        (
            np.ones(len(tokens), np.float32),
            (np.arange(len(tokens)), [vocabulary[stem_word(t)] for t in tokens]),
        ),
        shape=(len(tokens), len(vocabulary)),
    )
    matrices, stats = {}, {}
    for field, column in FIELDS.items():
        matrix_path, stats_path = (
            cache / f"{tag}_{field}_bm25.npz",
            cache / f"{tag}_{field}_stats.npz",
        )
        if not fresh or not matrix_path.exists() or not stats_path.exists():
            counts = (raw.transform(items[column]) @ projection).tocsr()
            counts.sum_duplicates()
            counts.eliminate_zeros()
            lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
            df = np.bincount(counts.indices, minlength=counts.shape[1])
            idf = np.log1p((len(items) - df + 0.5) / (df + 0.5)).astype(np.float32)
            sparse.save_npz(matrix_path, bm25(counts).T.tocsr())
            np.savez_compressed(stats_path, lengths=lengths, idf=idf)
            del counts
            gc.collect()
        matrices[field] = sparse.load_npz(matrix_path)
        with np.load(stats_path) as saved:
            stats[field] = {name: saved[name] for name in ["lengths", "idf"]}
    stamp.write_text(signature)
    return vocabulary, matrices, stats


def semantic_candidates(items, queries, geo, vectors, q, config):
    locations = items.item_location_id.to_numpy()
    positions, location_set = geo.destination_positions(locations), set(locations)
    result = {
        name: np.empty((len(q), min(500, len(items))), dtype=dtype)
        for name, dtype in [
            ("global_ids", np.int32),
            ("geo_ids", np.int32),
            ("global_scores", np.float32),
            ("geo_scores", np.float32),
        ]
    }
    for start in range(0, len(q), 16):
        for j, cosine in enumerate(q[start : start + 16] @ vectors.T):
            i = start + j
            source = queries.iloc[i].search_location_id
            factors = location_factor(
                source, locations, positions, geo, config, source in location_set
            )
            scores = cosine + config["e5_geo"] * np.log(factors)
            a, b = top(cosine, 500), top(scores, 500)
            result["global_ids"][i], result["geo_ids"][i] = a, b
            result["global_scores"][i], result["geo_scores"][i] = cosine[a], scores[b]
    return result


def prepare_core(
    items,
    queries,
    fit,
    config,
    folder,
    *,
    bundle=None,
    rebuild=False,
    semantic=False,
    training=False,
    fold=0,
    seed=SEED,
    tag="local",
    index_dir=None,
    item_vectors=None,
    query_vectors=None,
):
    if "query_norm" in queries:
        assert not set(queries.query_norm) & set(fit.query_norm), "Query leakage"
    folder = Path(folder)
    key = feature_key(
        items,
        queries,
        fit,
        config,
        semantic=semantic,
        training=training,
        seed=seed if training else 0,
        fold=fold if training else 0,
    )
    if not rebuild:
        source = folder if (folder / "core.npz").exists() else Path(bundle or folder)
        return load_features(source, items, queries, key)
    folder.mkdir(parents=True, exist_ok=True)
    cache = Path(index_dir) if index_dir is not None else ROOT / "cache/indices"
    geo = GeoModel.fit(fit)
    probabilities, classes = prior_probabilities(fit, queries)
    candidates = {
        channel: collect(items, queries, channel, cache, geo, config, tag)
        for channel in ["stem", "char"]
    }
    vectorizer = joblib.load(cache / f"{tag}_stem_vectorizer.joblib")
    vocabulary = getattr(vectorizer, "vocabulary_", None) or vectorizer.vocabulary
    mapping = {value: i for i, value in enumerate(classes)}
    item_data = {
        name: items[name].to_numpy()
        for name in [
            "item_id",
            "item_location_id",
            "item_category_id",
            "item_microcat_id",
        ]
    }
    item_data.update(
        rating=pd.to_numeric(items.item_rating, errors="coerce").to_numpy(np.float32),
        reviews=pd.to_numeric(
            items.item_rating_reviews_count, errors="coerce"
        ).to_numpy(np.float32),
        location_set=set(items.item_location_id),
        vocabulary=vocabulary,
        geo_positions=geo.destination_positions(items.item_location_id),
        prior_positions=np.array(
            [mapping.get(c, len(classes)) for c in items.item_microcat_id]
        ),
    )
    if semantic:
        vectors = (
            item_vectors
            if item_vectors is not None
            else embeddings(
                item_text(items),
                "passage: ",
                ROOT / "cache/embeddings" / f"{tag}_items.npy",
                max_length=config.get("e5_max_length", 80),
            )
        )
        q = (
            query_vectors
            if query_vectors is not None
            else embeddings(
                queries.search_query.map(normalize).tolist(),
                "query: ",
                folder / "query_embeddings.npy",
                max_length=config.get("e5_max_length", 80),
            )
        )
        assert vectors.shape == (len(items), 384) and q.shape == (len(queries), 384)
        neighbors = semantic_candidates(items, queries, geo, vectors, q, config)
        np.savez_compressed(folder / "neighbors.npz", **neighbors)
    capacity = (
        sum(192 + len(x) for x in queries.gold)
        if training
        else len(queries) * (3800 if semantic else 2800)
    )
    arrays = {}
    for name, dtype, shape in [
        ("X", np.float32, (capacity, len(CORE_NAMES))),
        ("ids", np.int32, (capacity,)),
        ("y", np.float32, (capacity,)),
        ("rrf", np.float64, (capacity,)),
        ("text_rrf", np.float64, (capacity,)),
        ("text_mask", np.bool_, (capacity,)),
    ]:
        arrays[name] = np.lib.format.open_memmap(
            folder / f"{name}.work.npy", mode="w+", dtype=dtype, shape=shape
        )
    offsets = [0]
    for i, query in enumerate(queries.itertuples(index=False)):
        text_ids = union_candidates(candidates, i, len(items))
        ids = (
            np.unique(
                np.r_[text_ids, neighbors["global_ids"][i], neighbors["geo_ids"][i]]
            )
            if semantic
            else text_ids
        )
        core, text_rrf = core_features(
            query, ids, candidates, i, item_data, probabilities[i], geo, config
        )
        rrf = text_rrf.copy()
        mask = np.ones(len(ids), bool)
        if semantic:
            rrf += (
                config["e5_weight"]
                * rank_column(ids, neighbors["geo_ids"][i])
                * (1 + config["prior"] * core[:, 20])
            )
            mask = np.isin(ids, text_ids)
        if training:
            rng = np.random.default_rng(seed + fold * 10000 + i)
            keep = training_selection(
                ids,
                rrf,
                item_data["item_id"],
                set(query.gold),
                set(query.known_positives),
                rng,
            )
            ids, core, rrf, text_rrf, mask = (
                ids[keep],
                core[keep],
                rrf[keep],
                text_rrf[keep],
                mask[keep],
            )
        lo, hi = offsets[-1], offsets[-1] + len(ids)
        arrays["X"][lo:hi], arrays["ids"][lo:hi] = core, ids
        arrays["rrf"][lo:hi], arrays["text_rrf"][lo:hi], arrays["text_mask"][lo:hi] = (
            rrf,
            text_rrf,
            mask,
        )
        arrays["y"][lo:hi] = (
            np.isin(item_data["item_id"][ids], query.gold) if training else -1
        )
        offsets.append(hi)
        if i % 400 == 0:
            print("core features", folder.name, i, "/", len(queries), flush=True)
    result = {name: a[: offsets[-1]] for name, a in arrays.items()}
    result["offsets"] = np.array(offsets, np.int64)
    save_features(folder, result, items, queries, key)
    del arrays, result
    for file in folder.glob("*.work.npy"):
        file.unlink()
    return load_features(folder, items, queries, key)


def add_fields(features, items, queries, *, rebuild=False, tag="local", index_dir=None):
    assert features["X"].shape[1] == len(CORE_NAMES)
    if not rebuild:
        return add_columns(features, "fields")
    cache = Path(index_dir) if index_dir is not None else ROOT / "cache/indices"
    vocabulary, matrices, stats = field_indexes(items, tag, cache)
    texts = {name: items[col].to_numpy() for name, col in FIELDS.items()}
    texts["numbers"] = [
        set(re.findall(r"\d+", a + " " + b))
        for a, b in zip(texts["title"], texts["params"])
    ]
    rating = pd.to_numeric(items.item_rating, errors="coerce").to_numpy(np.float32)
    values = np.empty((len(features["ids"]), len(FIELD_NAMES)), np.float32)
    for i, query in enumerate(queries.itertuples(index=False)):
        lo, hi = features["offsets"][i : i + 2]
        ids = features["ids"][lo:hi]
        if len(ids):
            values[lo:hi] = field_features(
                query, ids, matrices, stats, texts, vocabulary, rating
            )
    return add_columns(features, "fields", values)


def add_embeddings(
    features,
    items,
    queries,
    fit,
    config,
    *,
    rebuild=False,
    item_vectors=None,
    query_vectors=None,
    tag="local",
):
    assert features["X"].shape[1] == len(CORE_NAMES) + len(FIELD_NAMES)
    if not rebuild:
        return add_columns(features, "embedding_features")
    folder = features["_folder"]
    vectors = (
        item_vectors
        if item_vectors is not None
        else embeddings(
            item_text(items),
            "passage: ",
            ROOT / "cache/embeddings" / f"{tag}_items.npy",
            max_length=config.get("e5_max_length", 80),
        )
    )
    q = (
        query_vectors
        if query_vectors is not None
        else embeddings(
            queries.search_query.map(normalize).tolist(),
            "query: ",
            folder / "query_embeddings.npy",
            max_length=config.get("e5_max_length", 80),
        )
    )
    geo = GeoModel.fit(fit)
    locations = items.item_location_id.to_numpy()
    positions, location_set = geo.destination_positions(locations), set(locations)
    with np.load(folder / "neighbors.npz") as saved:
        neighbors = dict(saved)
    values = np.empty((len(features["ids"]), len(EMBEDDING_NAMES)), np.float32)
    for i, query in enumerate(queries.itertuples(index=False)):
        lo, hi = features["offsets"][i : i + 2]
        ids = features["ids"][lo:hi]
        if not len(ids):
            continue
        factors = location_factor(
            query.search_location_id,
            locations[ids],
            positions[ids],
            geo,
            config,
            query.search_location_id in location_set,
        )
        values[lo:hi], _, _ = embedding_features(
            ids,
            ids[features["text_mask"][lo:hi]],
            q[i],
            vectors,
            neighbors,
            i,
            factors,
            features["text_rrf"][lo:hi],
            features["X"][lo:hi, 20],
            config,
        )
    return add_columns(features, "embedding_features", values)


def predict(
    features,
    model_path=None,
    trees=None,
    weight=None,
    dimensions=None,
    *,
    indices=None,
    text_only=False,
):
    model_path = model_path or ROOT / "model/ranker.cbm"
    model = (
        model_path
        if isinstance(model_path, CatBoostRanker)
        else CatBoostRanker().load_model(str(model_path))
    )
    metadata = dict(model.get_metadata())
    trees = int(metadata.get("selected_trees", TREES)) if trees is None else trees
    weight = (
        float(metadata.get("ranker_weight", RANKER_WEIGHT))
        if weight is None
        else weight
    )
    dimensions = len(model.feature_names_) if dimensions is None else dimensions
    if features["X"].shape[1] < dimensions:
        features = (
            add_columns(features, "fields")
            if features["X"].shape[1] == 35
            else features
        )
        if dimensions == 66 and features["X"].shape[1] == 57:
            features = add_columns(features, "embedding_features")
    if features["X"].shape[1] < dimensions:
        raise ValueError(f"Модель ожидает {dimensions} признаков")
    indices = (
        np.arange(len(features["offsets"]) - 1)
        if indices is None
        else np.asarray(indices)
    )
    predictions = []
    # Предсказание блоками ограничивает расход памяти на полном каталоге.
    for start in range(0, len(indices), 64):
        batch = indices[start : start + 64]
        bounds = [(features["offsets"][i], features["offsets"][i + 1]) for i in batch]
        x = np.concatenate([features["X"][lo:hi, :dimensions] for lo, hi in bounds])
        scores = model.predict(x, ntree_end=trees, thread_count=6)
        cursor = 0
        for lo, hi in bounds:
            values = scores[cursor : cursor + hi - lo]
            cursor += hi - lo
            if text_only:
                from .features import ranks

                mixed = weight / (60 + ranks(values)) + (1 - weight) / (
                    60 + ranks(features["rrf"][lo:hi])
                )
                keep = np.flatnonzero(features["text_mask"][lo:hi])
                order = keep[top(mixed[keep], 50)]
            else:
                order = blend_order(values, features["rrf"][lo:hi], weight)
            predictions.append(features["ids"][lo:hi][order])
    return predictions


def export_answer(queries, items, predictions, path):
    ids = items.item_id.to_numpy()
    answer = pd.DataFrame(
        {
            "query_id": queries.query_id,
            "answer": [" ".join(ids[p]) for p in predictions],
        }
    )
    assert answer.query_id.is_unique and len(answer) == len(queries)
    assert answer.query_id.astype(str).str.fullmatch(r"[A-Za-z0-9]{16}").all()
    for value in answer.answer:
        chosen = value.split()
        assert len(chosen) == len(set(chosen)) == 50
        assert all(re.fullmatch(r"[a-f0-9]{16}", x) for x in chosen)
    answer.to_csv(path, index=False)
    return {"rows": len(answer), "file": str(path.relative_to(ROOT))}
