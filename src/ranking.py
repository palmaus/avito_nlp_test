import hashlib
import json
from pathlib import Path

from catboost import CatBoostRanker, Pool

from .data import SEED
import numpy as np
from .text import top
from .features import ranks, CORE_NAMES, FIELD_NAMES, EMBEDDING_NAMES

TREES = 100
RANKER_WEIGHT = 0.5
PARAMS = dict(
    loss_function="PairLogit",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    thread_count=6,
    allow_writing_files=False,
    verbose=100,
)


def training_pool(blocks):
    x = np.concatenate([block["X"] for block in blocks])
    y = np.concatenate([block["y"] for block in blocks])
    sizes = np.concatenate([np.diff(block["offsets"]) for block in blocks])
    offsets = np.r_[0, np.cumsum(sizes)]
    groups = np.repeat(np.arange(len(sizes)), sizes)
    pairs, weights = balanced_pairs(y, offsets)
    assert np.all(groups[pairs[:, 0]] == groups[pairs[:, 1]])
    names = (CORE_NAMES + FIELD_NAMES + EMBEDDING_NAMES)[: x.shape[1]]
    return Pool(
        x,
        label=y,
        group_id=groups,
        pairs=pairs,
        pairs_weight=weights,
        feature_names=names,
    )


def training_key(blocks, params):
    values = [(b["_key"], b["X"].shape) for b in blocks]
    return hashlib.sha256(
        json.dumps([values, params], sort_keys=True).encode()
    ).hexdigest()


def fit_ranker(blocks, path, bundle, params, *, train=False):
    path = Path(path)
    key = training_key(blocks, params)
    if train:
        model = CatBoostRanker(**params, metadata={"training_key": key})
        model.fit(training_pool(blocks))
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
    else:
        source = path if path.exists() else Path(bundle)
        model = CatBoostRanker().load_model(str(source))
        if model.get_metadata().get("training_key") != key:
            raise ValueError(
                f"Обучающие данные или параметры {source.name} изменились. Включите TRAIN."
            )
    return model


def save_choice(
    model, path, trees, weight, config, *, semantic=True, split_name="holdout"
):
    metadata = model.get_metadata()
    metadata["selected_trees"] = str(int(trees))
    metadata["ranker_weight"] = str(float(weight))
    metadata["search_config"] = json.dumps(config, sort_keys=True)
    metadata["semantic"] = str(int(semantic))
    metadata["evaluation_split"] = split_name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def balanced_pairs(y, offsets):
    """Сумма весов пар каждого непустого запроса равна 192. Пары остаются внутри запроса."""
    pairs, weights = [], []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        positives = np.flatnonzero(y[lo:hi] == 1) + lo
        negatives = np.flatnonzero(y[lo:hi] == 0) + lo
        if not len(positives) or not len(negatives):
            continue
        count = len(positives) * len(negatives)
        pairs.append(
            np.column_stack(
                [
                    np.repeat(positives, len(negatives)),
                    np.tile(negatives, len(positives)),
                ]
            )
        )
        weights.append(np.full(count, 192.0 / count, np.float32))
    return np.concatenate(pairs).astype(np.int32), np.concatenate(weights)


def blend_order(prediction, rrf, weight):
    if weight == 1:
        return top(prediction, min(50, len(prediction)))
    if weight == 0:
        return top(rrf, min(50, len(rrf)))
    score = weight / (60 + ranks(prediction)) + (1 - weight) / (60 + ranks(rrf))
    return top(score, min(50, len(score)))


def recall_at_50(predicted_ids, gold):
    gold = set(gold)
    return len(set(predicted_ids[:50]) & gold) / len(gold) if gold else 0.0


def bootstrap(values, seed=SEED, repeats=3000):
    rng = np.random.default_rng(seed)
    means = np.mean(
        values[rng.integers(0, len(values), size=(repeats, len(values)))], axis=1
    )
    return np.quantile(means, [0.025, 0.975]).tolist()
