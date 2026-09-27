import argparse
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from src.data import ROOT, SEED, SAVED_SEARCH, corpus, labels, split_data
from src.pipeline import (
    prepare_core,
    add_fields,
    add_embeddings,
    predict,
    export_answer,
)
from src.ranking import PARAMS, fit_ranker, recall_at_50, save_choice


def train(family, rebuild):
    items = corpus()
    _, _, blocks = split_data(labels())
    semantic = family != "text"
    name = "e5_ranker" if semantic else "text_reference"
    filename = {
        "text": "text_57_reference",
        "e5": "ranker",
        "e5-features": "e5_features",
    }[family]
    arrays = []
    for fold, (queries, fitting) in enumerate(blocks):
        folder = f"training/{name}/fold{fold}"
        features = prepare_core(
            items,
            queries,
            fitting,
            SAVED_SEARCH,
            ROOT / "cache" / folder,
            bundle=ROOT / "artifacts" / folder,
            rebuild=rebuild,
            semantic=semantic,
            training=True,
            fold=fold,
        )
        features = add_fields(features, items, queries, rebuild=rebuild)
        if family == "e5-features":
            features = add_embeddings(
                features, items, queries, fitting, SAVED_SEARCH, rebuild=rebuild
            )
        arrays.append(features)
    path = ROOT / "cache/model" / f"{filename}.cbm"
    model = fit_ranker(
        arrays,
        path,
        ROOT / "model" / path.name,
        {**PARAMS, "random_seed": SEED},
        train=True,
    )
    # Число деревьев и смешивание выбраны в notebook по val.
    save_choice(
        model,
        path,
        500 if family == "text" else 100,
        0.5,
        SAVED_SEARCH,
        semantic=semantic,
    )
    return {
        "model": str(path.relative_to(ROOT)),
        "rows": sum(len(a["y"]) for a in arrays),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["predict", "evaluate", "train"])
    parser.add_argument("--family", choices=["text", "e5", "e5-features"], default="e5")
    parser.add_argument(
        "--rebuild", action="store_true", help="Пересчитать кандидатов и признаки"
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--trees", type=int)
    parser.add_argument("--weight", type=float)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "train":
        result = train(args.family, args.rebuild)
    else:
        cached_model = ROOT / "cache/model/answer.cbm"
        path = args.model or (
            cached_model if cached_model.exists() else ROOT / "model/ranker.cbm"
        )
        model = CatBoostRanker().load_model(str(path))
        metadata = dict(model.get_metadata())
        config = (
            json.loads(metadata["search_config"])
            if "search_config" in metadata
            else SAVED_SEARCH
        )
        semantic = metadata.get("semantic", "1") == "1"
        benchmark = args.mode == "predict"
        if benchmark:
            queries = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
            fitting = labels()
        elif metadata.get("evaluation_split") == "text_ranker":
            queries, fitting, _ = split_data(
                labels(), 2026092806, (20260927, 2026092705)
            )
        else:
            queries, fitting, _ = split_data(labels())
        items = corpus(benchmark=benchmark)
        folder = "benchmark" if benchmark else "holdout"
        if not semantic:
            folder += "_text"
        if not benchmark and metadata.get("evaluation_split") == "text_ranker":
            folder = "text_ranker"
        features = prepare_core(
            items,
            queries,
            fitting,
            config,
            ROOT / "cache" / folder,
            bundle=ROOT / "artifacts" / folder,
            rebuild=args.rebuild,
            semantic=semantic,
            tag="benchmark" if benchmark else "local",
        )
        dimensions = len(model.feature_names_)
        if dimensions >= 57:
            features = add_fields(
                features,
                items,
                queries,
                rebuild=args.rebuild,
                tag="benchmark" if benchmark else "local",
            )
        if dimensions == 66:
            features = add_embeddings(
                features,
                items,
                queries,
                fitting,
                config,
                rebuild=args.rebuild,
                tag="benchmark" if benchmark else "local",
            )
        predictions = predict(features, model, args.trees, args.weight)
        if benchmark:
            result = export_answer(queries, items, predictions, ROOT / "answer.csv")
        else:
            ids = items.item_id.to_numpy()
            values = np.array(
                [
                    recall_at_50(ids[p], gold)
                    for p, gold in zip(predictions, queries.gold)
                ]
            )
            result = {
                split: float(values[queries.split.eq(split)].mean())
                for split in ["val", "test"]
            }
    result["seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
