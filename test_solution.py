import unittest
import numpy as np
import pandas as pd
from src.geography import GeoModel, location_factor
from src.features import training_selection, union_candidates
from src.ranking import balanced_pairs, blend_order, recall_at_50


class GeographyTests(unittest.TestCase):
    def labels(self):
        return pd.DataFrame(
            [
                ("a", 10, "x", 1),
                ("a", 10, "y", 2),
                ("b", 10, "z", 2),
                ("c", 20, "u", 1),
                ("d", 20, "v", 1),
            ],
            columns=["query_norm", "search_location_id", "item_id", "item_location_id"],
        )

    def test_context_balance_and_duplicates(self):
        labels = self.labels()
        original = GeoModel.fit(labels)
        repeated = GeoModel.fit(pd.concat([labels, labels.iloc[[0] * 100]]))
        np.testing.assert_allclose(original.counts.toarray(), [[0.5, 1.5], [2, 0]])
        np.testing.assert_allclose(original.counts.toarray(), repeated.counts.toarray())

    def test_unknown_and_shrinkage(self):
        model = GeoModel.fit(self.labels())
        self.assertTrue(np.all(model.signal(999) == 0))
        self.assertEqual(model.signal(10, 2, 0)[1], 0.5)
        positions = model.destination_positions([1, 2, 999])
        self.assertEqual(model.signal(10, 2, 0)[positions[-1]], 0)

    def test_regional_search_can_prefer_different_id(self):
        model = GeoModel.fit(self.labels())
        locations = np.array([1, 2, 99])
        factor = location_factor(
            10,
            locations,
            model.destination_positions(locations),
            model,
            {"alpha": 2, "beta": 0, "geo_strength": 32},
            False,
        )
        self.assertGreater(factor[1], factor[0])
        self.assertGreater(factor[0], factor[2])

    def test_missing_scope_keeps_exact_location_bonus_for_present_source(self):
        model = GeoModel.fit(self.labels())
        locations = np.array([10, 2, 99])
        factor = location_factor(
            10,
            locations,
            model.destination_positions(locations),
            model,
            {"alpha": 2, "beta": 0, "geo_strength": 32, "scope": "missing"},
            True,
        )
        np.testing.assert_array_equal(factor, [8, 1, 1])


class RankingTests(unittest.TestCase):
    def test_empty_search_with_category_prior(self):
        from src.retrieval import search

        items = pd.DataFrame(
            {
                "item_location_id": np.ones(60, dtype=int),
                "item_category_id": np.full(60, 114),
                "item_microcat_id": np.full(60, 7),
            }
        )
        queries = pd.DataFrame({"search_location_id": [2]})
        candidates = {
            channel: (np.array([[0, 1, 2]]), np.zeros((1, 3), dtype=np.float32))
            for channel in ["stem", "char"]
        }
        predictions = search(
            items,
            queries,
            candidates,
            {"channels": {"stem": 1, "char": 0.5}, "location": 8, "prior": 0.5},
            [0],
            prior=(np.array([[0.4, 0]], dtype=np.float32), np.array([7])),
        )
        np.testing.assert_array_equal(predictions[0], np.arange(50))

    def test_other_context_positive_is_not_a_negative(self):
        ids = np.arange(6)
        items = np.array(["a", "b", "c", "d", "e", "f"])
        keep = training_selection(
            ids, np.arange(6.0), items, {"a"}, {"a", "b"}, np.random.default_rng(1)
        )
        self.assertIn(0, keep)
        self.assertNotIn(1, keep)
        self.assertEqual(set(keep), {0, 2, 3, 4, 5})

    def test_missing_positive_is_not_injected(self):
        keep = training_selection(
            np.arange(3),
            np.arange(3.0),
            np.array(["a", "b", "c"]),
            {"unretrieved"},
            {"unretrieved"},
            np.random.default_rng(1),
        )
        self.assertEqual(len(keep), 0)

    def test_query_pair_weights_and_boundaries(self):
        y = np.array([1, 0, 1, 1, 0, 0, 0])
        offsets = np.array([0, 2, 2, 7])
        pairs, weights = balanced_pairs(y, offsets)
        self.assertTrue(np.all(y[pairs[:, 0]] == 1))
        self.assertTrue(np.all(y[pairs[:, 1]] == 0))
        self.assertEqual(set(map(tuple, pairs[:1])), {(0, 1)})
        self.assertTrue(np.all(pairs[1:] >= 2))
        self.assertAlmostEqual(float(weights[:1].sum()), 192)
        self.assertAlmostEqual(float(weights[1:].sum()), 192)

    def test_full_gold_denominator_and_unique_hits(self):
        self.assertEqual(
            recall_at_50(["a", "a", "x"], {"a", "b", "outside_pool"}), 1 / 3
        )
        self.assertEqual(recall_at_50(["x"] * 50 + ["a"], {"a"}), 0)

    def test_blend_endpoints_and_stable_ties(self):
        np.testing.assert_array_equal(
            blend_order(np.array([0.0, 1, 1]), np.array([3.0, 2, 1]), 1), [1, 2, 0]
        )
        np.testing.assert_array_equal(
            blend_order(np.array([0.0, 1, 1]), np.array([3.0, 2, 1]), 0), [0, 1, 2]
        )

    def test_empty_pool_has_deterministic_fallback_without_labels(self):
        pools = {
            "stem": (np.array([[-1, -1]]), np.zeros((1, 2))),
            "char": (np.array([[4, 5]]), np.zeros((1, 2))),
        }
        np.testing.assert_array_equal(union_candidates(pools, 0, 100), np.arange(50))


class EmbeddingTests(unittest.TestCase):
    def test_deduplication_and_resume(self):
        import tempfile
        from pathlib import Path
        from src.encoder import embeddings

        class FakeEncoder:
            signature = "test-encoder-v1"

            def __init__(self, fail=False):
                self.calls = 0
                self.fail = fail

            def encode(self, texts, prefix):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError("interrupted batch")
                result = np.zeros((len(texts), 384), np.float32)
                result[:, 0] = [int(t) for t in texts]
                return result

        texts = [str(i) for i in range(4200)] + ["0", "1"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.npy"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                embeddings(texts, "passage: ", path, FakeEncoder(fail=True))
            encoder = FakeEncoder()
            result = embeddings(texts, "passage: ", path, encoder)
            self.assertEqual(encoder.calls, 1)
            np.testing.assert_array_equal(result[:, 0], np.array(texts, dtype=float))
            cached = embeddings(texts, "passage: ", path, encoder)
            self.assertEqual(encoder.calls, 1)
            np.testing.assert_array_equal(cached, result)


class SavedFeaturesTests(unittest.TestCase):
    def test_saved_features_roundtrip_and_input_order(self):
        import tempfile
        from pathlib import Path
        from src.artifacts import save_features, load_features
        from src.data import SEARCH

        features = {
            "X": np.arange(4 * 66, dtype=np.float32).reshape(4, 66),
            "ids": np.array([0, 2, 1, 3]),
            "rrf": np.array([0.1, 0.2, 0.3, 0.4]),
            "text_mask": np.array([True, False, True, False]),
            "offsets": np.array([0, 2, 4]),
        }
        items = pd.DataFrame({"item_id": ["a", "b", "c", "d"]})
        queries = pd.DataFrame(
            [["q1", "ремонт", 1, False, "", 114], ["q2", "доставка", 2, True, "", 114]],
            columns=["query_id", *SEARCH],
        )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_features(folder, features, items, queries, "settings-a")
            loaded = load_features(folder, items, queries)
            for name in features:
                np.testing.assert_array_equal(loaded[name], features[name])
            with self.assertRaisesRegex(ValueError, "объявлений"):
                load_features(folder, items.iloc[::-1], queries)
            with self.assertRaisesRegex(ValueError, "Запросы"):
                load_features(folder, items, queries.iloc[::-1])
            with self.assertRaisesRegex(ValueError, "настройки"):
                load_features(folder, items, queries, "settings-b")
            changed = queries.copy()
            changed.loc[0, "search_location_id"] = 3
            with self.assertRaisesRegex(ValueError, "Запросы"):
                load_features(folder, items, changed)


class CacheTests(unittest.TestCase):
    def test_encoder_change_invalidates_vectors(self):
        import tempfile
        from pathlib import Path
        from src.encoder import embeddings

        class Encoder:
            def __init__(self, value):
                self.signature = str(value)
                self.value = value
                self.calls = 0

            def encode(self, texts, prefix):
                self.calls += 1
                return np.full((len(texts), 384), self.value, np.float32)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.npy"
            first, second = Encoder(1), Encoder(2)
            embeddings(["один текст"], "query: ", path, first)
            result = embeddings(["один текст"], "query: ", path, second)
            self.assertEqual(second.calls, 1)
            self.assertEqual(result[0, 0], 2)
            with self.assertRaisesRegex(ValueError, "encoder"):
                embeddings(["один текст"], "query: ", path, first, rebuild=False)

    def test_collect_refreshes_same_named_catalogue(self):
        import tempfile
        from pathlib import Path
        from src.retrieval import collect

        items = pd.DataFrame(
            {
                "item_title_raw": ["ремонт", "маникюр"],
                "item_infm_params_text": ["", ""],
                "item_description_raw": ["", ""],
                "item_location_id": [1, 1],
                "item_category_id": [114, 114],
            }
        )
        queries = pd.DataFrame({"search_query": ["ремонт"], "search_location_id": [1]})
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            a, av = collect(items, queries, "word", cache)
            items["item_title_raw"] = ["маникюр", "ремонт"]
            b, bv = collect(items, queries, "word", cache)
            self.assertEqual(a[0, av[0].argmax()], 0)
            self.assertEqual(b[0, bv[0].argmax()], 1)

    def test_retrained_model_is_reloaded_and_checks_settings(self):
        import tempfile
        from pathlib import Path
        from src.ranking import fit_ranker, save_choice

        rng = np.random.default_rng(2)
        arrays = [
            {
                "X": rng.normal(size=(12, 35)),
                "y": np.tile([1.0, 0.0, 0.0], 4),
                "offsets": np.arange(0, 13, 3),
                "_key": "first",
            }
        ]
        params = dict(
            loss_function="PairLogit",
            iterations=3,
            depth=2,
            random_seed=7,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.cbm"
            model = fit_ranker(arrays, path, "missing-bundle.cbm", params, train=True)
            save_choice(
                model,
                path,
                2,
                0.75,
                {"channels": {"stem": 1}},
                semantic=False,
                split_name="text_ranker",
            )
            loaded = fit_ranker(arrays, path, "missing-bundle.cbm", params)
            self.assertEqual(loaded.get_metadata()["evaluation_split"], "text_ranker")
            self.assertEqual(loaded.get_metadata()["selected_trees"], "2")
            self.assertEqual(loaded.get_metadata()["semantic"], "0")
            np.testing.assert_array_equal(
                model.predict(arrays[0]["X"]), loaded.predict(arrays[0]["X"])
            )
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                fit_ranker(arrays, path, "missing-bundle.cbm", {**params, "depth": 3})

    def test_predict_loads_extra_columns_and_keeps_query_order(self):
        import tempfile
        from pathlib import Path
        from catboost import CatBoostRanker
        from src.features import CORE_NAMES, FIELD_NAMES, EMBEDDING_NAMES
        from src.artifacts import add_columns
        from src.pipeline import predict
        from src.ranking import blend_order

        rng = np.random.default_rng(8)
        x = rng.normal(size=(90, 66)).astype(np.float32)
        model = CatBoostRanker(
            iterations=4,
            depth=2,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x,
            (x[:, 57] > 0).astype(float),
            group_id=np.repeat(np.arange(3), 30),
        )
        model.set_feature_names(CORE_NAMES + FIELD_NAMES + EMBEDDING_NAMES)
        with tempfile.TemporaryDirectory() as folder:
            features = {
                "X": x[:, :57],
                "rrf": rng.random(90),
                "ids": np.arange(90),
                "offsets": np.array([0, 30, 60, 90]),
                "_key": "a",
                "_folder": Path(folder),
            }
            add_columns(features, "embedding_features", x[:, 57:])
            predictions = predict(features, model, trees=4, weight=0.5, indices=[2, 0])
            score = model.predict(x)
            for pred, i in zip(predictions, [2, 0]):
                a, b = features["offsets"][i : i + 2]
                expected = features["ids"][a:b][
                    blend_order(score[a:b], features["rrf"][a:b], 0.5)
                ]
                np.testing.assert_array_equal(pred, expected)


if __name__ == "__main__":
    unittest.main()
