import re

import numpy as np

from .text import normalize, stem_text, top
from .geography import location_factor
from .data import FIELDS

CORE_NAMES = [
    "stem_log_score",
    "stem_relative_score",
    "stem_raw_rr",
    "stem_geo_rr",
    "stem_present",
    "char_score",
    "char_relative_score",
    "char_raw_rr",
    "char_geo_rr",
    "char_present",
    "search_rrf",
    "rrf_rank",
    "both_channels",
    "same_location",
    "geo_signal",
    "geo_log_factor",
    "source_present_in_corpus",
    "source_log_support",
    "destination_log_background",
    "prior_probability",
    "prior_relative",
    "query_prior_max",
    "query_prior_entropy",
    "item_rating",
    "rating_missing",
    "log_reviews",
    "reviews_missing",
    "same_category",
    "is_service_category",
    "query_category_specified",
    "query_words",
    "query_characters",
    "query_oov_fraction",
    "has_filters",
    "delivery_search",
]
FIELD_NAMES = []
for field in FIELDS:
    FIELD_NAMES += [
        f"{field}_log_bm25",
        f"{field}_normalized_bm25",
        f"{field}_query_coverage",
        f"{field}_log_length",
        f"{field}_phrase_match",
    ]
FIELD_NAMES += [
    "params_filter_log_bm25",
    "params_filter_coverage",
    "required_rating",
    "rating_filter_match",
    "rating_filter_gap",
    "query_has_numbers",
    "number_coverage",
]
FEATURE_NAMES = CORE_NAMES + FIELD_NAMES
EMBEDDING_NAMES = [
    "e5_cosine",
    "e5_cosine_gap",
    "e5_rank",
    "e5_geo_score",
    "e5_geo_rank",
    "e5_present",
    "e5_geo_present",
    "text_present",
    "hybrid_rrf",
]


def ranks(scores):
    order = top(scores, len(scores))
    result = np.empty(len(scores), np.int32)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def union_candidates(pools, index, n_items):
    parts = []
    for ids, scores in pools.values():
        valid = (ids[index] >= 0) & (scores[index] > 0)
        parts.append(ids[index, valid])
    result = np.unique(np.concatenate(parts))
    if len(result) < 50:
        result = np.union1d(result, np.arange(min(n_items, 50)))
    return result.astype(np.int32)


def core_features(query, candidate_ids, pools, index, item_data, prior, geo, config):
    source = query.search_location_id
    source_present = source in item_data["location_set"]
    factors = location_factor(
        source,
        item_data["item_location_id"][candidate_ids],
        item_data["geo_positions"][candidate_ids],
        geo,
        config,
        source_present,
    )
    signal = geo.signal(source, config["alpha"], config["beta"])[
        item_data["geo_positions"][candidate_ids]
    ]
    columns, reciprocal, membership = [], [], []
    for channel in ["stem", "char"]:
        ids, raw = pools[channel]
        valid = ids[index] >= 0
        ids, raw = ids[index, valid], raw[index, valid]
        loc_factors = location_factor(
            source,
            item_data["item_location_id"][ids],
            item_data["geo_positions"][ids],
            geo,
            config,
            source_present,
        )
        rr_raw = 1.0 / (60 + ranks(raw))
        category_factor = np.where(
            item_data["item_category_id"][ids] == 114, config.get("service", 1), 1
        )
        rr_geo = 1.0 / (60 + ranks(raw * loc_factors * category_factor))
        positive = raw > 0
        # Для отсутствующего кандидата оценка равна нулю; наличие хранится отдельно.
        positions = np.searchsorted(candidate_ids, ids)
        in_union = positions < len(candidate_ids)
        in_union[in_union] &= candidate_ids[positions[in_union]] == ids[in_union]
        active = positive & in_union
        score = np.zeros(len(candidate_ids), np.float32)
        raw_rank = np.zeros(len(candidate_ids), np.float64)
        geo_rank = np.zeros(len(candidate_ids), np.float64)
        present = np.zeros(len(candidate_ids), np.float32)
        score[positions[active]] = raw[active]
        raw_rank[positions[active]] = rr_raw[active]
        geo_rank[positions[active]] = rr_geo[active]
        present[positions[active]] = 1
        relative = score / max(float(raw.max(initial=0)), 1e-12)
        columns += [
            np.log1p(score) if channel == "stem" else score,
            relative,
            raw_rank,
            geo_rank,
            present,
        ]
        reciprocal.append(geo_rank)
        membership.append(present)
    p = prior[item_data["prior_positions"][candidate_ids]]
    pmax = float(prior.max(initial=0))
    normalized_prior = p / max(pmax, 1e-12)
    rrf = sum(
        config["channels"].get(channel, 0) * score
        for channel, score in zip(["stem", "char"], reciprocal)
    )
    rrf *= 1 + config["prior"] * normalized_prior
    row = int(np.searchsorted(geo.sources, source))
    support = (
        geo.support[row] if row < len(geo.sources) and geo.sources[row] == source else 0
    )
    background = np.r_[geo.background, 0][item_data["geo_positions"][candidate_ids]]
    prior_entropy = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
    words = stem_text(query.search_query).split()
    vocabulary = item_data["vocabulary"]
    unique = set(words)
    oov = 1 - sum(word in vocabulary for word in unique) / max(len(unique), 1)
    rating = item_data["rating"][candidate_ids]
    reviews = item_data["reviews"][candidate_ids]

    def constant(value):
        return np.full(len(candidate_ids), value, np.float32)

    columns += [
        rrf,
        1.0 / (60 + ranks(rrf)),
        membership[0] * membership[1],
        (item_data["item_location_id"][candidate_ids] == source).astype(np.float32),
        signal,
        np.log1p(factors),
        constant(source_present),
        constant(np.log1p(support)),
        np.log1p(1_000_000 * background),
        p,
        normalized_prior,
        constant(pmax),
        constant(prior_entropy),
        rating,
        np.isnan(rating).astype(np.float32),
        np.log1p(np.maximum(np.nan_to_num(reviews), 0)),
        np.isnan(reviews).astype(np.float32),
        (item_data["item_category_id"][candidate_ids] == query.search_category).astype(
            np.float32
        ),
        (item_data["item_category_id"][candidate_ids] == 114).astype(np.float32),
        constant(query.search_category != 0),
        constant(len(words)),
        constant(len(normalize(query.search_query))),
        constant(oov),
        constant(bool(str(query.search_infm_params_text).strip())),
        constant(query.search_is_delivery_search),
    ]
    result = np.column_stack(columns).astype(np.float32)
    assert result.shape[1] == len(CORE_NAMES)
    return result, rrf


def training_selection(candidate_ids, rrf, item_ids, gold, known_positives, rng):
    """Не добавляем пропущенные позитивы и не берём известные позитивы в негативы."""
    candidate_items = item_ids[candidate_ids]
    positive = np.flatnonzero(np.isin(candidate_items, list(gold)))
    negative = np.flatnonzero(~np.isin(candidate_items, list(known_positives)))
    if not len(positive) or not len(negative):
        return np.array([], np.int32)
    ordered = negative[top(rrf[negative], len(negative))]
    hard = ordered[:128]
    remainder = ordered[128:]
    random = rng.choice(remainder, min(64, len(remainder)), replace=False)
    return np.unique(np.r_[positive, hard, random]).astype(np.int32)


def field_features(query, candidate_ids, matrices, stats, texts, vocabulary, ratings):
    stems = set(stem_text(query.search_query).split())
    terms = sorted(vocabulary[word] for word in stems if word in vocabulary)
    query_norm = normalize(query.search_query)
    filter_norm = normalize(query.search_infm_params_text)
    rating_match = re.search(r"рейтинг пользователя\s+(\d+(?:[.,]\d+)?)", filter_norm)
    required = float(rating_match.group(1).replace(",", ".")) if rating_match else 0.0
    lexical_filter = re.sub(
        r"рейтинг пользователя\s+\d+(?:[.,]\d+)?\s*(?:звезд[а-я]*)?\s*(?:и выше)?",
        "",
        filter_norm,
    )
    filter_stems = set(stem_text(lexical_filter).split())
    filter_terms = sorted(
        vocabulary[word] for word in filter_stems if word in vocabulary
    )
    columns = []
    for field in FIELDS:
        subset = matrices[field][candidate_ids][:, terms]
        values = np.asarray(subset.sum(axis=1)).ravel()
        denominator = max(float(stats[field]["idf"][terms].sum()), 1e-12)
        coverage = subset.getnnz(axis=1) / max(len(stems), 1)
        phrase = np.fromiter(
            (
                bool(query_norm) and query_norm in texts[field][index]
                for index in candidate_ids
            ),
            dtype=np.float32,
            count=len(candidate_ids),
        )
        columns += [
            np.log1p(values),
            values / denominator,
            coverage,
            np.log1p(stats[field]["lengths"][candidate_ids]),
            phrase,
        ]
    subset = matrices["params"][candidate_ids][:, filter_terms]
    fvalues = np.asarray(subset.sum(axis=1)).ravel()
    filter_coverage = subset.getnnz(axis=1) / max(len(filter_stems), 1)
    ratings = ratings[candidate_ids]
    known_rating = np.isfinite(ratings)
    rating_indicator = np.zeros(len(candidate_ids), np.float32)
    gap = np.zeros(len(candidate_ids), np.float32)
    if required:
        rating_indicator[known_rating] = np.where(
            ratings[known_rating] >= required, 1, -1
        )
        gap = ratings - required
    numbers = set(re.findall(r"\d+", query_norm))
    if numbers:
        number_coverage = np.array(
            [
                len(numbers & texts["numbers"][index]) / len(numbers)
                for index in candidate_ids
            ],
            np.float32,
        )
    else:
        number_coverage = np.zeros(len(candidate_ids), np.float32)
    columns += [
        np.log1p(fvalues),
        filter_coverage,
        np.full(len(candidate_ids), required),
        rating_indicator,
        gap,
        np.full(len(candidate_ids), bool(numbers)),
        number_coverage,
    ]
    result = np.column_stack(columns).astype(np.float32)
    assert result.shape[1] == len(FIELD_NAMES)
    return result


def semantic_score(cosine, factors, config):
    return cosine + config["e5_geo"] * np.log(factors)


def rank_column(candidate_ids, ranked_ids):
    """Обратная позиция в точном top-500, ноль для остальных кандидатов."""
    values = np.zeros(len(candidate_ids), np.float32)
    positions = np.searchsorted(candidate_ids, ranked_ids)
    valid = positions < len(candidate_ids)
    valid[valid] &= candidate_ids[positions[valid]] == ranked_ids[valid]
    values[positions[valid]] = 1.0 / (60 + np.arange(1, len(ranked_ids) + 1)[valid])
    return values


def embedding_features(
    candidate_ids,
    lexical_ids,
    query_vector,
    vectors,
    neighbors,
    i,
    factors,
    lexical_rrf,
    prior_relative,
    config,
):
    cosine = vectors[candidate_ids] @ query_vector
    raw_rank = rank_column(candidate_ids, neighbors["global_ids"][i])
    geo_rank = rank_column(candidate_ids, neighbors["geo_ids"][i])
    in_lexical = np.isin(candidate_ids, lexical_ids)
    hybrid = lexical_rrf + config["e5_weight"] * geo_rank * (
        1 + config["prior"] * prior_relative
    )
    values = np.column_stack(
        [
            cosine,
            cosine - neighbors["global_scores"][i, 0],
            raw_rank,
            semantic_score(cosine, factors, config),
            geo_rank,
            raw_rank > 0,
            geo_rank > 0,
            in_lexical,
            hybrid,
        ]
    ).astype(np.float32)
    return values, hybrid, in_lexical
