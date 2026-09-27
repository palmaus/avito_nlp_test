from dataclasses import dataclass, field

import numpy as np
from scipy import sparse


@dataclass
class GeoModel:
    sources: np.ndarray
    destinations: np.ndarray
    counts: sparse.csr_matrix
    support: np.ndarray
    background: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, labels):
        pairs = labels[
            ["query_norm", "search_location_id", "item_id", "item_location_id"]
        ].copy()
        pairs = pairs.drop_duplicates(["query_norm", "search_location_id", "item_id"])
        sizes = pairs.groupby(["query_norm", "search_location_id"]).item_id.transform(
            "size"
        )
        weights = 1.0 / sizes.to_numpy(dtype=np.float64)
        sources = np.sort(pairs.search_location_id.unique())
        destinations = np.sort(pairs.item_location_id.unique())
        row = np.searchsorted(sources, pairs.search_location_id.to_numpy())
        col = np.searchsorted(destinations, pairs.item_location_id.to_numpy())
        counts = sparse.csr_matrix(
            (weights, (row, col)), shape=(len(sources), len(destinations))
        )
        support = np.asarray(counts.sum(axis=1)).ravel()
        background = np.asarray(counts.sum(axis=0)).ravel() / support.sum()
        return cls(sources, destinations, counts, support, background)

    def destination_positions(self, locations):
        """Для неизвестной локации используется последняя позиция с нулевым значением."""
        values = np.asarray(locations)
        indices = np.searchsorted(self.destinations, values)
        valid = indices < len(self.destinations)
        valid[valid] &= self.destinations[indices[valid]] == values[valid]
        return np.where(valid, indices, len(self.destinations))

    def signal(self, source, alpha=5.0, beta=0.0):
        key = (int(source), float(alpha), float(beta))
        if key in self._cache:
            return self._cache[key]
        result = np.zeros(len(self.destinations) + 1, dtype=np.float32)
        row = int(np.searchsorted(self.sources, source))
        if row < len(self.sources) and self.sources[row] == source:
            distribution = self.counts.getrow(row)
            conditional = distribution.data / self.support[row]
            adjusted = conditional / self.background[distribution.indices] ** beta
            confidence = self.support[row] / (self.support[row] + alpha)
            result[distribution.indices] = adjusted / adjusted.max() * confidence
        # Кэшируем оценки локаций, чтобы не хранить матрицу запросов и объявлений.
        self._cache[key] = result
        return result


def location_factor(
    source, locations, destination_positions, geo, config, source_present
):
    """Бонус за совпадение локации с поправкой по обучающей статистике."""
    locations = np.asarray(locations)
    factor = np.where(locations == source, config.get("location", 8.0), 1.0).astype(
        np.float32
    )
    strength = float(config.get("geo_strength", 0))
    if strength and (config.get("scope", "all") == "all" or not source_present):
        signal = geo.signal(source, config["alpha"], config["beta"])
        factor += strength * signal[destination_positions]
    return factor
