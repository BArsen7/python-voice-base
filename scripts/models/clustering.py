"""Блок 6: кластеризация эмбеддингов в спикеров.

- AgglomerativeClusteringWrapper — иерархическая кластеризация с косинусным
  расстоянием и порогом distance_threshold (или фиксированным n_clusters).
- SpectralClusteringWrapper — спектральная кластеризация (через sklearn).

Оба возвращают целочисленные метки кластеров; -1 означает «шум/не удалось».
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from scripts.core.interfaces import BaseClustering

logger = logging.getLogger(__name__)


class AgglomerativeClusteringWrapper(BaseClustering):
    """Agglomerative clustering с косинусным расстоянием.

    Если в конфиге задан `threshold` (косинусное расстояние, 1 - cos_sim),
    используем distance_threshold. Иначе — фиксированное `max_speakers`
    (или n_clusters, если передан).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.75)
        self.max_speakers = int(config.get("max_speakers", 10))
        self.linkage = config.get("linkage", "average")
        self.n_clusters = config.get("n_clusters")  # опционально

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([], dtype=int)
        if len(embeddings) == 1:
            return np.array([0], dtype=int)

        from sklearn.cluster import AgglomerativeClustering

        # Косинусное расстояние через precomputed матрицу
        # (sklearn корректно работает с cosine и для Agglomerative)
        if self.n_clusters is not None:
            model = AgglomerativeClustering(
                n_clusters=int(self.n_clusters),
                metric="cosine",
                linkage=self.linkage,
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(self.threshold),
                metric="cosine",
                linkage=self.linkage,
            )

        labels = model.fit_predict(embeddings)

        # Ограничение max_speakers: если кластеров больше — принудительно
        # уменьшаем n_clusters (грубо, но для прототипа достаточно).
        unique = np.unique(labels)
        if len(unique) > self.max_speakers:
            logger.warning(
                "Clustering produced %d speakers > max_speakers=%d; "
                "re-running with n_clusters=%d",
                len(unique),
                self.max_speakers,
                self.max_speakers,
            )
            model = AgglomerativeClustering(
                n_clusters=self.max_speakers,
                metric="cosine",
                linkage=self.linkage,
            )
            labels = model.fit_predict(embeddings)

        logger.info("Agglomerative: %d clusters", len(np.unique(labels)))
        return labels.astype(int)


class SpectralClusteringWrapper(BaseClustering):
    """Spectral clustering по косинусному сходству."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_speakers = int(config.get("max_speakers", 10))
        self.n_clusters = config.get("n_clusters")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([], dtype=int)
        if len(embeddings) == 1:
            return np.array([0], dtype=int)

        from sklearn.cluster import SpectralClustering

        n_clusters = int(self.n_clusters) if self.n_clusters is not None else min(
            self.max_speakers, len(embeddings)
        )
        # SpectralClustering с косинусным affinity
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="cosine",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = model.fit_predict(embeddings)
        logger.info("Spectral: %d clusters", len(np.unique(labels)))
        return labels.astype(int)
