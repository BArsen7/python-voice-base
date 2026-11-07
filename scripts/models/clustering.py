"""Блок 6: кластеризация эмбеддингов в спикеров.

- AgglomerativeClusteringWrapper — иерархическая кластеризация с косинусным
  расстоянием и порогом distance_threshold (или фиксированным n_clusters).
- SpectralClusteringWrapper — спектральная кластеризация (через sklearn).

Оба возвращают целочисленные метки кластеров; -1 означает «шум/не удалось».
"""
from __future__ import annotations

import logging
from typing import List, Any, Dict

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
    """
    Спектральная кластеризация на графе ближайших соседей.

    Accuracy Mode: affinity="nearest_neighbors" даёт более устойчивое
    разбиение на сложных пересечениях спикеров, чем косинусное сходство
    на разреженных эмбеддингах.

    Для вырожденных входов (малое N или n_clusters >= N-1) спектральный
    путь неустойчив: sklearn вызывает eigsh(k=n_components+1), и при
    k >= N падает с "Cannot use scipy.linalg.eigh for sparse A".
    В таких случаях используем AgglomerativeClustering (тот же контракт).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_speakers = int(config.get("max_speakers", 10))
        self.n_clusters = config.get("n_clusters")
        self.n_neighbors = int(config.get("n_neighbors", 10))
        self.assign_labels = config.get("assign_labels", "kmeans")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([], dtype=int)
        if len(embeddings) == 1:
            return np.array([0], dtype=int)

        n = len(embeddings)
        n_clusters = (
            int(self.n_clusters)
            if self.n_clusters is not None
            else self._auto_n_clusters(embeddings)
        )

        # Вырожденные случаи, где spectral падает по k >= N.
        if n < 4 or n_clusters >= n - 1:
            return self._fit_predict_agglomerative(embeddings, n_clusters)

        from sklearn.cluster import SpectralClustering

        # Гарантируем k < N для eigsh: n_components + 1 < N.
        n_clusters = max(2, min(n_clusters, n - 2))
        n_neighbors = max(2, min(self.n_neighbors, n - 2))

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels=self.assign_labels,
            random_state=42,
            n_jobs=-1,
        )
        labels = model.fit_predict(embeddings)
        logger.info(
            "Spectral(nearest_neighbors): %d clusters (N=%d, n_neighbors=%d)",
            len(np.unique(labels)), n, n_neighbors,
        )
        return labels.astype(int)

    def _auto_n_clusters(self, embeddings: np.ndarray) -> int:
        """Подбор k=2..max_speakers по максимуму silhouette (cosine).

        Нужен, когда n_clusters не задан явно: без этого спектральный
        путь по умолчанию берёт max_speakers кластеров и «дробит»
        небольшое число реальных спикеров.
        """
        n = len(embeddings)
        k_max = min(self.max_speakers, n - 2)
        if k_max < 2:
            return min(self.max_speakers, n)

        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score

        n_neighbors = max(2, min(self.n_neighbors, n - 2))
        best_k, best_score = 2, -1.0
        for k in range(2, k_max + 1):
            try:
                m = SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    assign_labels=self.assign_labels,
                    random_state=42,
                    n_jobs=-1,
                )
                labels = m.fit_predict(embeddings)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(embeddings, labels, metric="cosine")
                if score > best_score:
                    best_k, best_score = k, score
            except Exception as exc:
                logger.debug("auto-k: k=%d пропущен (%s)", k, exc)

        logger.info(
            "auto-k: selected n_clusters=%d (silhouette=%.3f, N=%d, k_max=%d)",
            best_k, best_score, n, k_max,
        )
        return best_k

    def _fit_predict_agglomerative(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Робастный fallback для малых N и вырожденных n_clusters."""
        from sklearn.cluster import AgglomerativeClustering

        n = len(embeddings)
        k = max(1, min(int(n_clusters), n))
        if k <= 1:
            return np.zeros(n, dtype=int)

        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = agg.fit_predict(embeddings)
        logger.info(
            "Spectral fallback -> Agglomerative: %d clusters (N=%d)", k, n
        )
        return labels.astype(int)


# === SPEED_MODE_CLUSTERING ===
# Appended by apply_speed_mode_v2.sh — do not delete this marker.
from dataclasses import dataclass as _sm_dataclass  # noqa: E402


@_sm_dataclass
class _SpeakerCentroid:
    speaker_id: str
    centroid: np.ndarray
    n_segments: int = 0

    def update(self, emb: np.ndarray) -> None:
        self.n_segments += 1
        alpha = max(0.05, 1.0 / self.n_segments)
        self.centroid = (1 - alpha) * self.centroid + alpha * emb
        n = float(np.linalg.norm(self.centroid))
        if n > 0:
            self.centroid = self.centroid / n


class OnlineThresholdClustering(BaseClustering):
    """O(N*K) online cosine-threshold speaker clustering for Speed Mode.

    Compatible with BaseClustering.fit_predict() — processes embeddings
    sequentially (streaming logic), returns an int label array.
    Also exposes assign(embedding) -> str for real-time streaming.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = float(config.get("similarity_threshold", 0.65))
        self.max_speakers = int(config.get("max_speakers", 3))
        # Short segments create noisy centroids; only allow new speaker
        # creation from segments at least this long.
        self.min_new_speaker_duration_ms = int(
            config.get("min_new_speaker_duration_ms", 2000)
        )
        self._centroids: List[_SpeakerCentroid] = []
        self._next_id = 0

    # --- BaseClustering API ---------------------------------------------
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([], dtype=int)
        self.reset()
        ids: List[int] = []
        for emb in embeddings:
            sid = self.assign(emb)
            ids.append(int(sid.split("_")[-1]))
        return np.asarray(ids, dtype=int)

    # --- Streaming API --------------------------------------------------
    def reset(self) -> None:
        self._centroids.clear()
        self._next_id = 0

    def assign(self, embedding: np.ndarray, duration_ms: int | None = None) -> str:
        emb = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n == 0.0:
            raise ValueError("embedding has zero norm")
        emb = emb / n

        is_short = (
            duration_ms is not None
            and duration_ms < self.min_new_speaker_duration_ms
        )

        # No centroids yet: only a long-enough segment may seed the first
        # speaker. Short segments before that are tagged "unknown".
        if not self._centroids:
            if is_short:
                return "unknown"
            return self._create(emb)

        sims = [float(np.dot(c.centroid, emb)) for c in self._centroids]
        best = int(np.argmax(sims))

        if sims[best] >= self.threshold:
            self._centroids[best].update(emb)
            return self._centroids[best].speaker_id

        if len(self._centroids) < self.max_speakers and not is_short:
            return self._create(emb)

        # Max speakers reached and no confident match.
        # Long segment -> real new voice: force-assign AND train the centroid
        #                 (alpha = max(0.05, 1/n) keeps this gentle and adaptive).
        # Short segment -> likely noise: force-assign, no update.
        if not is_short:
            self._centroids[best].update(emb)
        return self._centroids[best].speaker_id

    def _create(self, emb: np.ndarray) -> str:
        sid = f"spk_{self._next_id}"
        self._next_id += 1
        self._centroids.append(_SpeakerCentroid(speaker_id=sid, centroid=emb.copy(),
                                                n_segments=1))
        return sid
