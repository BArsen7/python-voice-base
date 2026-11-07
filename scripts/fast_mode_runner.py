#!/usr/bin/env python3
"""
fast_mode_runner.py — Направление 1 (Fast Mode).

Что делает:
  1. Прогоняет WAV через пайплайн (configs/final_threshold.yaml по умолчанию).
  2. Сохраняет транскрипт в JSON в требуемом формате:
        {"transcript": [{"speaker_id", "start_ms", "end_ms", "text"}]}
  3. Регистрирует эмбеддинги спикеров в FAISS-индексе.
  4. При повторном запуске распознаёт известных спикеров по embedding.

Использование:
    python scripts/fast_mode_runner.py input.wav output.json
    python scripts/fast_mode_runner.py input.wav output.json --known-speakers speaker_db/known.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fast_mode")


# ---------------------------------------------------------------------------
# FAISS speaker registry
# ---------------------------------------------------------------------------

class SpeakerRegistry:
    """FAISS-индекс эмбеддингов известных спикеров.

    Формат: L2-нормализованные векторы (IndexFlatIP = косинусное сходство).
    """

    def __init__(self, dim: int = 256, threshold: float = 0.55):
        self.dim = dim
        self.threshold = threshold
        self._index = None
        self._names: List[str] = []
        self._name_to_slot: Dict[str, int] = {}

    def _ensure_index(self):
        if self._index is None:
            try:
                import faiss
            except ImportError as e:
                raise ImportError(
                    "FAISS не установлен. pip install faiss-cpu"
                ) from e
            self._index = faiss.IndexFlatIP(self.dim)

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Добавить эмбеддинг под имя. Если имя уже есть — усреднить."""
        self._ensure_index()
        v = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        # L2-нормализация
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

        if name in self._name_to_slot:
            slot = self._name_to_slot[name]
            old = self._index.reconstruct(slot).reshape(1, -1)
            merged = old + v
            merged = merged / (np.linalg.norm(merged, axis=1, keepdims=True) + 1e-9)
            self._index.remove_ids(np.array([slot], dtype=np.int64))
            self._index.add(merged.astype(np.float32))
            logger.info("Updated embedding for '%s'", name)
            return

        slot = self._index.ntotal
        self._index.add(v.astype(np.float32))
        self._names.append(name)
        self._name_to_slot[name] = slot
        logger.info("Registered new speaker '%s' (slot=%d)", name, slot)

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Найти ближайшего известного спикера.

        Возвращает (name, score) или (None, best_score) если порог не пройден.
        """
        if self._index is None or self._index.ntotal == 0:
            return None, 0.0
        v = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        scores, ids = self._index.search(v.astype(np.float32), 1)
        score = float(scores[0, 0])
        idx = int(ids[0, 0])
        if idx < 0 or score < self.threshold:
            return None, score
        return self._names[idx], score

    def save(self, path: Path) -> None:
        """Сохранить реестр в JSON (эмбеддинги как list[float])."""
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "dim": self.dim,
            "threshold": self.threshold,
            "names": self._names,
            "embeddings": [],
        }
        if self._index is not None and self._index.ntotal > 0:
            for slot in range(self._index.ntotal):
                data["embeddings"].append(
                    self._index.reconstruct(slot).tolist()
                )
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Registry saved: %s (%d speakers)", path, len(self._names))

    @classmethod
    def load(cls, path: Path, threshold: float = 0.55) -> "SpeakerRegistry":
        if not path.exists():
            return cls(threshold=threshold)
        data = json.loads(path.read_text(encoding="utf-8"))
        reg = cls(dim=data.get("dim", 256), threshold=threshold)
        reg._ensure_index()
        for name, emb in zip(data["names"], data["embeddings"]):
            reg.add(name, np.array(emb, dtype=np.float32))
        logger.info("Registry loaded: %s (%d speakers)", path, len(reg._names))
        return reg


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run_fast_mode(
    wav_path: Path,
    config_path: Path,
    known_speakers_path: Optional[Path] = None,
    threshold: float = 0.75,
) -> Dict[str, Any]:
    """Прогон Fast Mode: транскрипт + регистрация спикеров в FAISS."""
    from scripts.pipeline.orchestrator import (
        DiarizationPipeline, load_config, load_wav,
    )

    config = load_config(config_path)
    pipe = DiarizationPipeline(config)

    waveform, sr = load_wav(
        wav_path, target_sr=int(config.get("pipeline", {}).get("sample_rate", 16000))
    )
    logger.info("Audio: %s, %.1f s", wav_path.name, len(waveform) / sr)

    # --- Пайплайн ---
    entries = pipe.run(waveform, sr)
    logger.info("Diarization: %d entries", len(entries))

    # --- Собрать эмбеддинги по спикерам (для FAISS) ---
    # Повторно прогоняем блоки VAD → embeddings, чтобы получить центроиды
    from scripts.core.interfaces import SpeechSegment

    speech_segments = pipe.vad.detect(waveform, sr)
    embeddings, emb_segments = pipe.embeddings.extract(waveform, sr, speech_segments)
    labels = pipe.clustering.fit_predict(embeddings)

    # Центроид на кластер
    speaker_centroids: Dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        vecs = embeddings[labels == label]
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        speaker_centroids[f"speaker_{int(label):02d}"] = centroid.astype(np.float32)

    # --- FAISS: распознать известных ---
    registry = SpeakerRegistry(
        dim=embeddings.shape[1] if embeddings.size else 256,
        threshold=threshold,
    )
    if known_speakers_path is not None and known_speakers_path.exists():
        registry = SpeakerRegistry.load(known_speakers_path, threshold=threshold)

    rename_map: Dict[str, str] = {}
    for auto_id, centroid in speaker_centroids.items():
        known_name, score = registry.identify(centroid)
        if known_name is not None:
            rename_map[auto_id] = known_name
            logger.info(
                "Identified '%s' as known '%s' (score=%.3f)",
                auto_id, known_name, score,
            )
        else:
            rename_map[auto_id] = auto_id
            registry.add(auto_id, centroid)
            logger.info(
                "New speaker '%s' registered (best score=%.3f < %.2f)",
                auto_id, score, threshold,
            )

    # --- Финальный JSON ---
    transcript = []
    for e in entries:
        transcript.append({
            "speaker_id": rename_map.get(e.speaker_id, e.speaker_id),
            "start_ms": int(e.start_ms),
            "end_ms": int(e.end_ms),
            "text": str(e.text).strip(),
        })
    transcript.sort(key=lambda x: (x["start_ms"], x["end_ms"]))

    return {
        "transcript": transcript,
        "_meta": {
            "source": str(wav_path),
            "duration_s": round(len(waveform) / sr, 2),
            "n_speakers": len(speaker_centroids),
            "known_speakers": sorted(
                {v for k, v in rename_map.items() if v != k}
            ),
        },
    }, registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", type=Path, help="Входной WAV")
    ap.add_argument("out_json", type=Path, help="Куда сохранить транскрипт")
    ap.add_argument("--config", type=Path, default=Path("configs/final_threshold.yaml"))
    ap.add_argument("--known-speakers", type=Path, default=Path("speaker_db/known.json"))
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="Порог cosine-similarity для опознания известного спикера")
    ap.add_argument("--save-speakers", type=Path, default=None,
                    help="Куда сохранить обновлённый реестр (по умолчанию --known-speakers)")
    args = ap.parse_args()

    payload, registry = run_fast_mode(
        args.wav, args.config,
        known_speakers_path=args.known_speakers,
        threshold=args.threshold,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Transcript saved: %s", args.out_json)

    save_path = args.save_speakers or args.known_speakers
    registry.save(save_path)

    # Краткая сводка
    print(json.dumps({"transcript": payload["transcript"]}, ensure_ascii=False, indent=2))
    print(f"\nTranscript entries: {len(payload['transcript'])}")
    for e in payload["transcript"][:3]:
        print(f"  [{e['speaker_id']}] {e['start_ms']:>5}–{e['end_ms']:<5} {e['text']!r}")


if __name__ == "__main__":
    main()
