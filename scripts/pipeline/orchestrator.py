"""Оркестратор пайплайна диаризации и транскрибации.

Собирает компоненты на основе config.yaml и прогоняет аудио через
блоки 3–8. На выходе — JSON-совместимый список TranscriptEntry.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.core.interfaces import (
    BaseASR,
    BaseClustering,
    BaseEmbeddingExtractor,
    BaseOverlapDetector,
    BaseVAD,
    TranscriptEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ленивые импорты конкретных реализаций (чтобы не тянуть тяжёлые зависимости,
# если модель не выбрана в конфиге).
# ---------------------------------------------------------------------------

def _build_vad(cfg: Dict[str, Any]) -> BaseVAD:
    model = cfg.get("model", "silero_vad")
    if model == "silero_vad":
        from src.models.vad import SileroVAD
        return SileroVAD(cfg)
    if model == "energy_vad":
        from src.models.vad import EnergyVAD
        return EnergyVAD(cfg)
    raise ValueError(f"Unknown VAD model: {model}")


def _build_overlap(cfg: Dict[str, Any]) -> BaseOverlapDetector:
    if not cfg.get("enabled", True):
        from src.models.overlap import NullOverlapDetector
        return NullOverlapDetector(cfg)
    model = cfg.get("model", "energy_heuristic")
    if model == "pyannote_segmentation":
        from src.models.overlap import PyannoteOverlapDetector
        return PyannoteOverlapDetector(cfg)
    if model == "energy_heuristic":
        from src.models.overlap import EnergyHeuristicOverlapDetector
        return EnergyHeuristicOverlapDetector(cfg)
    raise ValueError(f"Unknown overlap model: {model}")


def _build_embeddings(cfg: Dict[str, Any]) -> BaseEmbeddingExtractor:
    model = cfg.get("model", "wespeaker_resnet34")
    if model == "wespeaker_resnet34":
        from src.models.embeddings import WeSpeakerEmbeddingExtractor
        return WeSpeakerEmbeddingExtractor(cfg)
    if model == "ecapa_tdnn":
        from src.models.embeddings import ECAPAEmbeddingExtractor
        return ECAPAEmbeddingExtractor(cfg)
    raise ValueError(f"Unknown embedding model: {model}")


def _build_clustering(cfg: Dict[str, Any]) -> BaseClustering:
    algo = cfg.get("algorithm", "agglomerative")
    if algo == "agglomerative":
        from src.models.clustering import AgglomerativeClusteringWrapper
        return AgglomerativeClusteringWrapper(cfg)
    if algo == "spectral":
        from src.models.clustering import SpectralClusteringWrapper
        return SpectralClusteringWrapper(cfg)
    raise ValueError(f"Unknown clustering algorithm: {algo}")


def _build_asr(cfg: Dict[str, Any]) -> BaseASR:
    model = cfg.get("model", "faster_whisper_small")
    if model == "faster_whisper_small":
        from src.models.asr import FasterWhisperASR
        return FasterWhisperASR(cfg)
    if model == "sherpa_onnx_zipformer":
        from src.models.asr import SherpaOnnxASR
        return SherpaOnnxASR(cfg)
    raise ValueError(f"Unknown ASR model: {model}")


# ---------------------------------------------------------------------------
# Оркестратор
# ---------------------------------------------------------------------------

class DiarizationPipeline:
    """Полный пайплайн: VAD → Overlap → Embeddings → Clustering → ASR → Alignment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        p = config.get("pipeline", {})
        self.sample_rate = int(p.get("sample_rate", 16000))

        self.vad = _build_vad(config.get("vad", {}))
        self.overlap = _build_overlap(config.get("overlap", {}))
        self.embeddings = _build_embeddings(config.get("embeddings", {}))
        self.clustering = _build_clustering(config.get("clustering", {}))
        self.asr = _build_asr(config.get("asr", {}))

        logger.info(
            "Pipeline initialized: VAD=%s, Overlap=%s, Emb=%s, Clust=%s, ASR=%s",
            type(self.vad).__name__,
            type(self.overlap).__name__,
            type(self.embeddings).__name__,
            type(self.clustering).__name__,
            type(self.asr).__name__,
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def run_file(self, audio_path: str | Path) -> List[Dict[str, Any]]:
        """Загрузить wav, прогнать пайплайн, вернуть JSON-ready список."""
        waveform, sr = load_wav(audio_path, target_sr=self.sample_rate)
        entries = self.run(waveform, sr)
        return [e.__dict__ for e in entries]

    def run(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[TranscriptEntry]:
        """Основной метод: waveform (float32, mono) → TranscriptEntry list."""
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # моно
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample_rate={self.sample_rate}, got {sample_rate}"
            )

        # --- Блок 3: VAD ---
        speech_segments = self.vad.detect(waveform, sample_rate)
        logger.info("VAD: %d segments", len(speech_segments))

        # --- Блок 4: Overlap ---
        overlap_segments = self.overlap.detect(
            waveform, sample_rate, speech_segments
        )
        logger.info("Overlap: %d segments", len(overlap_segments))

        # --- Блок 5: Embeddings ---
        if not speech_segments:
            logger.warning("No speech segments — returning empty transcript")
            return []
        emb, emb_segments = self.embeddings.extract(
            waveform, sample_rate, speech_segments
        )
        logger.info("Embeddings: shape=%s", emb.shape)

        # --- Блок 6: Clustering ---
        labels = self.clustering.fit_predict(emb)
        logger.info("Clustering: %d clusters", len(set(labels.tolist())))

        # --- Блок 7: ASR ---
        words = self.asr.transcribe(waveform, sample_rate)
        logger.info("ASR: %d words", len(words))

        # --- Блок 8: Alignment ---
        from src.pipeline.alignment import align_transcript

        entries = align_transcript(
            words=words,
            emb_segments=emb_segments,
            labels=labels,
            overlap_segments=overlap_segments,
            config=self.config.get("alignment", {}),
        )
        logger.info("Alignment: %d transcript entries", len(entries))
        return entries


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wav(
    path: str | Path, target_sr: int = 16000
) -> tuple[np.ndarray, int]:
    """Загрузка wav через soundfile. Возвращает (float32 mono, sample_rate)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        # простой ресемплинг через scipy (для прототипа достаточно)
        from scipy.signal import resample_poly

        ratio = target_sr / sr
        up, down = ratio.as_integer_ratio() if isinstance(ratio, float) else (1, 1)
        # более надёжно через gcd
        from math import gcd

        g = gcd(target_sr, sr)
        up, down = target_sr // g, sr // g
        data = resample_poly(data, up, down).astype(np.float32)
        sr = target_sr
    return data, sr


def save_transcript(entries: List[TranscriptEntry], path: str | Path) -> None:
    """Сохранить TranscriptEntry list в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [e.__dict__ for e in entries], f, ensure_ascii=False, indent=2
        )
