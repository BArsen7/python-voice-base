"""Блок 4: детекция перекрытий речи.

- NullOverlapDetector — заглушка, если overlap отключён.
- EnergyHeuristicOverlapDetector — эвристика на основе энергии и VAD.
- PyannoteOverlapDetector — pyannote/overlapped-speech-detection (опционально).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.core.interfaces import (
    BaseOverlapDetector,
    OverlapSegment,
    SpeechSegment,
)

logger = logging.getLogger(__name__)


class NullOverlapDetector(BaseOverlapDetector):
    """Overlap отключён — всегда пустой список."""

    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_segments: Sequence[SpeechSegment],
    ) -> List[OverlapSegment]:
        return []


class EnergyHeuristicOverlapDetector(BaseOverlapDetector):
    """Простая эвристика: участок считается перекрытием, если его энергия
    значительно выше средней энергии одиночной речи.

    Это не настоящая диаризация перекрытий, но даёт сигнал для alignment,
    когда в одном временном окне энергия «слишком высокая» для одного голоса.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = float(config.get("threshold", 0.5))
        self.min_overlap_ms = int(config.get("min_overlap_duration_ms", 100))
        self.frame_ms = int(config.get("frame_ms", 30))
        # Во сколько раз энергия должна превысить медиану речи
        self.energy_ratio = float(config.get("energy_ratio", 2.0))

    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_segments: Sequence[SpeechSegment],
    ) -> List[OverlapSegment]:
        if len(speech_segments) < 2:
            return []

        frame_len = max(1, int(sample_rate * self.frame_ms / 1000))
        n_frames = len(waveform) // frame_len
        if n_frames == 0:
            return []

        frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.mean(frames**2, axis=1) + 1e-12

        # Эталон: медианная энергия внутри речевых сегментов
        speech_mask = np.zeros(n_frames, dtype=bool)
        for seg in speech_segments:
            i0 = seg.start_ms * sample_rate // 1000 // frame_len
            i1 = seg.end_ms * sample_rate // 1000 // frame_len
            speech_mask[i0:i1] = True

        if not speech_mask.any():
            return []

        baseline = float(np.median(energy[speech_mask]))
        if baseline <= 0:
            return []

        ratio = energy / baseline
        is_overlap = (ratio > self.energy_ratio) & speech_mask

        # Собираем сегменты перекрытия
        overlaps: List[OverlapSegment] = []
        i = 0
        while i < n_frames:
            if is_overlap[i]:
                j = i
                while j < n_frames and is_overlap[j]:
                    j += 1
                start_ms = int(i * frame_len / sample_rate * 1000)
                end_ms = int(j * frame_len / sample_rate * 1000)
                if end_ms - start_ms >= self.min_overlap_ms:
                    # "вероятность" = нормированное превышение энергии
                    mean_ratio = float(np.mean(ratio[i:j]))
                    prob = min(1.0, mean_ratio / (self.energy_ratio * 2))
                    overlaps.append(
                        OverlapSegment(start_ms, end_ms, probability=prob)
                    )
                i = j
            else:
                i += 1

        logger.info("Energy heuristic overlap: %d segments", len(overlaps))
        return overlaps


class PyannoteOverlapDetector(BaseOverlapDetector):
    """pyannote/overlapped-speech-detection через pyannote.audio Pipeline."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = float(config.get("threshold", 0.5))
        self.min_overlap_ms = int(config.get("min_overlap_duration_ms", 100))
        self._pipeline = None
        self._hf_token = config.get("hf_token")  # опционально

    def _lazy_load(self):
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise ImportError(
                "PyannoteOverlapDetector requires pyannote.audio. "
                "Install with: pip install pyannote.audio"
            ) from e

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=self._hf_token,
        )
        logger.info("pyannote overlapped-speech-detection loaded")

    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_segments: Sequence[SpeechSegment],
    ) -> List[OverlapSegment]:
        self._lazy_load()

        import torch

        # pyannote ожидает (batch, channel, samples) и 16 kHz
        tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
        if sample_rate != 16000:
            # pyannote Pipeline умеет ресемплить, но лучше подавать 16k
            raise ValueError("PyannoteOverlapDetector expects 16 kHz audio")

        output = self._pipeline(
            {"waveform": tensor, "sample_rate": sample_rate}
        )
        # output — pyannote.core.Annotation с метками "overlap"
        overlaps: List[OverlapSegment] = []
        for segment, _, label in output.itertracks(yield_label=True):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            if end_ms - start_ms < self.min_overlap_ms:
                continue
            overlaps.append(
                OverlapSegment(start_ms, end_ms, probability=self.threshold)
            )
        return overlaps
