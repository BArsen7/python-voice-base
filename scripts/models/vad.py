"""Блок 3: VAD-детекторы.

SileroVAD работает через ONNX Runtime (пакет vadonnx, который бандлит
Silero-модель). EnergyVAD — простая энергетическая эвристика для отладки
и как fallback на устройствах без ONNX Runtime.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.core.interfaces import BaseVAD, SpeechSegment

logger = logging.getLogger(__name__)


class SileroVAD(BaseVAD):
    """Silero VAD через vadonnx (ONNX Runtime, offline)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = float(config.get("threshold", 0.5))
        self.min_speech_ms = int(config.get("min_speech_duration_ms", 250))
        self.speech_pad_ms = int(config.get("speech_pad_ms", 30))
        self.window_size = int(config.get("window_size_samples", 512))
        self._vad = None  # lazy

    def _lazy_load(self):
        if self._vad is not None:
            return
        try:
            from vadonnx import load_vad

            # "silero" — бандлится в wheel, работает полностью offline
            self._vad = load_vad("silero")
            logger.info("Silero VAD loaded via vadonnx (ONNX Runtime)")
        except ImportError as e:
            raise ImportError(
                "SileroVAD requires `vadonnx`. Install with: pip install vadonnx"
            ) from e

    def detect(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[SpeechSegment]:
        self._lazy_load()

        if sample_rate != 16000:
            raise ValueError("Silero VAD expects 16 kHz audio")

        # vadonnx.get_speech_segments принимает numpy float32 / int16
        raw_segments = self._vad.get_speech_segments(
            waveform, sample_rate=sample_rate
        )

        # Приводим к SpeechSegment и применяем padding / min_duration
        segments: List[SpeechSegment] = []
        total_ms = int(len(waveform) / sample_rate * 1000)
        for seg in raw_segments:
            # vadonnx возвращает start/end в СЕКУНДАХ
            start_ms = int(seg.start * 1000) - self.speech_pad_ms
            end_ms = int(seg.end * 1000) + self.speech_pad_ms
            start_ms = max(0, start_ms)
            end_ms = min(total_ms, end_ms)
            if end_ms - start_ms < self.min_speech_ms:
                continue
            segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))

        # Склейка пересекающихся после padding
        return _merge_segments(segments, gap_ms=0)


class EnergyVAD(BaseVAD):
    """Простой энергетический VAD — fallback без ONNX Runtime.

    Полезен для быстрой проверки пайплайна и на устройствах, где
    Silero-модель недоступна. Качество ниже, но зависимости — только numpy.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold_db = float(config.get("energy_threshold_db", -35.0))
        self.min_speech_ms = int(config.get("min_speech_duration_ms", 250))
        self.min_silence_ms = int(config.get("min_silence_duration_ms", 100))
        self.speech_pad_ms = int(config.get("speech_pad_ms", 30))
        self.frame_ms = int(config.get("frame_ms", 20))

    def detect(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[SpeechSegment]:
        frame_len = max(1, int(sample_rate * self.frame_ms / 1000))
        n_frames = len(waveform) // frame_len
        if n_frames == 0:
            return []

        # RMS по кадрам → dB
        frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)

        voiced = db > self.threshold_db

        # Морфология: закрываем короткие паузы, убираем короткие всплески
        voiced = _close_gaps(voiced, max_gap_frames=self.min_silence_ms // self.frame_ms)
        voiced = _remove_short(voiced, min_len_frames=self.min_speech_ms // self.frame_ms)

        # Собираем сегменты
        segments: List[SpeechSegment] = []
        total_ms = int(len(waveform) / sample_rate * 1000)
        start_frame = None
        for i, v in enumerate(voiced):
            if v and start_frame is None:
                start_frame = i
            elif not v and start_frame is not None:
                start_ms = int(start_frame * frame_len / sample_rate * 1000)
                end_ms = int(i * frame_len / sample_rate * 1000)
                start_ms = max(0, start_ms - self.speech_pad_ms)
                end_ms = min(total_ms, end_ms + self.speech_pad_ms)
                if end_ms - start_ms >= self.min_speech_ms:
                    segments.append(SpeechSegment(start_ms, end_ms))
                start_frame = None
        if start_frame is not None:
            start_ms = int(start_frame * frame_len / sample_rate * 1000)
            end_ms = total_ms
            if end_ms - start_ms >= self.min_speech_ms:
                segments.append(SpeechSegment(start_ms, end_ms))

        return _merge_segments(segments, gap_ms=0)


# ---------------------------------------------------------------------------
# Утилиты для VAD
# ---------------------------------------------------------------------------

def _merge_segments(
    segments: Sequence[SpeechSegment], gap_ms: int = 0
) -> List[SpeechSegment]:
    """Склеить пересекающиеся/близкие сегменты."""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s.start_ms)
    merged = [sorted_segs[0]]
    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg.start_ms <= last.end_ms + gap_ms:
            merged[-1] = SpeechSegment(
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, seg.end_ms),
                confidence=max(last.confidence, seg.confidence),
            )
        else:
            merged.append(seg)
    return merged


def _close_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Заменить False→True в коротких паузах между True-участками."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            gap = j - i
            if i > 0 and j < n and gap <= max_gap_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _remove_short(mask: np.ndarray, min_len_frames: int) -> np.ndarray:
    """Убрать True-участки короче min_len_frames."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_len_frames:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out
