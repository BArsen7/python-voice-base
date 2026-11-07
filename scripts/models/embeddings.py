"""Блок 5: извлечение эмбеддингов спикера.

- WeSpeakerEmbeddingExtractor — через pyannote.audio PretrainedSpeakerEmbedding
  (обёртка над wespeaker-voxceleb-resnet34-LM).
- ECAPAEmbeddingExtractor — через SpeechBrain (speechbrain/spkrec-ecapa-voxceleb).

Оба возвращают L2-нормализованные векторы формы (N, embedding_dim).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from scripts.core.interfaces import (
    BaseEmbeddingExtractor,
    SpeechSegment,
)

logger = logging.getLogger(__name__)


class WeSpeakerEmbeddingExtractor(BaseEmbeddingExtractor):
    """WeSpeaker ResNet34 через pyannote.audio."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dim = int(config.get("embedding_dim", 256))
        self.window_sec = float(config.get("window_sec", 1.5))
        self.hop_sec = float(config.get("hop_sec", 0.75))
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                "WeSpeakerEmbeddingExtractor requires pyannote.audio. "
                "Install with: pip install pyannote.audio"
            ) from e

        self._model = PretrainedSpeakerEmbedding(
            "hbredin/wespeaker-voxceleb-resnet34-LM"
        )
        logger.info("WeSpeaker ResNet34 embedding model loaded")

    def extract(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        segments: Sequence[SpeechSegment],
    ) -> Tuple[np.ndarray, List[SpeechSegment]]:
        self._lazy_load()

        import torch

        windows: List[np.ndarray] = []
        window_segments: List[SpeechSegment] = []

        for seg in segments:
            seg_len_ms = seg.duration_ms
            win_ms = int(self.window_sec * 1000)
            hop_ms = int(self.hop_sec * 1000)

            if seg_len_ms <= win_ms:
                # Сегмент короче окна — берём целиком (с минимальной длиной)
                audio = _slice_waveform(waveform, sample_rate, seg.start_ms, seg.end_ms)
                if len(audio) < int(0.1 * sample_rate):
                    continue
                windows.append(audio)
                window_segments.append(seg)
            else:
                # Нарезаем на перекрывающиеся окна
                pos = seg.start_ms
                while pos + win_ms <= seg.end_ms:
                    audio = _slice_waveform(waveform, sample_rate, pos, pos + win_ms)
                    windows.append(audio)
                    window_segments.append(
                        SpeechSegment(start_ms=pos, end_ms=pos + win_ms)
                    )
                    pos += hop_ms
                # Хвост
                tail_start = seg.end_ms - win_ms
                if tail_start > seg.start_ms and tail_start > pos - hop_ms:
                    audio = _slice_waveform(
                        waveform, sample_rate, tail_start, seg.end_ms
                    )
                    windows.append(audio)
                    window_segments.append(
                        SpeechSegment(start_ms=tail_start, end_ms=seg.end_ms)
                    )

        if not windows:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        # Батчевый инференс: pyannote ожидает (batch, 1, samples)
        max_len = max(len(w) for w in windows)
        batch = np.zeros((len(windows), 1, max_len), dtype=np.float32)
        lengths = []
        for i, w in enumerate(windows):
            batch[i, 0, : len(w)] = w
            lengths.append(len(w))

        tensor = torch.from_numpy(batch)
        with torch.no_grad():
            embeddings = self._model(tensor)  # (N, D)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        # L2-нормализация
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        return embeddings, window_segments


class ECAPAEmbeddingExtractor(BaseEmbeddingExtractor):
    """ECAPA-TDNN через SpeechBrain."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dim = int(config.get("embedding_dim", 192))
        self.window_sec = float(config.get("window_sec", 1.5))
        self.hop_sec = float(config.get("hop_sec", 0.75))
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "ECAPAEmbeddingExtractor requires speechbrain. "
                "Install with: pip install speechbrain"
            ) from e

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        logger.info("ECAPA-TDNN embedding model loaded (SpeechBrain)")

    def extract(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        segments: Sequence[SpeechSegment],
    ) -> Tuple[np.ndarray, List[SpeechSegment]]:
        self._lazy_load()

        import torch

        windows: List[np.ndarray] = []
        window_segments: List[SpeechSegment] = []

        for seg in segments:
            seg_len_ms = seg.duration_ms
            win_ms = int(self.window_sec * 1000)
            hop_ms = int(self.hop_sec * 1000)

            if seg_len_ms <= win_ms:
                audio = _slice_waveform(waveform, sample_rate, seg.start_ms, seg.end_ms)
                if len(audio) < int(0.1 * sample_rate):
                    continue
                windows.append(audio)
                window_segments.append(seg)
            else:
                pos = seg.start_ms
                while pos + win_ms <= seg.end_ms:
                    audio = _slice_waveform(waveform, sample_rate, pos, pos + win_ms)
                    windows.append(audio)
                    window_segments.append(
                        SpeechSegment(start_ms=pos, end_ms=pos + win_ms)
                    )
                    pos += hop_ms
                tail_start = seg.end_ms - win_ms
                if tail_start > seg.start_ms and tail_start > pos - hop_ms:
                    audio = _slice_waveform(
                        waveform, sample_rate, tail_start, seg.end_ms
                    )
                    windows.append(audio)
                    window_segments.append(
                        SpeechSegment(start_ms=tail_start, end_ms=seg.end_ms)
                    )

        if not windows:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        max_len = max(len(w) for w in windows)
        batch = np.zeros((len(windows), max_len), dtype=np.float32)
        for i, w in enumerate(windows):
            batch[i, : len(w)] = w

        tensor = torch.from_numpy(batch)
        with torch.no_grad():
            embeddings = self._model.encode_batch(tensor)  # (N, 1, D)

        embeddings = embeddings.squeeze(1).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        return embeddings, window_segments


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _slice_waveform(
    waveform: np.ndarray, sample_rate: int, start_ms: int, end_ms: int
) -> np.ndarray:
    i0 = max(0, int(start_ms * sample_rate / 1000))
    i1 = min(len(waveform), int(end_ms * sample_rate / 1000))
    return waveform[i0:i1]
