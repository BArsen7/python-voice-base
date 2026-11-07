"""Блок 5: извлечение эмбеддингов спикера.

- WeSpeakerEmbeddingExtractor — через pyannote.audio PretrainedSpeakerEmbedding
  (обёртка над wespeaker-voxceleb-resnet34-LM).
- ECAPAEmbeddingExtractor — через SpeechBrain (speechbrain/spkrec-ecapa-voxceleb).

Оба возвращают L2-нормализованные векторы формы (N, embedding_dim).
"""
from __future__ import annotations

from pathlib import Path

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

        use_onnx = bool(self.config.get("use_onnx", False))
        onnx_path = self.config.get("onnx_path")

        if use_onnx:
            if not onnx_path:
                raise ValueError(
                    "embeddings.use_onnx=true requires embeddings.onnx_path"
                )
            if not Path(onnx_path).exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            try:
                from pyannote.audio.pipelines.speaker_verification import (
                    ONNXWeSpeakerPretrainedSpeakerEmbedding,
                )
            except ImportError as e:
                raise ImportError(
                    "ONNX path requires pyannote.audio >= 3.3"
                ) from e
            self._model = ONNXWeSpeakerPretrainedSpeakerEmbedding(onnx_path)
            logger.info("WeSpeaker ResNet34 ONNX loaded from %s", onnx_path)
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
        logger.info("WeSpeaker ResNet34 embedding model loaded (pyannote)")

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


def extract_windowed_embeddings(extractor, waveform, sample_rate,
                                window_sec=2.5, hop_sec=1.0,
                                min_remainder_sec=0.5):
    """
    Извлечение эмбеддингов спикера на перекрывающихся окнах.

    Для Accuracy Mode окно 2.5 с даёт более стабильный вектор спикера,
    чем покадровое извлечение с шагом 1 с.

    Parameters
    ----------
    extractor : object
        Экстрактор с методом `extract(waveform, sample_rate)` или
        `compute_embedding(waveform, sample_rate)`.
    waveform : np.ndarray
        Аудио (моно, float32).
    sample_rate : int
    window_sec : float
        Длина окна, сек (рекомендуется 2.0-3.0).
    hop_sec : float
        Шаг между окнами, сек.
    min_remainder_sec : float
        Минимальная длина хвоста — если короче, отбрасываем.

    Returns
    -------
    List[Tuple[int, int, np.ndarray]]
        Список (start_ms, end_ms, embedding).
    """
    n = len(waveform)
    window_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)
    min_remainder_samples = int(min_remainder_sec * sample_rate)

    def _call(chunk):
        for attr in ('extract', 'compute_embedding', 'compute'):
            fn = getattr(extractor, attr, None)
            if callable(fn):
                try:
                    emb = fn(chunk, sample_rate)
                except TypeError:
                    try:
                        emb = fn(chunk)
                    except TypeError:
                        continue
                if emb is not None:
                    return np.asarray(emb, dtype=np.float32).reshape(-1)
        raise AttributeError(
            "extractor has no compatible extract/compute_embedding method"
        )

    out = []
    if n < window_samples:
        emb = _call(waveform)
        if emb is not None and emb.size > 0:
            out.append((0, int(n / sample_rate * 1000), emb))
        return out

    for start in range(0, n - window_samples + 1, hop_samples):
        chunk = waveform[start:start + window_samples]
        emb = _call(chunk)
        if emb is not None and emb.size > 0:
            out.append((
                int(start / sample_rate * 1000),
                int((start + window_samples) / sample_rate * 1000),
                emb,
            ))

    last_end = ((n - window_samples) // hop_samples) * hop_samples + window_samples
    remainder = n - last_end
    if remainder >= min_remainder_samples:
        chunk = waveform[last_end:]
        emb = _call(chunk)
        if emb is not None and emb.size > 0:
            out.append((
                int(last_end / sample_rate * 1000),
                int(n / sample_rate * 1000),
                emb,
            ))
    return out


# === SPEED_MODE_EMBEDDER ===
# Appended by install_embedder.sh. Lightweight ONNX speaker embedder for
# Speed Mode — uses sherpa-onnx SpeakerEmbeddingExtractor (no torch, ~28 MB).
import sherpa_onnx as _sm_sherpa  # noqa: E402

_SM_EMB_SR = 16000


class SherpaOnnxEmbedder(BaseEmbeddingExtractor):
    """Speed-mode speaker embedder: sherpa-onnx Campplus (~28 MB, ~5 ms/segment).

    Loads in ~200 ms, uses ONNX Runtime only. Replaces WeSpeakerEmbeddingExtractor
    (which pulls torch/pyannote) for low-latency scenarios.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get(
            "model_path",
            "models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        )
        self.num_threads = int(config.get("num_threads", 1))
        self.provider = config.get("provider", "cpu")
        self._extractor = None

    def _lazy_load(self):
        if self._extractor is not None:
            return
        # sherpa-onnx >= 1.10 API: pass a config object, not kwargs.
        config = _sm_sherpa.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=self.num_threads,
            debug=False,
            provider=self.config.get("provider", "cpu"),
        )
        self._extractor = _sm_sherpa.SpeakerEmbeddingExtractor(config)
        logger.info("SherpaOnnxEmbedder loaded: %s", self.model_path)

    def extract(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        segments: Sequence[SpeechSegment],
    ) -> Tuple[np.ndarray, List[SpeechSegment]]:
        self._lazy_load()
        if sample_rate != _SM_EMB_SR:
            raise ValueError(f"SherpaOnnxEmbedder expects {_SM_EMB_SR} Hz, got {sample_rate}")

        embs: List[np.ndarray] = []
        valid: List[SpeechSegment] = []

        # Two calling conventions:
        #   (a) streaming: orchestrator passes per-segment audio (waveform is
        #       exactly one segment). We must NOT slice by absolute ms.
        #   (b) offline: caller passes the whole recording + segments with
        #       absolute timestamps; then we slice per segment.
        #
        # Heuristic: if the waveform length matches the segment duration
        # (within 100 ms) for every segment, treat it as per-segment.
        def _is_per_segment_waveform(wf: np.ndarray, segs) -> bool:
            if not segs:
                return True
            wf_s = len(wf) / _SM_EMB_SR
            for s in segs:
                dur_s = (s.end_ms - s.start_ms) / 1000.0
                if abs(wf_s - dur_s) > 0.1:
                    return False
            return True

        per_segment = _is_per_segment_waveform(waveform, segments)

        for seg in segments:
            audio = getattr(seg, "audio", None)
            if audio is None:
                if per_segment and len(segments) == 1:
                    audio = waveform
                else:
                    start = int(seg.start_ms * _SM_EMB_SR / 1000)
                    end = int(seg.end_ms * _SM_EMB_SR / 1000)
                    audio = waveform[start:end]
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size < _SM_EMB_SR // 10:
                continue

            stream = self._extractor.create_stream()
            stream.accept_waveform(_SM_EMB_SR, audio)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                continue
            emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n > 0:
                emb = emb / n
            embs.append(emb)
            valid.append(seg)

        if not embs:
            dim = getattr(self._extractor, "dim", 192)
            return np.zeros((0, dim), dtype=np.float32), []
        return np.stack(embs, axis=0), valid
