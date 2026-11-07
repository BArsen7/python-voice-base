"""Speed Mode streaming orchestrator.

Audio chunk -> StreamingEnergyVAD -> [SpeechSegment+audio]
            -> SherpaOnnxZipformerSmallASR -> words
            -> BaseEmbeddingExtractor.extract(wave, sr, [seg]) -> (N,D)
            -> OnlineThresholdClustering.assign(...) -> speaker_id
            -> TranscriptEntry (yielded immediately)
"""
from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

import numpy as np

from scripts.core.interfaces import TranscriptEntry
from scripts.models.asr import SherpaOnnxZipformerSmallASR
from scripts.models.clustering import OnlineThresholdClustering
from scripts.models.vad import StreamingEnergyVAD

logger = logging.getLogger(__name__)


class SpeedModeOrchestrator:
    def __init__(
        self,
        asr: SherpaOnnxZipformerSmallASR,
        vad: StreamingEnergyVAD,
        clustering: OnlineThresholdClustering,
        embedder,
        on_entry: Optional[Callable[[TranscriptEntry], None]] = None,
    ) -> None:
        self.asr = asr
        self.vad = vad
        self.clustering = clustering
        self.embedder = embedder
        self.on_entry = on_entry
        self._entries: List[TranscriptEntry] = []

    # ------------------------------------------------------------------
    def push_chunk(self, chunk: np.ndarray, sample_rate: int = 16000) -> List[TranscriptEntry]:
        new_entries: List[TranscriptEntry] = []
        for seg, seg_audio in self.vad.push_with_audio(chunk):
            entry = self._process_segment(seg, seg_audio, sample_rate)
            if entry is not None:
                new_entries.append(entry)
                self._entries.append(entry)
                if self.on_entry:
                    self.on_entry(entry)
        return new_entries

    def flush(self, sample_rate: int = 16000) -> List[TranscriptEntry]:
        out: List[TranscriptEntry] = []
        pair = self.vad.flush_with_audio()
        if pair is not None:
            seg, seg_audio = pair
            entry = self._process_segment(seg, seg_audio, sample_rate)
            if entry is not None:
                out.append(entry)
                self._entries.append(entry)
                if self.on_entry:
                    self.on_entry(entry)
        return out

    # ------------------------------------------------------------------
    def _process_segment(self, seg, seg_audio: np.ndarray, sample_rate: int):
        if seg_audio.size == 0:
            return None

        words = self.asr.transcribe(seg_audio, sample_rate)
        text = " ".join(w.word for w in words).strip() if words else ""
        if not text:
            return None

        speaker_id = "unknown"
        try:
            # BaseEmbeddingExtractor contract:
            #   extract(waveform, sample_rate, segments) -> (embeddings, valid_segments)
            embeddings, valid_segments = self.embedder.extract(
                seg_audio, sample_rate, [seg]
            )
            if embeddings is not None and len(embeddings) > 0:
                seg_dur_ms = seg.end_ms - seg.start_ms
                speaker_id = self.clustering.assign(embeddings[0], duration_ms=seg_dur_ms)
        except Exception as e:
            logger.warning("embedder failed, tagging as 'unknown': %s", e)

        return TranscriptEntry(
            speaker_id=speaker_id,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            text=text,
        )

    # ------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "transcript": [
                {
                    "speaker_id": e.speaker_id,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                    "text": e.text,
                }
                for e in self._entries
            ]
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=2)
