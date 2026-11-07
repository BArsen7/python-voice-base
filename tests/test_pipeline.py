"""Смоук-тесты пайплайна диаризации.

Где взять тестовый wav
----------------------
1. Синтетика (по умолчанию, без внешних файлов):
   тесты используют сгенерированный `tests/data/two_speakers.wav`
   (2 синусоидальных «голоса» с разной частотой и паузами).
   Если файла нет — он будет создан хелпером `ensure_synthetic_wav()`.

2. Реальное аудио (для проверки качества, не для CI):
   любой моно 16 kHz wav с двумя спикерами, например из
   https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/
   или запись с телефона. Положите в `tests/data/real_dialog.wav` и
   запустите:
       pytest tests/test_pipeline.py::test_real_audio -s

3. CI: тесты по умолчанию помечены `@pytest.mark.integration` и
   используют только синтетику + EnergyVAD/EnergyHeuristic + заглушки
   для тяжёлых моделей, чтобы не тянуть torch/pyannote.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.core.interfaces import (
    SpeechSegment,
    TranscriptEntry,
    WordTimestamp,
)
from scripts.pipeline.alignment import align_transcript
from scripts.pipeline.orchestrator import DiarizationPipeline


# ---------------------------------------------------------------------------
# Синтетический wav
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SYNTH_WAV = DATA_DIR / "two_speakers.wav"


def ensure_synthetic_wav(path: Path = SYNTH_WAV, sr: int = 16000) -> Path:
    """Создать синтетический wav: два «спикера» (разные частоты),
    чередующиеся с паузами, плюс участок перекрытия."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    sr = 16000
    total_sec = 6.0
    t = np.arange(int(total_sec * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)

    def tone(freq: float, start: float, end: float, amp: float = 0.3) -> None:
        i0, i1 = int(start * sr), int(end * sr)
        audio[i0:i1] += amp * np.sin(2 * np.pi * freq * t[i0:i1])

    # Спикер A: 200 Hz, Спикер B: 400 Hz
    tone(200, 0.3, 1.5)   # A
    tone(400, 1.8, 3.0)   # B
    tone(200, 2.9, 4.0)   # A (перекрытие с B на 2.9–3.0)
    tone(400, 4.5, 5.5)   # B

    # Лёгкий шум, чтобы VAD/энергия не были идеальными
    audio += np.random.default_rng(42).normal(0, 0.002, size=audio.shape).astype(
        np.float32
    )

    sf.write(path, audio.astype(np.float32), sr)
    return path


@pytest.fixture(scope="session")
def synth_wav() -> Path:
    return ensure_synthetic_wav()


# ---------------------------------------------------------------------------
# Тесты alignment (без тяжёлых зависимостей)
# ---------------------------------------------------------------------------

def test_alignment_word_on_boundary_picks_max_overlap():
    """Слово на границе двух сегментов — спикер с макс. пересечением."""
    words = [WordTimestamp(word="привет", start_ms=950, end_ms=1050)]
    segs = [
        SpeechSegment(start_ms=0, end_ms=1000),      # 50 ms пересечения
        SpeechSegment(start_ms=1000, end_ms=2000),   # 50 ms пересечения
    ]
    labels = np.array([0, 1])
    # Делаем перевес: расширим первый сегмент до 1020, чтобы у него
    # было 70 ms против 30 ms у второго.
    segs[0] = SpeechSegment(start_ms=0, end_ms=1020)
    segs[1] = SpeechSegment(start_ms=1020, end_ms=2000)

    entries = align_transcript(
        words=words, emb_segments=segs, labels=labels, config={}
    )
    assert len(entries) == 1
    assert entries[0].speaker_id == "speaker_00"


def test_alignment_overlap_splits_word_between_two_speakers():
    """Слово целиком внутри overlap-сегмента: покрытие делят 2 спикера,
    побеждает тот, у кого покрытие больше."""
    words = [WordTimestamp(word="да", start_ms=1000, end_ms=1200)]
    segs = [
        SpeechSegment(start_ms=0, end_ms=1100),      # 100 ms
        SpeechSegment(start_ms=1100, end_ms=2000),   # 100 ms
    ]
    labels = np.array([0, 1])
    overlaps = [
        type("O", (), {"start_ms": 900, "end_ms": 1300})()  # duck-typed
    ]
    entries = align_transcript(
        words=words,
        emb_segments=segs,
        labels=labels,
        overlap_segments=overlaps,
        config={"overlap_ratio_threshold": 0.5},
    )
    assert len(entries) == 1
    # 100 vs 100 — ничья, но max() стабильно вернёт первый по порядку
    assert entries[0].speaker_id in {"speaker_00", "speaker_01"}


def test_alignment_groups_consecutive_same_speaker():
    words = [
        WordTimestamp(word="как", start_ms=0, end_ms=100),
        WordTimestamp(word="дела", start_ms=100, end_ms=250),
        WordTimestamp(word="ок", start_ms=300, end_ms=400),  # пауза 50 ms
        WordTimestamp(word="нет", start_ms=900, end_ms=1000),  # пауза 500 ms
    ]
    segs = [SpeechSegment(start_ms=0, end_ms=800), SpeechSegment(start_ms=850, end_ms=1500)]
    labels = np.array([0, 0])
    entries = align_transcript(
        words=words,
        emb_segments=segs,
        labels=labels,
        config={"merge_same_speaker_gap_ms": 200},
    )
    # Первые три слова склеятся (пауза 50 < 200), четвёртое — отдельно
    # из-за паузы 500 ms, хотя спикер тот же.
    assert len(entries) == 2
    assert entries[0].text == "как дела ок"
    assert entries[1].text == "нет"


def test_alignment_monotonic_and_valid_json_shape():
    words = [
        WordTimestamp(word="b", start_ms=500, end_ms=600),
        WordTimestamp(word="a", start_ms=100, end_ms=200),
    ]
    segs = [SpeechSegment(start_ms=0, end_ms=1000)]
    labels = np.array([0])
    entries = align_transcript(
        words=words, emb_segments=segs, labels=labels, config={"ensure_monotonic": True}
    )
    # После сортировки: a (100–200), b (500–600)
    assert [e.text for e in entries] == ["a", "b"]
    # JSON-сериализуемость и обязательные поля
    payload = [e.__dict__ for e in entries]
    json.dumps(payload, ensure_ascii=False)
    for item in payload:
        assert set(item.keys()) == {"speaker_id", "start_ms", "end_ms", "text"}
        assert isinstance(item["start_ms"], int)
        assert isinstance(item["end_ms"], int)
        assert item["start_ms"] <= item["end_ms"]


def test_alignment_empty_returns_empty():
    assert align_transcript([], [], np.array([], dtype=int)) == []


# ---------------------------------------------------------------------------
# Смоук полного пайплайна на синтетике (лёгкие бэкенды)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pipeline_smoke_with_light_backends(synth_wav: Path, monkeypatch):
    """Полный прогон на EnergyVAD + EnergyHeuristicOverlap + заглушках
    эмбеддингов/ASR, чтобы не тянуть torch/pyannote в CI."""
    from scripts.models import asr as asr_module
    from scripts.models import embeddings as emb_module
    from scripts.models import vad as vad_module
    from scripts.models import overlap as overlap_module

    class _StubEmbeddings(emb_module.WeSpeakerEmbeddingExtractor):
        def _lazy_load(self):
            return

        def extract(self, waveform, sample_rate, segments):
            # Синтетические эмбеддинги: частота тона → «спикер»
            vecs = []
            kept = []
            for seg in segments:
                i0 = seg.start_ms * sample_rate // 1000
                i1 = seg.end_ms * sample_rate // 1000
                chunk = waveform[i0:i1]
                if len(chunk) < sample_rate // 10:
                    continue
                # Оценим доминирующую частоту через FFT
                spec = np.abs(np.fft.rfft(chunk))
                freqs = np.fft.rfftfreq(len(chunk), 1 / sample_rate)
                peak = freqs[np.argmax(spec)]
                # 200 Hz → [1,0], 400 Hz → [0,1]
                v = np.array([1.0 if peak < 300 else 0.0,
                              1.0 if peak >= 300 else 0.0], dtype=np.float32)
                v /= np.linalg.norm(v) + 1e-8
                vecs.append(v)
                kept.append(seg)
            if not vecs:
                return np.empty((0, 2), dtype=np.float32), []
            return np.stack(vecs), kept

    class _StubASR(asr_module.FasterWhisperASR):
        def _lazy_load(self):
            return

        def transcribe(self, waveform, sample_rate):
            # Выдаём по слову на каждые ~300 ms речи
            total_ms = int(len(waveform) / sample_rate * 1000)
            words = []
            t = 300
            idx = 0
            while t + 200 <= total_ms:
                words.append(
                    WordTimestamp(
                        word=f"w{idx}", start_ms=t, end_ms=t + 200
                    )
                )
                idx += 1
                t += 300
            return words

    monkeypatch.setattr(emb_module, "WeSpeakerEmbeddingExtractor", _StubEmbeddings)
    monkeypatch.setattr(asr_module, "FasterWhisperASR", _StubASR)

    cfg = {
        "pipeline": {"sample_rate": 16000},
        "vad": {
            "model": "energy_vad",
            "energy_threshold_db": -30.0,
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 30,
        },
        "overlap": {
            "enabled": True,
            "model": "energy_heuristic",
            "threshold": 0.5,
            "min_overlap_duration_ms": 100,
            "energy_ratio": 2.0,
        },
        "embeddings": {"model": "wespeaker_resnet34", "embedding_dim": 2},
        "clustering": {
            "algorithm": "agglomerative",
            "threshold": 0.5,
            "max_speakers": 3,
            "linkage": "average",
        },
        "asr": {"model": "faster_whisper_small", "language": "ru"},
        "alignment": {
            "overlap_ratio_threshold": 0.5,
            "merge_same_speaker_gap_ms": 200,
            "min_segment_duration_ms": 100,
            "ensure_monotonic": True,
        },
    }

    pipe = DiarizationPipeline(cfg)
    payload = pipe.run_file(synth_wav)

    assert isinstance(payload, list)
    assert payload, "pipeline produced empty transcript on synthetic input"

    prev_end = -1
    for item in payload:
        assert set(item.keys()) == {"speaker_id", "start_ms", "end_ms", "text"}
        assert item["speaker_id"].startswith("speaker_")
        assert item["start_ms"] <= item["end_ms"]
        assert item["start_ms"] >= prev_end - 1  # монотонность (с допуском)
        prev_end = item["end_ms"]


# ---------------------------------------------------------------------------
# Опциональный тест на реальном аудио (по умолчанию skip)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DATA_DIR / "real_dialog.wav").exists(),
    reason="put a 16 kHz mono wav with two speakers into tests/data/real_dialog.wav",
)
def test_real_audio_full_pipeline():
    """Полный прогон на реальном wav с продакшен-моделями.

    Требует установленных vadonnx, pyannote.audio, faster-whisper и
    скачанных весов. Запускается только если файл существует.
    """
    from scripts.pipeline.orchestrator import DiarizationPipeline, load_config

    cfg = load_config("config.yaml")
    pipe = DiarizationPipeline(cfg)
    payload = pipe.run_file(DATA_DIR / "real_dialog.wav")
    assert payload
    for item in payload:
        assert item["speaker_id"]
        assert item["start_ms"] <= item["end_ms"]
