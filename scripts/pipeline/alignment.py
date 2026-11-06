"""Блок 8: слияние word-level таймкодов ASR с диаризацией.

Алгоритм:
  1. Для каждого слова ASR находим все пересекающиеся сегменты диаризации
     (emb_segments + labels), считаем долю перекрытия.
  2. Победитель — сегмент с максимальным пересечением, но не меньше
     overlap_ratio_threshold * длины слова. Если ни один сегмент не
     дотягивает до порога, берём ближайший по времени (граничный случай).
  3. Слова внутри overlap-сегментов помечаем: если в пересечении
     участвуют ≥2 спикера — оставляем наиболее «весомого» (max overlap),
     это и есть обработка перекрытий на уровне слова.
  4. Группируем слова в TranscriptEntry: одинаковый спикер + пауза
     ≤ merge_same_speaker_gap_ms → одна запись.
  5. Фильтр min_segment_duration_ms и (опционально) ensure_monotonic.

Важно: один и тот же спикер может иметь несколько кластеров (например,
спикер A = кластеры 0 и 3). Для прототипа считаем, что каждый кластер —
отдельный спикер. В README отмечено, как это улучшить (консолидация
кластеров по центроидам).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.interfaces import (
    OverlapSegment,
    SpeechSegment,
    TranscriptEntry,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Публичный вход
# ---------------------------------------------------------------------------

def align_transcript(
    words: Sequence[WordTimestamp],
    emb_segments: Sequence[SpeechSegment],
    labels: np.ndarray,
    overlap_segments: Sequence[OverlapSegment] | None = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[TranscriptEntry]:
    """Основная функция слияния.

    Параметры
    ---------
    words : список слов ASR с таймкодами (ms)
    emb_segments : сегменты, для которых извлекались эмбеддинги
    labels : массив меток кластеров той же длины, что emb_segments
    overlap_segments : участки перекрытия (ms), могут быть пустыми
    config : секция alignment из config.yaml

    Возвращает список TranscriptEntry в хронологическом порядке.
    """
    config = config or {}
    overlap_ratio_threshold = float(config.get("overlap_ratio_threshold", 0.5))
    merge_gap_ms = int(config.get("merge_same_speaker_gap_ms", 200))
    min_seg_dur_ms = int(config.get("min_segment_duration_ms", 100))
    ensure_monotonic = bool(config.get("ensure_monotonic", True))

    if len(emb_segments) != len(labels):
        raise ValueError(
            f"emb_segments ({len(emb_segments)}) and labels ({len(labels)}) "
            "must have the same length"
        )
    if not emb_segments:
        logger.warning("align_transcript: no diarization segments")
        return []

    overlap_segments = list(overlap_segments or [])

    # Приводим сегменты к списку и сопоставляем метки
    segments: List[Tuple[SpeechSegment, int]] = [
        (seg, int(lab)) for seg, lab in zip(emb_segments, labels)
    ]

    # --- Шаг 1+2+3: назначение спикера каждому слову ---
    word_speakers: List[Tuple[WordTimestamp, str]] = []
    for word in words:
        speaker = _assign_speaker(
            word=word,
            segments=segments,
            overlap_segments=overlap_segments,
            overlap_ratio_threshold=overlap_ratio_threshold,
        )
        if speaker is not None:
            word_speakers.append((word, speaker))

    if not word_speakers:
        logger.warning("align_transcript: no words got a speaker")
        return []

    # --- Шаг 4: группировка в TranscriptEntry ---
    entries = _group_words(word_speakers, merge_gap_ms=merge_gap_ms)

    # --- Шаг 5: фильтр коротких и монотонность ---
    entries = [e for e in entries if (e.end_ms - e.start_ms) >= min_seg_dur_ms]
    if ensure_monotonic:
        entries = _ensure_monotonic(entries)

    return entries


# ---------------------------------------------------------------------------
# Назначение спикера одному слову
# ---------------------------------------------------------------------------

def _assign_speaker(
    word: WordTimestamp,
    segments: Sequence[Tuple[SpeechSegment, int]],
    overlap_segments: Sequence[OverlapSegment],
    overlap_ratio_threshold: float,
) -> Optional[str]:
    """Выбрать speaker_id для слова.

    Логика:
      * Собираем пересечения слова со всеми сегментами диаризации.
      * Если у слова есть пересечение с overlap-сегментом, и в этом
        пересечении участвуют ≥2 спикера — фактически «выигрывает»
        спикер с наибольшим покрытием (это и есть обработка перекрытия).
      * Если ни одно пересечение не достигает порога — берём сегмент,
        ближайший к слову по времени (граничный случай).
      * Если сегментов вообще нет рядом — возвращаем None (слово
        выпадает из транскрипта).
    """
    word_dur = max(1, word.end_ms - word.start_ms)

    # 1) Пересечения с сегментами
    covered_by_speaker: Dict[int, int] = defaultdict(int)
    best_seg: Optional[Tuple[SpeechSegment, int]] = None
    best_ratio = 0.0

    for seg, label in segments:
        inter = _intersection_ms(word.start_ms, word.end_ms, seg.start_ms, seg.end_ms)
        if inter <= 0:
            continue
        covered_by_speaker[label] += inter
        ratio = inter / word_dur
        if ratio > best_ratio:
            best_ratio = ratio
            best_seg = (seg, label)

    # 2) Есть ли слово внутри overlap-сегмента?
    inside_overlap = any(
        _intersection_ms(word.start_ms, word.end_ms, o.start_ms, o.end_ms) > 0
        for o in overlap_segments
    )
    if inside_overlap and len(covered_by_speaker) >= 2:
        # Перекрытие: выбираем спикера с наибольшим покрытием внутри слова
        speaker_label = max(covered_by_speaker.items(), key=lambda kv: kv[1])[0]
        logger.debug(
            "overlap word=%r resolved to speaker=%d", word.word, speaker_label
        )
        return _label_to_id(speaker_label)

    # 3) Порог по пересечению
    if best_seg is not None and best_ratio >= overlap_ratio_threshold:
        return _label_to_id(best_seg[1])

    # 4) Граничный случай: ни один сегмент не дотянул до порога
    if best_seg is not None:
        # Берём сегмент с макс. пересечением (даже если < порога),
        # чтобы слово не потерялось на стыке.
        return _label_to_id(best_seg[1])

    # 5) Нет пересечений — ищем ближайший сегмент по времени
    nearest = _nearest_segment(word, segments)
    if nearest is None:
        return None
    return _label_to_id(nearest[1])


def _nearest_segment(
    word: WordTimestamp, segments: Sequence[Tuple[SpeechSegment, int]]
) -> Optional[Tuple[SpeechSegment, int]]:
    best = None
    best_dist = None
    for seg, label in segments:
        if seg.end_ms < word.start_ms:
            dist = word.start_ms - seg.end_ms
        elif seg.start_ms > word.end_ms:
            dist = seg.start_ms - word.end_ms
        else:
            dist = 0
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (seg, label)
    return best


# ---------------------------------------------------------------------------
# Группировка слов в TranscriptEntry
# ---------------------------------------------------------------------------

def _group_words(
    word_speakers: Sequence[Tuple[WordTimestamp, str]],
    merge_gap_ms: int,
) -> List[TranscriptEntry]:
    """Слить подряд идущие слова одного спикера в одну запись."""
    if not word_speakers:
        return []

    entries: List[TranscriptEntry] = []
    cur_speaker = word_speakers[0][1]
    cur_start = word_speakers[0][0].start_ms
    cur_end = word_speakers[0][0].end_ms
    cur_words: List[str] = [word_speakers[0][0].word]

    for word, speaker in word_speakers[1:]:
        if (
            speaker == cur_speaker
            and word.start_ms - cur_end <= merge_gap_ms
        ):
            cur_end = max(cur_end, word.end_ms)
            cur_words.append(word.word)
        else:
            entries.append(
                TranscriptEntry(
                    speaker_id=cur_speaker,
                    start_ms=cur_start,
                    end_ms=cur_end,
                    text=_join_words(cur_words),
                )
            )
            cur_speaker = speaker
            cur_start = word.start_ms
            cur_end = word.end_ms
            cur_words = [word.word]

    entries.append(
        TranscriptEntry(
            speaker_id=cur_speaker,
            start_ms=cur_start,
            end_ms=cur_end,
            text=_join_words(cur_words),
        )
    )
    return entries


def _join_words(words: Sequence[str]) -> str:
    """Склейка слов с учётом пунктуации.

    Правило простое: если предыдущее слово заканчивается на один из
    знаков .,!?;: — ставим пробел; иначе добавляем с пробелом же
    (русский язык, пробелы между словами уместны всегда).
    Для word-level ASR это достаточно.
    """
    return " ".join(w.strip() for w in words if w.strip()).strip()


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _intersection_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _label_to_id(label: int) -> str:
    """Стабильный человекочитаемый id спикера."""
    return f"speaker_{label:02d}"


def _ensure_monotonic(entries: List[TranscriptEntry]) -> List[TranscriptEntry]:
    """Гарантировать неубывание start_ms / end_ms.

    На случай, если ASR выдал слова не по порядку или группировка
    «съехала» — просто сортируем и подрезаем пересечения.
    """
    entries.sort(key=lambda e: (e.start_ms, e.end_ms))
    for i in range(1, len(entries)):
        if entries[i].start_ms < entries[i - 1].end_ms:
            entries[i].start_ms = entries[i - 1].end_ms
        if entries[i].end_ms < entries[i].start_ms:
            entries[i].end_ms = entries[i].start_ms
    return entries
