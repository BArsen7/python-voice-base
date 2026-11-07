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

from scripts.core.interfaces import (
    OverlapSegment,
    SpeechSegment,
    TranscriptEntry,
    WordTimestamp,
)
import re

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
    # ВАЖНО: если нужна гарантированная монотонность, сортируем слова ДО
    # группировки. Иначе out-of-order слова от ASR могут склеиться в одну
    # реплику (см. test_alignment_monotonic_and_valid_json_shape).
    if ensure_monotonic:
        word_speakers.sort(key=lambda ws: (ws[0].start_ms, ws[0].end_ms))

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
            and word.start_ms >= cur_start  # защита от out-of-order слов
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


def _align_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _align_set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def median_smooth_speakers(segments, window=3, key='speaker_id'):
    """
    Медианное сглаживание меток спикеров (устраняет "мигание" A→B→A).
    Работает с dict-ами и с объектами.
    """
    from collections import Counter
    n = len(segments)
    if n < window or window < 3:
        return list(segments)
    half = window // 2
    result = list(segments)
    for i in range(half, n - half):
        vals = [_align_get(segments[j], key) for j in range(i - half, i + half + 1)]
        most = Counter(vals).most_common(1)[0][0]
        _align_set(result[i], key, most)
    return result


def viterbi_smooth_speakers(segments, key='speaker_id'):
    """
    Упрощённое Viterbi-подобное сглаживание по меткам спикеров:
    короткие сегменты, зажатые между двумя одинаковыми спикерами,
    поглощаются соседями.
    """
    n = len(segments)
    if n < 3:
        return list(segments)
    durations = []
    for seg in segments:
        s = _align_get(seg, 'start_ms', 0)
        e = _align_get(seg, 'end_ms', 0)
        durations.append(max(1, e - s))
    median_dur = sorted(durations)[len(durations) // 2]
    result = list(segments)
    for i in range(1, n - 1):
        if durations[i] < median_dur * 0.5:
            prev = _align_get(result[i - 1], key)
            nxt = _align_get(result[i + 1], key)
            cur = _align_get(result[i], key)
            if prev == nxt and cur != prev:
                _align_set(result[i], key, prev)
    return result


_RU_NUM_WORDS = {
    "ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
    "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
    "восемь": "8", "девять": "9", "десять": "10", "одиннадцать": "11",
    "двенадцать": "12", "тринадцать": "13", "четырнадцать": "14",
    "пятнадцать": "15", "шестнадцать": "16", "семнадцать": "17",
    "восемнадцать": "18", "девятнадцать": "19", "двадцать": "20",
    "тридцать": "30", "сорок": "40", "пятьдесят": "50",
    "шестьдесят": "60", "семьдесят": "70", "восемьдесят": "80",
    "девяносто": "90", "сто": "100",
}


def normalize_russian_text(text, fix_numbers=True, fix_punctuation=True):
    """Локальная нормализация чисел и базовой пунктуации (без внешних API)."""
    if not text:
        return text
    if fix_numbers:
        for word, digit in _RU_NUM_WORDS.items():
            text = re.sub(rf"\b{word}\b", digit, text, flags=re.IGNORECASE)
    if fix_punctuation:
        text = re.sub(
            r"(^|[.!?]\s+)([а-яё])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def postprocess_transcript(segments,
                           smooth_method="median",
                           smooth_window=3,
                           fix_numbers=True,
                           fix_punctuation=True,
                           speaker_key="speaker_id",
                           text_key="text"):
    """
    Пост-обработка последовательности сегментов транскрипта:
      1. Сглаживание меток спикеров (median / viterbi).
      2. Локальная нормализация текста (числа, базовая пунктуация).

    Возвращает новый список; исходные объекты не мутируются.
    """
    import copy
    work = [dict(s) if isinstance(s, dict) else copy.copy(s) for s in segments]

    if smooth_method == "median":
        work = median_smooth_speakers(work, window=smooth_window, key=speaker_key)
    elif smooth_method == "viterbi":
        work = viterbi_smooth_speakers(work, key=speaker_key)

    for seg in work:
        txt = _align_get(seg, text_key, "")
        if txt:
            _align_set(seg, text_key, normalize_russian_text(
                txt, fix_numbers=fix_numbers, fix_punctuation=fix_punctuation,
            ))
    return work
