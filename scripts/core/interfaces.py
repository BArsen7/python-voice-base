"""Абстрактные интерфейсы пайплайна диаризации.

Все модели реализуют Strategy-паттерн: конкретный класс выбирается
в orchestrator.py на основе config.yaml.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Общие структуры данных
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """Речевой сегмент, детектированный VAD."""
    start_ms: int
    end_ms: int
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class OverlapSegment:
    """Участок перекрытия речи."""
    start_ms: int
    end_ms: int
    probability: float = 1.0


@dataclass
class SpeakerSegment:
    """Сегмент с назначенным спикером (результат кластеризации)."""
    speaker_id: str
    start_ms: int
    end_ms: int


@dataclass
class WordTimestamp:
    """Слово с таймкодами от ASR."""
    word: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptEntry:
    """Финальная запись транскрипта (выходной JSON)."""
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


# ---------------------------------------------------------------------------
# Блок 3: VAD
# ---------------------------------------------------------------------------

class BaseVAD(ABC):
    """Голосовая активность (Voice Activity Detection)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def detect(self, waveform: np.ndarray, sample_rate: int) -> List[SpeechSegment]:
        """Вернуть список речевых сегментов в миллисекундах."""
        ...


# ---------------------------------------------------------------------------
# Блок 4: Overlap Detection
# ---------------------------------------------------------------------------

class BaseOverlapDetector(ABC):
    """Детекция перекрытий речи."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        speech_segments: Sequence[SpeechSegment],
    ) -> List[OverlapSegment]:
        """Вернуть список участков перекрытия в миллисекундах."""
        ...


# ---------------------------------------------------------------------------
# Блок 5: Speaker Embeddings
# ---------------------------------------------------------------------------

class BaseEmbeddingExtractor(ABC):
    """Извлечение L2-нормализованных эмбеддингов спикера."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def extract(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        segments: Sequence[SpeechSegment],
    ) -> Tuple[np.ndarray, List[SpeechSegment]]:
        """
        Вернуть:
          embeddings: np.ndarray формы (N, embedding_dim), L2-нормализованные
          segments:   соответствующие сегменты (могут быть отфильтрованы)
        """
        ...


# ---------------------------------------------------------------------------
# Блок 6: Clustering
# ---------------------------------------------------------------------------

class BaseClustering(ABC):
    """Кластеризация эмбеддингов в спикеров."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Вернуть массив меток кластеров (int) длины N."""
        ...


# ---------------------------------------------------------------------------
# Блок 7: ASR
# ---------------------------------------------------------------------------

class BaseASR(ABC):
    """Автоматическое распознавание речи с word-level timestamps."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[WordTimestamp]:
        """Вернуть список слов с таймкодами в миллисекундах."""
        ...
