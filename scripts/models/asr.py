"""Блок 7: ASR с word-level timestamps.

- FasterWhisperASR — faster-whisper (CTranslate2), возвращает слова с
  таймкодами в миллисекундах.
- SherpaOnnxASR — sherpa-onnx (Zipformer transducer), также word-level
  таймкоды (если модель их отдаёт; для transducer-моделей sherpa-onnx
  обычно возвращает токены/слова с таймкодами).

Оба класса возвращают List[WordTimestamp] в миллисекундах.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.core.interfaces import BaseASR, WordTimestamp

logger = logging.getLogger(__name__)


class FasterWhisperASR(BaseASR):
    """faster-whisper с word_timestamps=True."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "faster_whisper_small")
        # config["model"] может быть "faster_whisper_small" → "small"
        self.model_size = self.model_name.replace("faster_whisper_", "")
        self.language = config.get("language", "ru")
        self.beam_size = int(config.get("beam_size", 5))
        self.word_timestamps = bool(config.get("word_timestamps", True))
        self.vad_filter = bool(config.get("vad_filter", False))
        self.device = config.get("device", "cpu")
        self.compute_type = config.get("compute_type", "default")
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "FasterWhisperASR requires faster-whisper. "
                "Install with: pip install faster-whisper"
            ) from e

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(
            "faster-whisper loaded: size=%s, device=%s",
            self.model_size,
            self.device,
        )

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[WordTimestamp]:
        self._lazy_load()

        # faster-whisper принимает numpy float32 (16 kHz)
        segments, _info = self._model.transcribe(
            waveform,
            language=self.language,
            max_active_paths=self.max_active_paths,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
        )

        words: List[WordTimestamp] = []
        for segment in segments:
            if not segment.words:
                continue
            for w in segment.words:
                words.append(
                    WordTimestamp(
                        word=w.word.strip(),
                        start_ms=int(w.start * 1000),
                        end_ms=int(w.end * 1000),
                    )
                )
        logger.info("faster-whisper: %d words", len(words))
        return words


class SherpaOnnxASR(BaseASR):
    """sherpa-onnx Zipformer transducer.

    Требует указания путей к ONNX-файлам модели в config:
      model_dir: путь к директории с encoder/decoder/joiner/tokens
    или явные пути:
      encoder, decoder, joiner, tokens
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.language = config.get("language", "ru")
        self.model_dir = config.get("model_dir")
        self.encoder = config.get("encoder")
        self.decoder = config.get("decoder")
        self.joiner = config.get("joiner")
        self.tokens = config.get("tokens")
        self._recognizer = None

    def _lazy_load(self):
        if self._recognizer is not None:
            return
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "SherpaOnnxASR requires sherpa-onnx. "
                "Install with: pip install sherpa-onnx"
            ) from e

        if self.model_dir:
            model_dir = Path(self.model_dir)
            encoder = str(model_dir / "encoder.onnx")
            decoder = str(model_dir / "decoder.onnx")
            joiner = str(model_dir / "joiner.onnx")
            tokens = str(model_dir / "tokens.txt")
        else:
            encoder, decoder, joiner, tokens = (
                self.encoder,
                self.decoder,
                self.joiner,
                self.tokens,
            )

        if not all([encoder, decoder, joiner, tokens]):
            raise ValueError(
                "SherpaOnnxASR needs model_dir or explicit "
                "encoder/decoder/joiner/tokens paths in config"
            )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        logger.info("sherpa-onnx Zipformer loaded from %s", self.model_dir or "explicit paths")

    MAX_CHUNK_SEC = 30

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[WordTimestamp]:
        self._lazy_load()
        if sample_rate != 16000:
            raise ValueError("SherpaOnnxASR expects 16 kHz audio")
        n = len(waveform)
        chunk_samples = self.MAX_CHUNK_SEC * sample_rate
        if n <= chunk_samples:
            return self._transcribe_chunk(waveform, sample_rate, offset_ms=0)
        all_words: List[WordTimestamp] = []
        for start in range(0, n, chunk_samples):
            end = min(start + chunk_samples, n)
            offset_ms = int(start / sample_rate * 1000)
            all_words.extend(
                self._transcribe_chunk(waveform[start:end], sample_rate, offset_ms)
            )
        return all_words

    def _transcribe_chunk(
        self, waveform: np.ndarray, sample_rate: int, offset_ms: int = 0
    ) -> List[WordTimestamp]:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform.astype(np.float32))
        self._recognizer.decode_stream(stream)
        result = stream.result
        if not result.text:
            return []
        words: List[WordTimestamp] = []
        tokens = getattr(result, "tokens", []) or []
        timestamps = getattr(result, "timestamps", []) or []
        if tokens and timestamps and len(tokens) == len(timestamps):
            current_word = ""
            start_ms = None
            last_token_ms = 0
            for tok, ts in zip(tokens, timestamps):
                ts_ms = int(ts * 1000) + offset_ms
                is_new = tok.startswith(" ") or tok.startswith("\u2581")
                clean = tok.lstrip(" \u2581")
                if not clean:
                    continue
                if is_new:
                    if current_word:
                        words.append(WordTimestamp(
                            word=current_word,
                            start_ms=start_ms + offset_ms,
                            end_ms=ts_ms,
                        ))
                    current_word = clean
                    start_ms = int(ts * 1000)
                else:
                    if not current_word:
                        current_word = clean
                        start_ms = int(ts * 1000)
                    else:
                        current_word += clean
                last_token_ms = ts_ms
            if current_word and start_ms is not None:
                words.append(WordTimestamp(
                    word=current_word,
                    start_ms=start_ms + offset_ms,
                    end_ms=last_token_ms,
                ))
        # Blob fallback: 600-char word from sherpa tail-degradation
        if any(len(w.word) > 30 for w in words):
            parts = result.text.split()
            if parts:
                total_ms = int(len(waveform) / sample_rate * 1000)
                step = max(1, total_ms // len(parts))
                return [
                    WordTimestamp(
                        word=p,
                        start_ms=offset_ms + i * step,
                        end_ms=offset_ms + (i + 1) * step,
                    )
                    for i, p in enumerate(parts)
                ]
        return words


class SherpaOnnxWhisperMediumASR(BaseASR):
    """
    ASR на базе Whisper (medium / large-v3) в формате ONNX через sherpa-onnx.
    Для Android обязательно использовать INT8- или FP16-квантованные модели.

    Пример конфига (configs/accuracy_mode.yaml):
        encoder: "models_bundle/whisper-medium/medium-encoder.int8.onnx"
        decoder: "models_bundle/whisper-medium/medium-decoder.int8.onnx"
        tokens:  "models_bundle/whisper-medium/medium-tokens.txt"
        decoding_method: "modified_beam_search"   # обязательно для Accuracy Mode
        beam_size: 4
        num_paths: 4
    """

    def __init__(self, config):
        super().__init__(config)
        self.language = config.get("language", "ru")
        self.encoder = config.get("encoder")
        self.decoder = config.get("decoder")
        self.tokens = config.get("tokens")
        self.decoding_method = config.get("decoding_method", "modified_beam_search")
        self.max_active_paths = int(
            config.get("max_active_paths", config.get("beam_size", 4))
        )
        self.num_threads = int(config.get("num_threads", 4))
        self._recognizer = None  # lazy

    def _lazy_load(self):
        if self._recognizer is not None:
            return
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "SherpaOnnxWhisperMediumASR requires sherpa-onnx. "
                "Install with: pip install sherpa-onnx"
            ) from e

        if not all([self.encoder, self.decoder, self.tokens]):
            raise ValueError(
                "SherpaOnnxWhisperMediumASR requires encoder, decoder и tokens."
            )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=self.encoder,
            decoder=self.decoder,
            tokens=self.tokens,
            language=self.language,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
            num_threads=self.num_threads,
            debug=False,
        )
        logger.info(
            "sherpa-onnx Whisper loaded: lang=%s method=%s beam=%d paths=%d",
            self.language, self.decoding_method, self.max_active_paths,
        )

    def transcribe(self, waveform, sample_rate):
        self._lazy_load()
        if sample_rate != 16000:
            raise ValueError("SherpaOnnxWhisperMediumASR expects 16 kHz audio")

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform)
        self._recognizer.decode_stream(stream)
        result = stream.result

        words = []
        if getattr(result, "tokens", None) and getattr(result, "timestamps", None):
            for token, ts in zip(result.tokens, result.timestamps):
                words.append(WordTimestamp(
                    word=token,
                    start_ms=int(ts.start * 1000),
                    end_ms=int(ts.end * 1000),
                ))
        else:
            words.append(WordTimestamp(
                word=result.text.strip(),
                start_ms=0,
                end_ms=int(len(waveform) / sample_rate * 1000),
            ))
        logger.info("sherpa-onnx Whisper: %d words", len(words))
        return words
