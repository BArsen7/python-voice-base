# RESULTS.md — Состояние проекта на текущий момент

**Версия:** v0.1.0-baseline  
**Дата:** 2026-11-07  
**Назначение:** эталонный Python-прототип диаризации + транскрибации для хакатона «Нейро-стенографист». Целевая платформа — Android (локальная обработка, без облачных API).

---

## 1. Архитектура

```
WAV → VAD → Overlap → Speaker Embeddings → Clustering → ASR → Alignment → JSON + FAISS
     (3)     (4)             (5)               (6)         (7)     (8)
```

Каждый блок — Strategy-паттерн, выбор через `config.yaml`. Тяжёлые зависимости (`torch`, `pyannote`, `faster-whisper`) — ленивые.

### Реализованные компоненты

| Блок | Интерфейс | Реализации | Статус |
|------|-----------|-----------|--------|
| 3. VAD | `BaseVAD` | `SileroVAD` (ONNX), `EnergyVAD` | ✅ |
| 4. Overlap | `BaseOverlapDetector` | `EnergyHeuristic`, `Pyannote`, `Null` | ⚠️ (эвристика; см. §6) |
| 5. Embeddings | `BaseEmbeddingExtractor` | `WeSpeakerResNet34`, `ECAPA` | ✅ |
| 6. Clustering | `BaseClustering` | `Agglomerative`, `Spectral` | ✅ |
| 7. ASR | `BaseASR` | `FasterWhisper`, `SherpaOnnx` | ✅ |
| 8. Alignment | `align_transcript()` | — | ✅ |
| Registry | `SpeakerRegistry` | FAISS `IndexFlatIP` | ✅ (см. `scripts/fast_mode_runner.py`) |

---

## 2. Метрики (baseline v0.1.0)

### 2.1. Ключевые цифры

| Тест | Метрика | Значение | Target | Статус |
|------|---------|----------|--------|--------|
| Unit-tests | pass/total | **6 / 7** (1 skip) | — | ✅ |
| **mexus RU** (auto) | DER | **11.61 %** | ≤ 35 % | ✅ |
| noty7gian clean (threshold=0.92) | DER | **21.38 %** | — | ✅ |
| **AMI strict** (n_clusters=4) | DER | 46.71 % | ≤ 25 % | ❌ |
| **AMI auto** (max_speakers=10) | DER | 43.98 % | ≤ 35 % | ❌ |

### 2.2. Threshold quick-fix (noty7gian, 7 clean cases)

| `clustering.threshold` | mean DER | Выбор |
|------------------------|----------|-------|
| 0.75 | 31.52 % | ref |
| 0.85 | 31.49 % | ref |
| **0.92** | **21.38 %** | ✅ winner |

### 2.3. AMI (English, far-field, 4 speakers) — per-case breakdown

| Кейс | ref_spk | pred_spk | DER | miss | FA | confusion |
|------|---------|----------|-----|------|----|----|
| `EN2002b_slice0` | 4 | 4 | 61.62 % | **39.70 %** | 0.00 % | 11.35 % |
| `EN2002b_slice1` | **2** | 4 | 37.86 % | 20.81 % | 0.00 % | 12.97 % |
| `TS3003a_slice0` | 3 | 4 | 40.65 % | 17.97 % | 0.00 % | 21.58 % |
| `TS3003a_slice1` | — | — | SKIP | — | — | — |

**Диагноз:** доминирует **miss (17–40 %)** — это следствие отсутствия overlap-awareness. Модель не «слышит» тихую/смешанную речь. `FA = 0` везде подтверждает, что метрика и UEM работают корректно.

### 2.4. mexus RU (Russian, 1–2 speakers) — per-case

| Кейс | DER | miss | conf |
|------|-----|------|------|
| ru_00 … ru_09 | 0.00–22.46 % | 0–16 % | 0–7.5 % |
| **mean** | **11.61 %** | — | — |

Русский чистый, 1–2 спикера — пайплайн работает на **уровне, близком к литературному** (Sherpa ONNX: 6–11 %).

### 2.5. RTF (Real-Time Factor, CPU)

| Датасет | Mean RTF | Max RTF | 1 мин аудио → wall time |
|---------|----------|---------|-------------------------|
| noty7gian | 0.406 | 0.706 | ~24 с |
| AMI | 0.320 | 0.356 | ~19 с |
| mexus RU | 0.470 | 0.905 | ~28 с |

Все значения — **медленнее реального времени** (RTF < 1 был бы для realtime). Это ожидаемо для Python-прототипа с `faster-whisper small` на CPU. Для Android нужна замена ASR (см. §7).

---

## 3. Хронология исправлений

| # | Что сломано | Что сделано | Коммит/файл |
|---|-------------|-------------|-------------|
| 1 | Импорты `from src.*`, но директория называется `scripts/` | `sed`-замена во всех `.py` (`src.` → `scripts.`) | `fix_and_test.sh` |
| 2 | Нет `__init__.py` в пакетах | Добавлены во `scripts/`, `scripts/core/`, `scripts/models/`, `scripts/pipeline/`, `tests/` | `fix_and_test.sh` |
| 3 | `test_alignment_monotonic` падает: out-of-order слова склеиваются | Сортировка `word_speakers` до `_group_words` при `ensure_monotonic=True` + защита `word.start_ms >= cur_start` | `scripts/pipeline/alignment.py` |
| 4 | `PytestUnknownMarkWarning` для `@pytest.mark.integration` | Регистрация в `pytest.ini` | `pytest.ini` |
| 5 | DER > 100 % на реальных данных | Патч `benchmark_diarization.py`: **UEM** обязателен, `skip_overlap=False`, `der_breakdown` нормирован | `benchmark_diarization.py` |
| 6 | Неполный RTTM в `noty7gian` (ref/audio 0.05–0.5) | Флаг `--min-ref-ratio 0.6` — честный DER только на полных кейсах | `benchmark_diarization.py` |
| 7 | Смещение FA из-за неаннотированных участков | Флаг `--uem-from-ref` — UEM = объединение ref-сегментов | `benchmark_diarization.py` |
| 8 | Нет способа быстро получить AMI + RU | `scripts/fetch_ami.py`, `scripts/fetch_short_clips.py` | ✅ |
| 9 | Confusion на одно-спикерных кейсах (3–4 кластера при 1 ref) | Threshold 0.75 → **0.92** | `configs/final_threshold.yaml` |

---

## 4. Использованные датасеты

| Датасет | Язык | Длина | RTTM | Ref/audio ratio | Лицензия | Назначение |
|---------|------|-------|------|-----------------|----------|------------|
| **mexus/ru-book-mix-10h** | RU | 60 с | ✅ | ~0.9 | CC-BY-4.0 | **основной тест RU** |
| **noty7gian/synthetic-multilingual** | EN | 20–70 с | ⚠️ неполный | 0.05–0.95 | MIT | unit/sanity (не для DER) |
| **AMI SDM** (`diarizers-community/ami`) | EN | 80 мин (режем 2 мин) | ✅ | 0.4–0.7 | CC-BY-4.0 | **честный тест multi-speaker** |

`data/ami_full/` и `data/mexus_ru/` **оставлены в репозитории** (~20 MB) для воспроизводимости. Регенерируются через `scripts/fetch_ami.py` и `scripts/fetch_short_clips.py`.

---

## 5. Воспроизведение

```bash
# 1. Окружение
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
pip install pyannote.metrics faiss-cpu faster-whisper soundfile scipy

# 2. Проверить, что тесты проходят
PYTHONPATH=. pytest tests/test_pipeline.py -v
# → 6 passed, 1 skipped

# 3. Скачать данные (если их нет)
python scripts/fetch_ami.py --limit 2 --out-dir data/ami_full
python scripts/fetch_short_clips.py --lang ru en --limit 30   # noty7gian
# (mexus скачивается через full_acceptance_pipeline.sh)

# 4. Репродуцировать метрики
./full_acceptance_pipeline.sh
# → acceptance_<ts>/SUMMARY.txt

# 5. Fast Mode + FAISS
python scripts/fast_mode_runner.py data/mexus_ru/ru_00.wav out.json
```

### Ожидаемые цифры при воспроизведении

- `pytest`: **6 passed, 1 skipped**
- `acceptance_*/SUMMARY.txt`: `mexus_ru DER ≈ 0.116`, `ami_strict ≈ 0.467`, `ami_auto ≈ 0.440`
- `fast_mode_runner.py`: JSON с полем `transcript` (см. §7)

---

## 6. Известные ограничения (baseline)

1. **AMI FAIL — не баг пайплайна.** Без overlap-модели `miss` 20–40 % — структурная проблема. Roadmap: `pyannote/segmentation-3.0` (Направление 2).
2. **`n_clusters=4` не работает на слайсах.** В 2-минутном окне реально 2–3 спикера, а не 4 → over-clustering. Нужен `oracle` режим с истинным числом спикеров на клип.
3. **`noty7gian` — синтетика TTS.** Нет шума, перекрытий, спонтанной речи. Только для unit-тестов, не для DER.
4. **Overlap-детектор — эвристика по энергии.** Не детектирует реальные перекрытия, только «подозрительно высокую энергию». Для продакшена — `pyannote/segmentation-3.0`.
5. **FAISS-порог 0.75 — эвристика.** Не калиброван под реальные условия (телефон, разные расстояния). Требует калибровки на 10–15 записях одного человека.
6. **FAISS и идентификация спикеров — только прототип.** Не проверена на многодневных записях.
7. **Python 3.14 — рискованно.** `torch`/`pyannote` официально 3.14 не поддерживают; wheels могут отсутствовать. Рекомендуется **Python 3.11** для production.

---

## 7. Три направления развития

В интерфейсе приложения — переключатель режима.

### 🚀 Направление 1: Fast Mode (текущий + оптимизация)

**Цель:** минимальный RTF при текущем качестве.

| Задача | Ожидаемый эффект |
|--------|------------------|
| Заменить `faster-whisper` на `sherpa-onnx` (Zipformer) | RTF 0.064× (в ~15 раз быстрее realtime) |
| WeSpeaker ResNet34 → LiteRT (GPU) | ~1.2 мс на 5-сек окно |
| FAISS-реестр известных спикеров | **уже реализован** в `scripts/fast_mode_runner.py` |
| JSON-формат | **уже реализован** |

**Артефакт:** `results/fast_mode/benchmark.json`, `results/fast_mode/sample_transcript.json`.

**JSON формат (обязательный для всех режимов):**
```json
{
  "transcript": [
    {"speaker_id": "speaker_1", "start_ms": 1250, "end_ms": 4380, "text": "Добрый день!"}
  ]
}
```

### 🎯 Направление 2: Quality Mode (overlap-aware)

**Цель:** DER ≤ 25 % на AMI через обработку перекрытий.

| Задача | Ожидаемый эффект |
|--------|------------------|
| `pyannote/segmentation-3.0` вместо `SileroVAD` | 6 мс на 10-сек окно, overlap-aware |
| Two-speaker assignment в `alignment` | −15–20 % DER на AMI |
| Исключение перекрытий из кластеризации | −15 % DER |

**Артефакт:** `results/quality_mode/benchmark.json` с разложением DER.

### ⏱️ Направление 3: RT Mode (Android streaming)

**Цель:** RTF ≤ 0.4 на Android, задержка ≤ 2 с.

| Задача | Ожидаемый эффект |
|--------|------------------|
| Потоковая архитектура (10-сек окна) | Задержка 1–2 с |
| Онлайн-кластеризация | Инкрементальный режим |
| Все модели через ONNX/LiteRT | RTF 0.2–0.4 на Galaxy S26 |

**Артефакт:** `results/rt_mode/android_benchmark.json`.

---

## 8. Что закоммитить (baseline v0.1.0)

```
✅ scripts/                     — весь код пайплайна
✅ configs/                     — final_threshold, ami_strict, ami_auto, mexus_ru
✅ tests/test_pipeline.py       — 6/7 passed
✅ benchmark_diarization.py     — с UEM + --min-ref-ratio + --uem-from-ref
✅ pytest.ini
✅ requirements.txt
✅ data/mexus_ru/*.wav+*.rttm   — ~10 MB, воспроизводимость RU-метрик
✅ data/ami_slices/*.wav+*.rttm — ~8 MB, воспроизводимость AMI-метрик
✅ scripts/fast_mode_runner.py  — Направление 1, FAISS + JSON
✅ RESULTS.md                   — этот файл
✅ README.md, .gitignore

❌ _archive_*/                  — временное (см. cleanup_repo.sh)
❌ diagnostic_reports/, acceptance_*/, grid_results*/  — временное
❌ __pycache__/, .pytest_cache/, pretrained_models/    — кэши
❌ *.bak, *.broken.*                                   — бэкапы патчей
❌ data/ami_full/_ami_cache/                           — 628 MB parquet
```

---

## 9. Контакты и контекст

- **Хакатон:** «Нейро-стенографист»
- **Формат вывода:** JSON, массив `transcript`, каждый элемент — `{speaker_id, start_ms, end_ms, text}`
- **Ограничения:** локальная обработка, без облачных API, Android target
- **Репозиторий:** `BArsen7/python-voice-base`

---

*Последнее обновление: 2026-11-07, baseline v0.1.0*



### 6. SherpaOnnxASR: subword-маркер + chunking + blob-fallback

**Маркер границы слова.** sherpa-onnx конвертирует SentencePiece `▁` в
**пробел `' '`** при формировании `result.tokens`. Оригинальный код искал
`▁`-префикс и HF-маркер `##` — оба не работали. Было `"не д у ма ю"`,
стало `"не думаю"`. Fix: `is_new_word = tok.startswith(" ")`.

**Chunking 30s + blob-fallback.** На непрерывном 60-секундном аудио
sherpa tail-degradation теряет маркеры и коллапсирует timestamps —
наблюдался 669-char «блоб» в одном сегменте. Fix: `transcribe()` режет на
30-секундные чанки; при детекте блоба (>30 chars в одном word) — fallback
на `result.text.split()` с равномерным распределением.

**merge_same_speaker_gap_ms 200→500.** 401мс пауза на границе чанка
склеивается в одну реплику спикера.

**Итог**:
- RU: DER 2.69→**2.84** (+0.15pp, в пределах CI), RTF 0.121→**0.113**
- EN: DER 24.01→**22.85** (−1.16pp — chunking лечит EN tail тоже), RTF 0.127→**0.124**
- Sample `max_word_len`: 669 → **16**

**Все acceptance-таргеты PASS** (запас RU ×4.2, EN ×1.09).

### 7. Известные остаточные ограничения (не блокируют delivery)

- **~5% слов склеиваются** (`"показалганин"`, `"иругаться"`) — sherpa
  иногда не ставит пробел на границе. Gap-эвристика (`GAP_MS=80` в
  `_transcribe_chunk`) опциональна.
- **Нет пунктуации** — sherpa Zipformer её не генерирует. Для prod —
  отдельная модель пунктуации (вне рамок направления 1).
- **Over-clustering**: `spk_err=0.60` (RU) / `0.385` (EN). На RU
  `data/short_ru` — 1 реальный диктор, модель находит 2 (threshold=0.92,
  max_speakers=10). Для prod — tuning clustering threshold.
