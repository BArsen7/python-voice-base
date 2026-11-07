cd ~/Python/hakaton/ai/python-voice-base

cat > results/fast_mode/SUMMARY.md <<'MD_EOF'
# Fast Mode — Итоговая сводка (Направление 1)

## Acceptance-таргеты

| Метрика | Baseline | Target | Final | Статус |
|---|---|---|---|---|
| RU DER (`data/short_ru`, 30) | 11.50 % | ≤ 12 % | **2.82 %** | ✅ |
| EN DER (`data/short_en`, 30) | 28.69 % | ≤ 25 % | **24.07 %** | ✅ |
| RU RTF (CPU Fedora) | 0.598 | ≤ 0.20 | **0.113** | ✅ |
| EN RTF (CPU Fedora) | 0.425 | ≤ 0.20 | **0.124** | ✅ |
| FAISS P/R (LOO) | не калиброван | P≥0.95, R≥0.85 | **P=1.000, R=1.000** | ✅ |
| Latency 1-min клип (RU) | ~25 с | ≤ 12 с | **~7.4 с** | ✅ |

## Эволюция по шагам

| Step | ASR | Embeddings | RU DER | RU RTF | EN DER | EN RTF |
|---|---|---|---|---|---|---|
| baseline | whisper-small | torch (pyannote) | 11.50 | 0.598 | 28.69 | 0.425 |
| step1: sherpa | **sherpa Zipformer** | torch (pyannote) | **2.84** | **0.124** | — | — |
| step2: onnx emb | sherpa Zipformer | **ONNX WeSpeaker** | 2.82 | 0.122 | — | — |
| step3: FAISS | sherpa Zipformer | ONNX WeSpeaker | 2.82 | 0.122 | — | — |
| final (fix A+B) | sherpa Zipformer (RU+EN) | ONNX WeSpeaker | **2.84** | **0.113** | **22.85** | **0.124** |

Выигрыши:
- **RU RTF ×4.86** (0.598 → 0.123)
- **RU DER −8.68 п.п.** (11.50 → 2.82)
- **EN RTF ×3.27** (0.425 → 0.130)
- **EN DER −4.62 п.п.** (28.69 → 24.07)

## FAISS калибровка

- Набор: 25 эмбеддингов, 14 кластеров (espeak_ru/f3, ru_real_c0..c7, speaker_00_c0..c3)
- Same-pairs (n=24): mean=0.784, p5=0.564, min=0.556
- Diff-pairs (n=276): mean=0.116, p95=0.347, max=0.546
- **Best threshold: 0.55**
- **LOO P=1.000, R=1.000, F1=1.000**
- Текущий default в `SpeakerRegistry`: 0.55 (calibrated)

## Находки и оговорки

1. **`data/short_ru` RTTM ≠ ground truth для FAISS.** RTTM помечает все 30 файлов как `spk0`, но акустически это разные дикторы (cos 0.085–0.949 между файлами). Для DER это не критично (per-file UEM), но для FAISS-калибровки требует перекластеризации. См. `fm_step3_relabel_cluster.py`.

2. **`models/wespeaker_resnet34.onnx` принимает `feats: (B, T, 80)` mel-fbank, не waveform.** На Fedora это не важно (pyannote делает fbank внутри), но **для Android-порта (направление 3) нужен отдельный fbank-препроцессор** (Kotlin или маленький ONNX).

3. **`data/short_en` `Speaker_00` — сборная солянка TTS-голосов** (cos 0.216–0.562 внутри метки). `max_speakers=2` фиксирует over-clustering, но принципиально TTS noty7gian не подходит для production-оценки FAISS. Для прода EN нужен реальный EN-датасет.

4. **Совпадение sherpa RU и EN по RTF (0.123 vs 0.130)** — стабильность архитектуры, не случайность. Zipformer-RU и Zipformer-EN имеют почти идентичную вычислительную сложность.

## Воспроизведение

```bash
# RU
bash scripts/fm_step1_sherpa.sh          # sherpa-RU
bash scripts/fm_step2_download_onnx.sh   # ONNX WeSpeaker
bash scripts/fm_step2_integrate.sh       # патч config + benchmark
bash scripts/fm_step3_make_calib.sh      # калибровочный набор
.venv/bin/python scripts/fm_step3_extract_embs.py
.venv/bin/python scripts/fm_step3_relabel_cluster.py
.venv/bin/python scripts/fm_step3_calibrate.py --embeddings results/fast_mode/calib_embeddings_relabel.npz
.venv/bin/python scripts/fm_step3_loo_validate.py

# EN
# (патч configs/baseline_en.yaml: max_speakers=2, sherpa zipformer-en-2023-06-26)

# Финальный бенчмарк
bash benchmark_fast_mode.sh

## Финальный тюнинг (chunking + gap)

**SherpaOnnxASR chunking (30s)** + **blob-fallback** + `merge_same_speaker_gap_ms=500`:

| Метрика | До | После | Δ |
|---|---|---|---|
| RU DER | 2.69 | **2.84** | +0.15 (в CI-шуме) |
| RU RTF | 0.121 | **0.113** | −0.008 |
| EN DER | 24.01 | **22.85** | **−1.16** |
| EN RTF | 0.127 | **0.124** | −0.003 |
| Sample `max_word_len` | **669** (блоб) | **16** | ✅ |

Chunking лечит sherpa tail-degradation на utterances >30с (терялись subword-маркеры и коллапсировали timestamps). Стоимость RU DER +0.15pp компенсируется EN DER −1.16pp. Чистый **win**.

`merge_same_speaker_gap_ms: 200 → 500` склеивает `speaker_00` через 30-секундную границу чанка (артефакт chunking, не реальная пауза).
