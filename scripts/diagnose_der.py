#!/usr/bin/env python3
"""diagnose_der.py — почему DER > 100% на одном клипе.

Печатает:
  - длительность аудио
  - суммарное время речи в RTTM, max_end, список спикеров
  - то же самое для вывода пайплайна
  - DER breakdown: miss / false alarm / confusion (абс. и %)
  - автодиагноз (причина A/B/C/D)

Usage:
  python scripts/diagnose_der.py data/short/en_0000.wav data/short/en_0000.rttm configs/baseline.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def load_rttm_stats(path: Path):
    total = 0.0
    max_end = 0.0
    speakers = set()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            tbeg = float(parts[3])
            tdur = float(parts[4])
            total += tdur
            max_end = max(max_end, tbeg + tdur)
            speakers.add(parts[7])
            n += 1
    return dict(total=total, max_end=max_end, n_segments=n, speakers=sorted(speakers))


def main(wav_path: Path, rttm_path: Path, cfg_path: Path) -> None:
    import soundfile as sf
    from scripts.pipeline.orchestrator import DiarizationPipeline, load_wav

    # --- Аудио ---
    info = sf.info(str(wav_path))
    wav_dur = info.frames / info.samplerate
    print(f"\n=== {wav_path.name} ===")
    print(f"Audio duration: {wav_dur:.3f} s  ({info.frames} samples @ {info.samplerate} Hz)")

    # --- RTTM ---
    ref = load_rttm_stats(rttm_path)
    print(f"\nRTTM (reference):")
    print(f"  Total speech: {ref['total']:.3f} s")
    print(f"  Max end:      {ref['max_end']:.3f} s")
    print(f"  Segments:     {ref['n_segments']}")
    print(f"  Speakers:     {ref['speakers']}")

    diag = []
    if ref["total"] > wav_dur * 1.5:
        diag.append(
            f"RTTM total ({ref['total']:.1f}s) > audio ({wav_dur:.1f}s) x{ref['total']/wav_dur:.1f}"
        )
        diag.append("  → таймстемпы RTTM, вероятно, в миллисекундах")
    elif ref["max_end"] > wav_dur * 1.1:
        diag.append(
            f"RTTM max_end ({ref['max_end']:.1f}s) > audio ({wav_dur:.1f}s)"
        )
        diag.append("  → RTTM выходит за аудио (сдвиг или ms)")
    elif ref["total"] < wav_dur * 0.3:
        diag.append(
            f"RTTM покрывает только {ref['total']/wav_dur*100:.0f}% аудио"
        )
        diag.append("  → RTTM может быть неполным")

    # --- Пайплайн ---
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    waveform, sr = load_wav(
        wav_path, target_sr=cfg.get("pipeline", {}).get("sample_rate", 16000)
    )
    pipe = DiarizationPipeline(cfg)
    entries = pipe.run(waveform, sr)

    hyp_total = sum((e.end_ms - e.start_ms) / 1000.0 for e in entries)
    hyp_max_end = max((e.end_ms / 1000.0 for e in entries), default=0)
    hyp_speakers = sorted({e.speaker_id for e in entries})
    print(f"\nHypothesis (pipeline):")
    print(f"  Total speech: {hyp_total:.3f} s")
    print(f"  Max end:      {hyp_max_end:.3f} s")
    print(f"  Entries:      {len(entries)}")
    print(f"  Speakers:     {hyp_speakers}")

    if hyp_max_end > wav_dur + 0.5:
        diag.append(
            f"hyp max_end ({hyp_max_end:.1f}s) > audio ({wav_dur:.1f}s)"
        )
        diag.append("  → ASR галлюцинирует за пределами аудио")
    if hyp_total > ref["total"] * 1.3:
        diag.append(
            f"hyp_total ({hyp_total:.1f}s) ≫ ref_total ({ref['total']:.1f}s)"
        )
        diag.append("  → огромный false alarm")

    # --- Детальный DER ---
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann = Annotation(uri=rttm_path.stem)
    for line in open(rttm_path, encoding="utf-8"):
        p = line.strip().split()
        if len(p) < 9 or p[0] != "SPEAKER":
            continue
        tbeg, tdur, spk = float(p[3]), float(p[4]), p[7]
        ref_ann[Segment(tbeg, tbeg + tdur)] = spk

    hyp_ann = Annotation(uri="hyp")
    for e in entries:
        if e.end_ms <= e.start_ms:
            continue
        hyp_ann[Segment(e.start_ms / 1000, e.end_ms / 1000)] = e.speaker_id

    uem = Segment(0.0, wav_dur)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    der = metric(ref_ann, hyp_ann, uem=uem)
    detailed = metric(ref_ann, hyp_ann, uem=uem, detailed=True)

    print(f"\nDER breakdown (collar=0.25, UEM=audio):")
    print(f"  {'component':<28} {'abs (s)':>10} {'%':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*8}")
    for k, v in detailed.items():
        if isinstance(v, (int, float)):
            # pyannote возвращает абсолютные значения (в секундах)
            print(f"  {k:<28} {float(v):>10.3f} {float(v)/max(ref['total'],1e-9)*100:>7.2f}%")
    print(f"  {'TOTAL DER':<28} {der*max(ref['total'],1e-9):>10.3f} {der*100:>7.2f}%")

    # --- Диагноз ---
    print(f"\nDiagnosis:")
    if diag:
        for line in diag:
            print(f"  ⚠️  {line}")
    elif der <= 0.6:
        print(f"  ✓ DER {der*100:.2f}% — в разумных пределах, метрика работает")
    else:
        print(f"  ⚠️  DER {der*100:.2f}% высокий, но структурных проблем нет")
        print(f"     Смотрите breakdown выше: какая компонента доминирует")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
