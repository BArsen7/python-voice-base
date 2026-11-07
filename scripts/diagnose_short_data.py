#!/usr/bin/env python3
"""diagnose_short_data.py — проверка RTTM vs WAV: единицы, ratio, аномалии.

Usage:
    python scripts/diagnose_short_data.py data/short_en
    python scripts/diagnose_short_data.py data/short_ru --fix  # конвертирует ms→s
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import soundfile as sf


def load_rttm_stats(path: Path):
    total, max_end, n = 0.0, 0.0, 0
    speakers = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            tbeg, tdur = float(p[3]), float(p[4])
            total += tdur
            max_end = max(max_end, tbeg + tdur)
            speakers.add(p[7])
            n += 1
    return dict(total=total, max_end=max_end, n=n, speakers=sorted(speakers))


def diagnose_dir(audio_dir: Path, fix: bool = False) -> dict:
    rows = []
    needs_ms = []
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        print(f"  ⚠ нет *.wav в {audio_dir}")
        return dict(n=0, ms_likely=0, sparse=0, ok=0)

    print(f"\n{'file':<20} {'audio_s':>9} {'ref_s':>9} {'ratio':>7} {'N':>4} {'verdict':<12}")
    print("-" * 70)
    for wav in wavs:
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            print(f"{wav.name:<20} {'':>9} {'':>9} {'':>7} {'':>4} NO_RTTM")
            continue
        info = sf.info(str(wav))
        wav_dur = info.frames / info.samplerate
        st = load_rttm_stats(rttm)
        ratio = st["total"] / max(wav_dur, 1e-9)
        verdict, flag = "ok", ""
        if ratio > 1.5 or st["max_end"] > wav_dur * 1.5:
            verdict = "MS_LIKELY"
            needs_ms.append(rttm)
        elif ratio < 0.3:
            verdict = "SPARSE"
        print(
            f"{wav.name:<20} {wav_dur:>9.2f} {st['total']:>9.2f} "
            f"{ratio:>7.2f} {st['n']:>4} {verdict:<12}{flag}"
        )
        rows.append(dict(wav=wav, rttm=rttm, dur=wav_dur, ratio=ratio, verdict=verdict))

    summary = dict(
        n=len(rows),
        ms_likely=sum(1 for r in rows if r["verdict"] == "MS_LIKELY"),
        sparse=sum(1 for r in rows if r["verdict"] == "SPARSE"),
        ok=sum(1 for r in rows if r["verdict"] == "ok"),
    )

    if fix and needs_ms:
        print(f"\n  FIX: конвертирую {len(needs_ms)} RTTM из ms в s")
        for rttm in needs_ms:
            bak = rttm.with_suffix(".rttm.bak")
            if not bak.exists():
                shutil.copy(rttm, bak)
            lines = []
            for line in rttm.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) >= 9 and p[0] == "SPEAKER":
                    p[3] = f"{float(p[3])/1000:.3f}"
                    p[4] = f"{float(p[4])/1000:.3f}"
                    lines.append(" ".join(p))
                else:
                    lines.append(line)
            rttm.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"    ✓ {rttm.name}")
    elif needs_ms:
        print(f"\n  ⚠ {len(needs_ms)} файлов с таймстемпами в ms — запусти с --fix")

    print(f"\n  Итого: {summary['ok']} ok, {summary['sparse']} sparse, {summary['ms_likely']} ms_likely")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("--fix", action="store_true", help="Конвертировать ms→s на месте (с бэкапом)")
    args = ap.parse_args()

    if not args.audio_dir.is_dir():
        sys.exit(f"не директория: {args.audio_dir}")
    print(f"=== {args.audio_dir} ===")
    diagnose_dir(args.audio_dir, fix=args.fix)


if __name__ == "__main__":
    main()
