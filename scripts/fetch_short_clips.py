#!/usr/bin/env python3
"""
fetch_short_clips.py — скачать короткие клипы RU (mexus/ru-book-mix-10h)
и EN (noty7gian/synthetic-multilingual-speaker-diarization), сохранить как WAV+RTTM.

Использование:
    python scripts/fetch_short_clips.py --lang ru --limit 30
    python scripts/fetch_short_clips.py --lang en --limit 50
    python scripts/fetch_short_clips.py --lang ru en --limit 30
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import tarfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fetch_short")


# ---------------------------------------------------------------------------
# Русский: mexus/ru-book-mix-10h (WebDataset tar, FLAC + RTTM)
# ---------------------------------------------------------------------------

def fetch_russian(out_dir: Path, limit: int) -> int:
    from huggingface_hub import hf_hub_download

    repo_id = "mexus/ru-book-mix-10h"
    logger.info("Скачиваю shard-0000.tar из %s...", repo_id)
    try:
        tar_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/shard-0000.tar",
            repo_type="dataset",
        )
    except Exception as e:
        logger.error("Не удалось скачать shard: %s", e)
        return 0

    saved = 0
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        flac_members = [m for m in members if m.name.endswith(".flac")]
        rttm_members = [m for m in members if m.name.endswith(".rttm")]
        rttm_by_id = {Path(m.name).stem: m for m in rttm_members}

        for flac_m in flac_members[:limit]:
            key = Path(flac_m.name).stem
            if key not in rttm_by_id:
                logger.warning("Нет RTTM для %s — пропускаю", key)
                continue

            flac_bytes = tar.extractfile(flac_m).read()
            audio, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)
            if sr != 16000:
                from scipy.signal import resample_poly
                g = gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
                sr = 16000

            wav_path = out_dir / f"ru_{key}.wav"
            sf.write(str(wav_path), audio, sr, subtype="PCM_16")

            rttm_bytes = tar.extractfile(rttm_by_id[key]).read()
            (out_dir / f"ru_{key}.rttm").write_bytes(rttm_bytes)

            saved += 1
            if saved % 10 == 0:
                logger.info("  RU: сохранено %d клипов...", saved)

    logger.info("RU готово: %d клипов → %s", saved, out_dir)
    return saved


# ---------------------------------------------------------------------------
# Английский: noty7gian/synthetic-multilingual-speaker-diarization (CSV + WAV)
# ---------------------------------------------------------------------------

def fetch_english(out_dir: Path, limit: int) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = "noty7gian/synthetic-multilingual-speaker-diarization"
    logger.info("Скачиваю аннотации из %s...", repo_id)

    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="all_visible_combined.csv",
            repo_type="dataset",
        )
    except Exception as e:
        logger.error("Не удалось скачать CSV: %s", e)
        return 0

    df = pd.read_csv(csv_path)
    lang_col = next((c for c in ("Language", "language") if c in df.columns), None)
    if lang_col:
        df = df[df[lang_col].astype(str).str.lower() == "en"]

    if df.empty:
        logger.error("В CSV нет английских строк — проверь колонку языка")
        return 0

    file_col = next(
        (c for c in ("AudioFileName", "audio_file", "filename") if c in df.columns),
        None,
    )
    if file_col is None:
        logger.error("В CSV нет колонки с именем файла")
        return 0

    unique_files = df[file_col].unique()[:limit]
    saved = 0

    for i, fname in enumerate(unique_files):
        rows = df[df[file_col] == fname]
        try:
            wav_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"audio/{fname}",
                repo_type="dataset",
            )
        except Exception as e:
            logger.warning("Не удалось скачать %s: %s", fname, e)
            continue

        base = f"en_{i:04d}"
        shutil.copy(wav_path, out_dir / f"{base}.wav")

        with (out_dir / f"{base}.rttm").open("w", encoding="utf-8") as f:
            for _, row in rows.iterrows():
                start = float(row["StartTS"])
                end = float(row["EndTS"])
                dur = end - start
                if dur <= 0:
                    continue
                f.write(
                    f"SPEAKER {base} 1 {start:.3f} {dur:.3f} "
                    f"<NA> <NA> {row['Speaker']} <NA> <NA>\n"
                )
        saved += 1
        if saved % 10 == 0:
            logger.info("  EN: сохранено %d клипов...", saved)

    logger.info("EN готово: %d клипов → %s", saved, out_dir)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lang", nargs="+", default=["ru", "en"], choices=["ru", "en"],
        help="Какие языки скачивать (default: ru en)",
    )
    ap.add_argument("--limit", type=int, default=30,
                    help="Максимум клипов на язык (default: 30)")
    ap.add_argument("--out-dir", default="data/short",
                    help="Куда сохранять WAV+RTTM")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    if "ru" in args.lang:
        total += fetch_russian(out_dir, args.limit)
    if "en" in args.lang:
        total += fetch_english(out_dir, args.limit)

    if total == 0:
        logger.error("Ничего не сохранили. Проверь доступ к датасетам.")
        raise SystemExit(1)

    logger.info("Итого: %d клипов в %s", total, out_dir)


if __name__ == "__main__":
    main()
