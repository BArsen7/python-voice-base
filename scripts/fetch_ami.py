#!/usr/bin/env python3
"""Download AMI SDM (single-distant-mic) test split → audio/ + RTTMs.

Pulls the SDM test parquet shards from ``diarizers-community/ami`` on
HuggingFace (CC-BY-4.0, no signup), extracts each meeting as a mono 16 kHz WAV
plus its RTTM ground-truth into ``audio/ami-<id>.wav``.

Usage:
    python fetch_ami.py [--limit N]

Limits memory by streaming row groups, writing each meeting to disk immediately.
AMI test = ~14 meetings, ~80 min each — full extraction is ~12 GB on disk.
Use ``--limit 3`` for a quick start.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

CORPUS_DIR = Path(__file__).resolve().parent
AUDIO_DIR = CORPUS_DIR / "audio"
PARQUET_CACHE = CORPUS_DIR / "_ami_cache"

# Public HF API endpoint — returns the list of shard URLs.
SHARDS_ENDPOINT = (
    "https://huggingface.co/api/datasets/diarizers-community/ami/parquet/sdm/test"
)

logger = logging.getLogger("fetch_ami")


def list_shards() -> list[str]:
    raw = urllib.request.urlopen(SHARDS_ENDPOINT, timeout=30).read()
    return json.loads(raw)


def download_shard(url: str, dest: Path) -> Path:
    """Download a parquet shard to disk (resume if size matches)."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("cached %s (%.1f MB)", dest.name, dest.stat().st_size / 1024 / 1024)
        return dest

    logger.info("downloading %s → %s", url, dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        with tmp.open("wb") as out:
            chunk = 1024 * 1024
            done = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total:
                    pct = done * 100 / total
                    sys.stdout.write(f"\r  {done/1024/1024:.0f}/{total/1024/1024:.0f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(dest)
    return dest


def extract_meeting(audio_bytes: bytes, name: str) -> Path:
    """Decode meeting audio (FLAC or WAV inside bytes), resample/downmix to mono 16k WAV."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_DIR / f"{name}.wav"
    sf.write(str(out), audio, sr, subtype="PCM_16")
    return out


def write_rttm(name: str, starts, ends, speakers) -> Path:
    """Write RTTM from parallel arrays (starts/ends in seconds, speakers = labels)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_DIR / f"{name}.rttm"
    with out.open("w", encoding="utf-8") as f:
        for tbeg, tdur, spk in zip(starts, ends, speakers):
            dur = float(tdur) - float(tbeg)
            if dur <= 0:
                continue
            f.write(
                f"SPEAKER {name} 1 {float(tbeg):.3f} {dur:.3f} "
                f"<NA> <NA> {spk} <NA> <NA>\n"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3, help="Max meetings to extract (default: 3)")
    ap.add_argument("--out-dir", type=str, default=None, help="Override output dir (default: script dir)")
    args = ap.parse_args()

    global AUDIO_DIR
    if args.out_dir:
        AUDIO_DIR = Path(args.out_dir)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    shard_urls = list_shards()
    logger.info("Found %d parquet shards", len(shard_urls))

    extracted = 0
    for i, url in enumerate(shard_urls):
        if extracted >= args.limit:
            break
        fname = url.rstrip("/").split("/")[-1]
        dest = PARQUET_CACHE / fname
        try:
            download_shard(url, dest)
        except Exception as e:
            logger.warning("Failed to download %s: %s", fname, e)
            continue

        table = pq.read_table(dest)
        cols = table.column_names
        logger.info("  shard %s: %d rows, columns: %s", fname, table.num_rows, cols)

        # diarizers-community/ami schema: 'audio' (dict), 'timestamps_start',
        # 'timestamps_end', 'speakers' — см. HF dataset card.
        if "audio" not in cols:
            logger.warning("No 'audio' column in %s — skip", fname)
            continue

        for row_idx in range(table.num_rows):
            if extracted >= args.limit:
                break
            row = table.slice(row_idx, 1).to_pylist()[0]
            audio_field = row["audio"]
            name = audio_field.get("path", f"ami-{fname}-{row_idx}").rsplit(".", 1)[0]

            try:
                wav_path = extract_meeting(audio_field["bytes"], name)
                rttm_path = write_rttm(
                    name,
                    row.get("timestamps_start", []),
                    row.get("timestamps_end", []),
                    row.get("speakers", []),
                )
            except Exception as e:
                logger.warning("  skip %s: %s", name, e)
                continue

            extracted += 1
            logger.info("  ✓ %s (+ %s)", wav_path.name, rttm_path.name)

    if extracted == 0:
        logger.error("Nothing extracted. Check HF dataset access / schema.")
        sys.exit(1)
    logger.info("Done: %d meetings → %s", extracted, AUDIO_DIR)


if __name__ == "__main__":
    main()
