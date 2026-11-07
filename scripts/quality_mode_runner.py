#!/usr/bin/env python3
"""Quality Mode runner: WAV → JSON с overlap-aware диаризацией."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pipeline.orchestrator import DiarizationPipeline, load_config, load_wav


def resolve_env_vars(cfg: dict) -> dict:
    """Рекурсивно подставляет ${VAR} из окружения."""
    if isinstance(cfg, dict):
        return {k: resolve_env_vars(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [resolve_env_vars(v) for v in cfg]
    if isinstance(cfg, str) and cfg.startswith("${") and cfg.endswith("}"):
        var = cfg[2:-1]
        return os.environ.get(var, "")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Путь к WAV")
    ap.add_argument("--config", default="configs/quality_mode.yaml")
    ap.add_argument("--out", default=None, help="Путь к выходному JSON (stdout если не указан)")
    ap.add_argument("--n-speakers", type=int, default=None, help="Oracle n_clusters")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_env_vars(cfg)
    if args.n_speakers is not None:
        cfg.setdefault("clustering", {})["ref_n_speakers"] = args.n_speakers
        cfg["clustering"]["n_clusters_mode"] = "oracle"

    pipe = DiarizationPipeline(cfg)
    waveform, sr = load_wav(args.audio, target_sr=cfg["pipeline"]["sample_rate"])
    entries = pipe.run(waveform, sr)

    out = {"transcript": [e.__dict__ for e in entries]}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
