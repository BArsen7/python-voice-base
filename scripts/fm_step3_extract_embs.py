"""Extract per-file embeddings via WeSpeakerEmbeddingExtractor's real API."""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, ".")
from scripts.models.embeddings import WeSpeakerEmbeddingExtractor
from scripts.core.interfaces import SpeechSegment


def embed_file(extractor, wav_path: Path, sr: int = 16000):
    wav, actual_sr = sf.read(str(wav_path), dtype="float32")
    if actual_sr != sr:
        raise ValueError(f"{wav_path.name}: sr={actual_sr}, expected {sr}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < int(0.3 * sr):
        return None
    seg = SpeechSegment(start_ms=0, end_ms=int(len(wav) * 1000 / sr))
    embs, _ = extractor.extract(wav, sr, [seg])
    if embs.shape[0] == 0:
        return None
    pooled = embs.mean(axis=0)
    pooled /= (np.linalg.norm(pooled) + 1e-9)
    return pooled.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="data/calib_ru/list.txt")
    ap.add_argument("--config", default="configs/baseline_ru.yaml")
    ap.add_argument("--out", default="results/fast_mode/calib_embeddings.npz")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["embeddings"]
    print(f"[i] extractor: use_onnx={cfg.get('use_onnx')}, model={cfg.get('model')}")
    ext = WeSpeakerEmbeddingExtractor(cfg)
    ext._lazy_load()

    embs, labels, paths = [], [], []
    for line in Path(args.list).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        wav_path, spk = s.rsplit(maxsplit=1)
        try:
            e = embed_file(ext, Path(wav_path))
        except Exception as ex:
            print(f"[!] {wav_path}: {ex}", file=sys.stderr); continue
        if e is None:
            print(f"[!] {wav_path}: too short", file=sys.stderr); continue
        embs.append(e); labels.append(spk); paths.append(wav_path)
        print(f"[OK] {Path(wav_path).name:35s}  spk={spk:12s}  norm={np.linalg.norm(e):.4f}")

    if not embs:
        raise SystemExit("ERR: no embeddings extracted")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        embeddings=np.stack(embs),
                        labels=np.array(labels),
                        paths=np.array(paths))
    print(f"\n[OK] saved {len(embs)} embeddings x {len(embs[0])}-dim -> {out}")


if __name__ == "__main__":
    main()
