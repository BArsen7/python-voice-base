"""Calibrate FAISS cosine threshold from extracted embeddings."""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="results/fast_mode/calib_embeddings.npz")
    ap.add_argument("--out", default="results/fast_mode/faiss_calibration.json")
    ap.add_argument("--precision", type=float, default=0.95)
    ap.add_argument("--recall",    type=float, default=0.85)
    ap.add_argument("--current",   type=float, default=0.75)
    args = ap.parse_args()

    d = np.load(args.embeddings, allow_pickle=True)
    embs   = d["embeddings"].astype(np.float32)
    labels = d["labels"].tolist()

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs  = embs / norms

    same, diff = [], []
    for (i, a), (j, b) in itertools.combinations(enumerate(labels), 2):
        cos = float(embs[i] @ embs[j])
        (same if a == b else diff).append(cos)

    same = np.array(same); diff = np.array(diff)
    print(f"n_embeddings={len(labels)}  speakers={sorted(set(labels))}")
    if len(same):
        print(f"same-pairs (n={len(same)}): mean={same.mean():.4f} std={same.std():.4f} "
              f"p5={np.percentile(same,5):.4f} min={same.min():.4f}")
    if len(diff):
        print(f"diff-pairs (n={len(diff)}): mean={diff.mean():.4f} std={diff.std():.4f} "
              f"p95={np.percentile(diff,95):.4f} max={diff.max():.4f}")

    rows = []
    for th in np.linspace(0.30, 0.99, 70):
        tp = int((same >= th).sum()); fn = int((same < th).sum())
        fp = int((diff >= th).sum()); tn = int((diff < th).sum())
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec  = tp / (tp + fn) if tp + fn else 1.0
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({"threshold": float(th), "precision": float(prec),
                     "recall": float(rec), "f1": float(f1),
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    best = None
    for r in rows:
        if r["precision"] >= args.precision and r["recall"] >= args.recall:
            if best is None or r["f1"] > best["f1"]:
                best = r
    best_f1 = max(rows, key=lambda r: r["f1"])
    cur = next((r for r in rows if abs(r["threshold"] - args.current) < 0.01), None)

    res = {
        "n_embeddings": int(len(labels)),
        "unique_speakers": sorted(set(labels)),
        "n_same_pairs": int(len(same)),
        "n_diff_pairs": int(len(diff)),
        "same": {"mean": float(same.mean()), "std": float(same.std()),
                 "p5": float(np.percentile(same,5)),
                 "min": float(same.min())} if len(same) else None,
        "diff": {"mean": float(diff.mean()), "std": float(diff.std()),
                 "p95": float(np.percentile(diff,95)),
                 "max": float(diff.max())} if len(diff) else None,
        "current_threshold": args.current,
        "current_metrics": cur,
        "best_under_constraints": best,
        "best_f1_unconstrained": best_f1,
        "targets": {"precision": args.precision, "recall": args.recall},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)

    print(f"\n[current th={args.current}]  "
          f"{'P='+str(round(cur['precision'],3)) if cur else 'N/A'}  "
          f"{'R='+str(round(cur['recall'],3)) if cur else 'N/A'}")
    print(f"[best under P>={args.precision} R>={args.recall}]  {best}")
    print(f"[best F1 overall]  th={best_f1['threshold']:.3f}  "
          f"P={best_f1['precision']:.3f}  R={best_f1['recall']:.3f}  "
          f"F1={best_f1['f1']:.3f}")

    if best is None:
        print("\n[!] no threshold meets targets.")
        if len(same) and len(diff):
            print(f"    same.min={same.min():.4f}  diff.max={diff.max():.4f}  "
                  f"overlap={'YES' if diff.max() >= same.min() else 'no'}")
        sys.exit(4)


if __name__ == "__main__":
    main()
