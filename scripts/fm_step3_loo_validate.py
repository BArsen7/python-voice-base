"""Leave-one-out validation: honest P/R for speaker identification.

For each embedding p:
  - build registry from all OTHER embeddings
  - query p at threshold τ
  - cases:
      * top match same-label  & score ≥ τ → TP
      * top match same-label  & score < τ → FN (own speaker not recognized)
      * top match diff-label  & score ≥ τ → FP (wrong speaker)
      * top match diff-label  & score < τ → correct "unknown"  (not counted in P/R)
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def loo_eval(E: np.ndarray, L: list, tau: float):
    n = len(E)
    S = E @ E.T   # cosine similarity matrix (already L2-normed)
    np.fill_diagonal(S, -np.inf)

    # exclude self, exclude same-file if path known — handled by caller
    tp = fp = fn = 0
    for i in range(n):
        j = int(np.argmax(S[i]))   # best match among all others
        best = float(S[i, j])
        if best >= tau:
            if L[j] == L[i]:
                tp += 1
            else:
                fp += 1
        else:
            # no match above τ
            # is p a "known" speaker whose own identity was in registry?
            # In our data — yes, every embedding's speaker exists among others
            has_own = any(L[k] == L[i] for k in range(n) if k != i)
            if has_own:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"threshold": float(tau), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="results/fast_mode/calib_embeddings_relabel.npz")
    ap.add_argument("--out", default="results/fast_mode/faiss_loo_validation.json")
    ap.add_argument("--precision", type=float, default=0.95)
    ap.add_argument("--recall",    type=float, default=0.85)
    args = ap.parse_args()

    d = np.load(args.embeddings, allow_pickle=True)
    E = d["embeddings"].astype(np.float32)
    L = d["labels"].tolist()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    rows = [loo_eval(E, L, tau) for tau in np.linspace(0.30, 0.90, 61)]

    best = None
    for r in rows:
        if r["precision"] >= args.precision and r["recall"] >= args.recall:
            if best is None or r["f1"] > best["f1"]:
                best = r
    best_f1 = max(rows, key=lambda r: r["f1"])

    print(f"n_embeddings={len(E)}  speakers={len(set(L))}")
    print(f"{'th':>5s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  TP FP FN")
    for r in rows[::5]:
        print(f"{r['threshold']:5.2f}  {r['precision']:6.3f}  "
              f"{r['recall']:6.3f}  {r['f1']:6.3f}  "
              f"{r['tp']:2d} {r['fp']:2d} {r['fn']:2d}")

    print(f"\n[LOO best under P≥{args.precision} R≥{args.recall}]  {best}")
    print(f"[LOO best F1]  th={best_f1['threshold']:.3f}  "
          f"P={best_f1['precision']:.3f}  R={best_f1['recall']:.3f}  "
          f"F1={best_f1['f1']:.3f}")

    res = {
        "n_embeddings": int(len(E)),
        "n_unique_speakers": int(len(set(L))),
        "sweep": rows,
        "best_under_constraints": best,
        "best_f1": best_f1,
        "targets": {"precision": args.precision, "recall": args.recall},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[OK] saved -> {args.out}")


if __name__ == "__main__":
    main()
