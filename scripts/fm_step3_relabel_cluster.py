"""Re-derive ground-truth speaker labels via clustering.

Problem: data/short_ru rttm labels all files 'spk0' but acoustically
they are different speakers. Same for speaker_00 in short_en.

Solution: cluster within each *rttm* label group using cosine similarity
threshold. espeak_* groups are trusted as-is (mean cos ~0.8).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_group(E_sub: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative cosine clustering, distance = 1 - cos."""
    n = len(E_sub)
    if n <= 1:
        return np.zeros(n, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(E_sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="results/fast_mode/calib_embeddings.npz")
    ap.add_argument("--out-npz", default="results/fast_mode/calib_embeddings_relabel.npz")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="cosine threshold for 'same speaker' within group")
    ap.add_argument("--trusted", nargs="+", default=["espeak_ru", "espeak_f3"],
                    help="labels trusted as-is (skip clustering)")
    args = ap.parse_args()

    d = np.load(args.embeddings, allow_pickle=True)
    E = d["embeddings"].astype(np.float32)
    L = d["labels"].tolist()
    P = d["paths"].tolist()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(L):
        groups[lab].append(i)

    new_labels = list(L)
    print(f"threshold={args.threshold}  trusted={args.trusted}\n")
    print(f"{'group':14s}  {'n':>2s}  {'clusters':>8s}  {'new labels'}")
    for lab, idx in sorted(groups.items()):
        if lab in args.trusted:
            print(f"{lab:14s}  {len(idx):2d}  {1:>8d}  (trusted, unchanged)")
            continue
        E_sub = E[idx]
        pred = cluster_group(E_sub, args.threshold)
        n_clusters = int(pred.max()) + 1
        assigned = []
        for k in range(n_clusters):
            for i_local, i_global in enumerate(idx):
                if pred[i_local] == k:
                    new_labels[i_global] = f"{lab}_c{k}"
                    assigned.append(f"c{k}")
        sizes = [int((pred == k).sum()) for k in range(n_clusters)]
        print(f"{lab:14s}  {len(idx):2d}  {n_clusters:>8d}  sizes={sizes}")

    # sanity
    unique_new = sorted(set(new_labels))
    print(f"\nunique speakers after relabel: {len(unique_new)}")
    print(f"  {unique_new}")

    # pairwise summary on new labels
    import itertools
    same, diff = [], []
    for i, j in itertools.combinations(range(len(new_labels)), 2):
        cos = float(E[i] @ E[j])
        (same if new_labels[i] == new_labels[j] else diff).append(cos)
    same_a = np.array(same); diff_a = np.array(diff)
    print(f"\nsame-pairs: n={len(same_a):4d}  mean={same_a.mean():.3f}  "
          f"min={same_a.min():.3f}  p5={np.percentile(same_a,5):.3f}")
    print(f"diff-pairs: n={len(diff_a):4d}  mean={diff_a.mean():.3f}  "
          f"max={diff_a.max():.3f}  p95={np.percentile(diff_a,95):.3f}")
    if diff_a.max() >= same_a.min():
        print(f"  overlap: YES (diff.max={diff_a.max():.3f} >= same.min={same_a.min():.3f})")
    else:
        print(f"  overlap: no  (gap = {same_a.min() - diff_a.max():.3f})")

    out = Path(args.out_npz); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        embeddings=d["embeddings"],
                        labels=np.array(new_labels),
                        paths=np.array(P))
    print(f"\n[OK] saved relabeled -> {out}")


if __name__ == "__main__":
    main()
