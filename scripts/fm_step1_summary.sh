#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=".venv"

"$VENV/bin/python" - <<'PY'
import sys, json
sys.path.insert(0, ".")
from scripts._metrics_loader import load_metrics
from pathlib import Path

def load(tag, step):
    p = Path(f"results/fast_mode/{step}/{tag}.result.json")
    return load_metrics(p) if p.exists() else None

print("| Step | Lang | DER % | RTF | n | spk_err |")
print("|---|---|---|---|---|---|")
data = {}
for step in ("baseline", "sherpa"):
    for lang in ("ru", "en"):
        m = load(lang, step)
        if not m: continue
        data[(step, lang)] = m
        print(f"| {step} | {lang.upper()} | "
              f"{m['mean_der_pct']:.2f} | {m['mean_rtf']:.3f} | "
              f"{m['n_scored']} | {m.get('mean_speakers_error')} |")

print()
for lang in ("ru", "en"):
    b = data.get(("baseline", lang)); s = data.get(("sherpa", lang))
    if not (b and s): continue
    dd = s["mean_der_pct"] - b["mean_der_pct"]
    dr = (s["mean_rtf"] or 1e-9) / (b["mean_rtf"] or 1)
    print(f"{lang.upper()}: ΔDER={dd:+.2f}pp   "
          f"RTF {b['mean_rtf']:.3f} → {s['mean_rtf']:.3f}  "
          f"({1/dr:.2f}× speedup)")
    # Acceptance check for step 1
    if lang == "ru":
        print(f"  target DER ≤ 12%: {'PASS' if s['mean_der_pct'] <= 12 else 'FAIL'}")
        print(f"  target RTF ≤ 0.20: {'PASS' if s['mean_rtf'] <= 0.20 else 'FAIL'}")
PY
