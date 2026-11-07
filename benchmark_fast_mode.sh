#!/usr/bin/env bash
set -euo pipefail

# script is at repo root
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV=".venv"
ROOT="results/fast_mode/final"
CFG_RU="configs/baseline_ru.yaml"
CFG_EN="configs/baseline_en.yaml"

[ -x "$VENV/bin/python" ] || { echo "ERR: $VENV/bin/python not found in $(pwd)"; exit 1; }
mkdir -p "$ROOT/ru" "$ROOT/en"

run_bench() {
  local dir="$1"
  local cfg="$2"
  local tag="$3"
  local subdir="$ROOT/$tag"
  PS1='$ ' PROMPT_COMMAND='' "$VENV/bin/python" benchmark_diarization.py \
    --audio-dir "$dir" --configs "$cfg" \
    --min-ref-ratio 0.5 --uem-from-ref \
    --output "$subdir" \
    > "$ROOT/$tag.stdout" 2> "$ROOT/$tag.log" || {
      echo "ERR: rc=$? on $tag"; tail -30 "$ROOT/$tag.log"; return 1
    }
  local json
  json=$(ls -t "$subdir"/*.json 2>/dev/null | head -1 || true)
  [ -n "$json" ] || { echo "ERR: no json in $subdir"; return 1; }
  cp "$json" "$ROOT/$tag.result.json"
  echo "[OK] $tag -> $json"
}

run_bench data/short_ru "$CFG_RU" ru
run_bench data/short_en "$CFG_EN" en

"$VENV/bin/python" - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts._metrics_loader import load_metrics
from pathlib import Path

rows = []
for tag in ("ru", "en"):
    p = Path(f"results/fast_mode/final/{tag}.result.json")
    if not p.exists(): continue
    m = load_metrics(p)
    rows.append((tag.upper(), m["mean_der_pct"], m["mean_rtf"],
                 m["n_scored"], m.get("mean_speakers_error")))

print("\n## Final (fast mode: Sherpa-Zipformer + ONNX-WeSpeaker)")
print("| Lang | DER % | RTF | n | spk_err |")
print("|---|---|---|---|---|")
for lang, der, rtf, n, spk_err in rows:
    spk = "—" if spk_err is None else f"{spk_err:.3f}"
    print(f"| {lang} | {der:.2f} | {rtf:.3f} | {n} | {spk} |")
PY
