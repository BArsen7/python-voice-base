#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"
CFG="configs/baseline_ru.yaml"
BAK="configs/baseline_ru.yaml.pre-onnxemb.bak"
ONNX_PATH="models/wespeaker_resnet34.onnx"
OUT="results/fast_mode/onnx_emb"

[ -x "$VENV/bin/python" ] || { echo "ERR: no venv"; exit 1; }
PYV=$("$VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[ "$PYV" = "3.11" ] || { echo "ERR: need 3.11, got $PYV"; exit 1; }
[ -f "$ONNX_PATH" ] || { echo "ERR: $ONNX_PATH missing — run fm_step2_download_onnx.sh first"; exit 1; }

# verify embeddings.py has use_onnx support
if ! grep -q "use_onnx" scripts/models/embeddings.py; then
  echo "ERR: scripts/models/embeddings.py does not support use_onnx."
  echo "     Apply the patch to _lazy_load first (see response)."
  exit 2
fi

rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT"

# --- patch config ---
[ -f "$BAK" ] || cp "$CFG" "$BAK"
"$VENV/bin/python" - <<PY
import yaml
from pathlib import Path
p = Path("$CFG")
d = yaml.safe_load(open(p))
e = d["embeddings"]
e["use_onnx"]  = True
e["onnx_path"] = "$ONNX_PATH"
yaml.safe_dump(d, open(p, "w"), allow_unicode=True, sort_keys=False)
print(f"[OK] patched {p}: use_onnx=True, onnx_path={e['onnx_path']}")
print(f"     embeddings block: {e}")
PY

# --- benchmark helper ---
run_bench() {
  local dir="$1"
  local tag="$2"
  local subdir="$OUT/$tag"
  mkdir -p "$subdir"
  PS1='$ ' PROMPT_COMMAND='' "$VENV/bin/python" benchmark_diarization.py \
    --audio-dir "$dir" --configs "$CFG" \
    --min-ref-ratio 0.5 --uem-from-ref \
    --output "$subdir" \
    > "$OUT/$tag.stdout" 2> "$OUT/$tag.log" || {
      echo "ERR: benchmark rc=$? on $tag"; tail -40 "$OUT/$tag.log"; return 1
    }
  local json
  json=$(ls -t "$subdir"/*.json 2>/dev/null | head -1 || true)
  [ -n "$json" ] && [ -s "$json" ] || { echo "ERR: no JSON in $subdir"; return 1; }
  cp "$json" "$OUT/$tag.result.json"
  echo "[OK] $tag -> $json"
}

# --- smoke (3 files) ---
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cnt=0
while IFS= read -r wav; do
  base=$(basename "$wav" .wav)
  [ -f "data/short_ru/$base.rttm" ] || continue
  cp "$wav" "data/short_ru/$base.rttm" "$TMP/"
  cnt=$((cnt+1)); [ "$cnt" -ge 3 ] && break
done < <(find data/short_ru -maxdepth 1 -name '*.wav' | sort)
[ "$cnt" -ge 1 ] || { echo "ERR: no wav+rttm"; exit 1; }

run_bench "$TMP" smoke || exit 1

# verify ONNX path took effect
if grep -q "ONNX loaded from" "$OUT/smoke.log"; then
  echo "[OK] ONNX embeddings active"
else
  echo "WARN: 'ONNX loaded from' not in smoke.log — verify _lazy_load"
  grep -E "WeSpeaker" "$OUT/smoke.log" | head -3
fi

# --- full RU ---
run_bench data/short_ru ru || exit 1

# --- compare vs step1 (sherpa) ---
"$VENV/bin/python" - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts._metrics_loader import load_metrics
from pathlib import Path

steps = ["sherpa", "onnx_emb"]
print("\n| Step | DER % | RTF | n |")
print("|---|---|---|---|")
prev = None
for step in steps:
    p = Path(f"results/fast_mode/{step}/ru.result.json")
    if not p.exists():
        print(f"| {step} | missing | | |"); continue
    m = load_metrics(p)
    print(f"| {step} | {m['mean_der_pct']:.2f} | {m['mean_rtf']:.3f} | {m['n_scored']} |")
    if step == "onnx_emb" and prev:
        dd = m["mean_der_pct"] - prev["mean_der_pct"]
        dr = (prev["mean_rtf"] or 1e-9) / (m["mean_rtf"] or 1)
        print(f"\nΔDER={dd:+.2f}pp   RTF speedup={dr:.2f}×")
        print(f"target RTF ≤ 0.20: {'PASS' if m['mean_rtf'] <= 0.20 else 'FAIL'}")
    prev = m
PY
