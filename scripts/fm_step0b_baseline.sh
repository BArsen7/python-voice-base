#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"
BAK="configs/baseline_ru.yaml.pre-sherpa.bak"
CFG_EN="configs/baseline_en.yaml"
OUT="results/fast_mode/baseline"
mkdir -p "$OUT"

[ -x "$VENV/bin/python" ] || { echo "ERR: no venv"; exit 1; }
PYV=$("$VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[ "$PYV" = "3.11" ] || { echo "ERR: need 3.11"; exit 1; }
[ -f "$BAK" ] || { echo "ERR: $BAK missing"; exit 1; }

run_bench() {  # $1=dir $2=cfg $3=tag
  local dir="$1"
  local cfg="$2"
  local tag="$3"
  local subdir="$OUT/$tag"
  mkdir -p "$subdir"
  echo "[*] baseline $tag: $dir × $cfg"
  PS1='$ ' PROMPT_COMMAND='' "$VENV/bin/python" benchmark_diarization.py \
    --audio-dir "$dir" --configs "$cfg" \
    --min-ref-ratio 0.5 --uem-from-ref \
    --output "$subdir" \
    > "$OUT/$tag.stdout" 2> "$OUT/$tag.log" || {
      echo "ERR rc=$? on $tag"; tail -40 "$OUT/$tag.log"; return 1
    }
  local json
  json=$(ls -t "$subdir"/*.json 2>/dev/null | head -1 || true)
  [ -n "$json" ] || { echo "ERR: no json in $subdir"; return 1; }
  cp "$json" "$OUT/$tag.result.json"
  echo "[OK] $tag -> $json"
}

run_bench data/short_ru "$BAK" ru
run_bench data/short_en "$CFG_EN" en

# сводка через universal loader
"$VENV/bin/python" - <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts._metrics_loader import load_metrics
from pathlib import Path

print("\n=== BASELINE (same flags: --uem-from-ref --min-ref-ratio 0.5) ===")
rows = []
for tag in ("ru", "en"):
    p = Path(f"results/fast_mode/baseline/{tag}.result.json")
    if not p.exists():
        print(f"{tag}: missing"); continue
    m = load_metrics(p)
    rows.append((tag, m))
    print(f"{tag.upper():3s}  DER={m['mean_der_pct']:.2f}%  "
          f"RTF={m['mean_rtf']:.3f}  n={m['n_scored']}  "
          f"spk_err={m.get('mean_speakers_error')}")
json.dump({tag: {k: v for k, v in m.items() if k != "raw"} for tag, m in rows},
          open("results/fast_mode/baseline/summary.json", "w"),
          indent=2, ensure_ascii=False)
PY
