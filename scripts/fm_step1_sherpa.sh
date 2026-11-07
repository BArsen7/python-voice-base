#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"
MODEL_DIR="models/sherpa-onnx-zipformer-ru-2025-04-20"
CFG="configs/baseline_ru.yaml"
BAK="configs/baseline_ru.yaml.pre-sherpa.bak"
OUT="results/fast_mode/sherpa"

[ -x "$VENV/bin/python" ] || { echo "ERR: no venv"; exit 1; }
PYV=$("$VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[ "$PYV" = "3.11" ] || { echo "ERR: need 3.11, got $PYV"; exit 1; }
"$VENV/bin/python" -c "import vadonnx" 2>/dev/null || "$VENV/bin/pip" install -q vadonnx
[ -f "$MODEL_DIR/tokens.txt" ] || { echo "ERR: model missing in $MODEL_DIR"; exit 1; }

rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT"

[ -f "$BAK" ] || { echo "ERR: $BAK missing"; exit 1; }
cp "$BAK" "$CFG"
echo "[i] config restored from $BAK"

SHERPA_STR=$(
  "$VENV/bin/python" - <<'PY'
import re, pathlib
src = pathlib.Path("scripts/pipeline/orchestrator.py").read_text()
lines = src.split("\n")
idx = next((i for i,l in enumerate(lines) if "import SherpaOnnxASR" in l), None)
if idx is None:
    print("sherpa_onnx_zipformer"); raise SystemExit(0)
for j in range(idx-1, max(0, idx-8), -1):
    m = re.findall(r'["\']([a-z_][a-z0-9_]*)["\']', lines[j])
    m = [h for h in m if "sherpa" in h.lower() or "zipformer" in h.lower()]
    if m:
        print(m[0]); raise SystemExit(0)
print("sherpa_onnx_zipformer")
PY
)
echo "[i] routing string for asr.model: '$SHERPA_STR'"

"$VENV/bin/python" - <<PY
import yaml
from pathlib import Path
p = Path("$CFG"); md = Path("$MODEL_DIR")
d = yaml.safe_load(open(p))
a = d["asr"]
a["model"]    = "$SHERPA_STR"
a["language"] = "ru"
a["encoder"]  = str(md / "encoder.onnx")
a["decoder"]  = str(md / "decoder.onnx")
a["joiner"]   = str(md / "joiner.onnx")
a["tokens"]   = str(md / "tokens.txt")
a.pop("word_timestamps", None)
a.pop("type", None)
yaml.safe_dump(d, open(p, "w"), allow_unicode=True, sort_keys=False)
print(f"[OK] asr.model='$SHERPA_STR', whisper keys removed")
print("     asr block:", a)
PY

# ---- helper: ВАЖНО, разбить local на отдельные строки ----
run_bench() {
  local dir="$1"
  local tag="$2"
  local subdir="$OUT/$tag"
  local log="$OUT/$tag.log"
  local stdout="$OUT/$tag.stdout"
  mkdir -p "$subdir"
  PS1='$ ' PROMPT_COMMAND='' "$VENV/bin/python" benchmark_diarization.py \
    --audio-dir "$dir" --configs "$CFG" \
    --min-ref-ratio 0.5 --uem-from-ref \
    --output "$subdir" \
    > "$stdout" 2> "$log" || {
      echo "ERR: benchmark rc=$? on $tag"; tail -40 "$log"; return 1
    }
  local json
  json=$(ls -t "$subdir"/*.json 2>/dev/null | head -1 || true)
  [ -n "$json" ] && [ -s "$json" ] || { echo "ERR: no JSON in $subdir"; ls -la "$subdir"; return 1; }
  cp "$json" "$OUT/$tag.result.json"
  echo "[OK] $tag -> $json"
}

# smoke
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

echo "[i] ASR classes seen in smoke:"
grep -oE "ASR=[A-Za-z_]+" "$OUT/smoke.log" | sort -u || true
if grep -q "ASR=FasterWhisperASR" "$OUT/smoke.log"; then
  echo "ERR: still FasterWhisperASR (routing '$SHERPA_STR' not matched)"; exit 1
fi

run_bench data/short_ru ru || exit 1

"$VENV/bin/python" - <<'PY'
import json
def load(t):
    try: d = json.load(open(f"results/fast_mode/{t}/ru.result.json"))
    except FileNotFoundError: return None
    return d[0] if isinstance(d, list) else d
b, s = load("baseline"), load("sherpa")
if s is None: raise SystemExit("no sherpa result")
print("[sherpa]  ", {k: s.get(k) for k in ("mean_der","mean_rtf","n_scored")})
if b:
    print("[baseline]", {k: b.get(k) for k in ("mean_der","mean_rtf","n_scored")})
    dd = (s.get("mean_der") or 0) - (b.get("mean_der") or 0)
    print(f"ΔDER={dd:+.2f}pp")
else:
    print("[i] baseline missing — only sherpa")
PY
