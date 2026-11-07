#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"
OUT="data/calib_ru"
LIST="$OUT/list.txt"
rm -rf "$OUT"; mkdir -p "$OUT"

# --- detect resampler ---
resample() {
  local in="$1" out="$2"
  if command -v sox >/dev/null 2>&1; then
    sox -r 16000 -c 1 -b 16 "$in" "$out"
  elif command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "$in" -ar 16000 -ac 1 "$out"
  else
    echo "ERR: need sox or ffmpeg" >&2; exit 1
  fi
}
echo "[i] resampler: $(command -v sox >/dev/null 2>&1 && echo sox || echo ffmpeg)"

# --- 1) short_ru: 10 files, label=ru_real ---
echo "[*] short_ru -> 10 samples (ru_real)"
cnt=0
for wav in $(ls data/short_ru/*.wav | sort | head -10); do
  cp "$wav" "$OUT/ru_real_$(basename "$wav")"
  echo "$OUT/ru_real_$(basename "$wav") ru_real" >> "$LIST"
  cnt=$((cnt+1))
done
echo "[i]  ru_real: $cnt samples"

# --- 2) espeak 2 voices × 5 phrases ---
PHRASES=(
  "Сегодня мы обсудим новый проект и его сроки."
  "Мне нужно подготовить отчёт до конца недели."
  "Пожалуйста, отправьте мне документы по электронной почте."
  "Встреча перенесена на завтра на десять утра."
  "Какой у нас план на следующую итерацию?"
)
make_espeak() {  # $1=voice  $2=label
  local voice="$1"
  local label="$2"
  local raw="$OUT/tmp_${label}.wav"
  for i in "${!PHRASES[@]}"; do
    espeak-ng -v "$voice" -s 150 -w "$raw" "${PHRASES[$i]}"
    local out="$OUT/${label}_$(printf '%02d' "$i").wav"
    resample "$raw" "$out"
    echo "$out $label" >> "$LIST"
  done
  rm -f "$raw"
}
make_espeak "ru"    "espeak_ru"
make_espeak "ru+f3" "espeak_f3"
echo "[i]  espeak_ru + espeak_f3: 5+5 samples"

# --- 3) short_en: pure (single-speaker ≥90%) files → up to 5 per speaker ---
echo "[*] short_en -> pure files (Speaker_00 / Speaker_01)"
"$VENV/bin/python" - <<'PY'
import sys; sys.path.insert(0, ".")
from pathlib import Path

rttm_dir = Path("data/short_en")
out_dir  = Path("data/calib_ru")
list_f   = out_dir / "list.txt"

buckets = {"Speaker_00": [], "Speaker_01": []}
for rttm in sorted(rttm_dir.glob("*.rttm")):
    wav = rttm_dir / (rttm.stem + ".wav")
    if not wav.exists(): continue
    spk_dur = {}
    for line in rttm.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER": continue
        spk = parts[7]; dur = float(parts[4])
        spk_dur[spk] = spk_dur.get(spk, 0) + dur
    total = sum(spk_dur.values())
    if total <= 0: continue
    top_spk, top_dur = max(spk_dur.items(), key=lambda x: x[1])
    if top_dur / total >= 0.90 and top_spk in buckets:
        buckets[top_spk].append(wav)

import shutil
for spk, wavs in buckets.items():
    for wav in wavs[:5]:
        dst = out_dir / f"{spk.lower()}_{wav.name}"
        shutil.copy(wav, dst)
        with list_f.open("a") as f:
            f.write(f"{dst} {spk.lower()}\n")
    print(f"[i]  {spk.lower()}: {len(wavs[:5])} samples")
PY

# --- 4) summary ---
echo
echo "=== calibration set ==="
awk '{print $2}' "$LIST" | sort | uniq -c
echo "Total: $(wc -l < "$LIST") samples"
