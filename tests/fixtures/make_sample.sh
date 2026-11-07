#!/usr/bin/env bash
# Генерирует синтетический sample.wav (sine 440Hz + тишина + sine 440Hz, 7 с).
# Годится только для smoke ab_test.py — VAD на чистом тоне вернёт 0 сегментов,
# для проверки качества распознавания используйте реальный speech.wav.
set -euo pipefail
cd "$(dirname "$0")"
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" \
       -f lavfi -i "anullsrc=r=16000:cl=mono:d=1" \
       -filter_complex "[0:a][1:a][0:a]concat=n=3:v=0:a=1" \
       -ar 16000 -ac 1 sample.wav -y
echo "OK: $(pwd)/sample.wav"
