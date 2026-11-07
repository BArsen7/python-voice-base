#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=".venv"
OUT="models/wespeaker_resnet34.onnx"
mkdir -p models

# 1) скачать готовый ONNX (idempotent)
if [ ! -f "$OUT" ]; then
  echo "[*] downloading speaker-embedding.onnx from HF"
  "$VENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(
    repo_id="hbredin/wespeaker-voxceleb-resnet34-LM",
    filename="speaker-embedding.onnx",
)
shutil.copy(p, "models/wespeaker_resnet34.onnx")
print("[OK] saved to models/wespeaker_resnet34.onnx")
PY
fi

[ -f "$OUT" ] || { echo "ERR: $OUT not found"; exit 1; }
ls -lh "$OUT"

# 2) валидация: ONNX vs pyannote-обёртка (PyTorch-путь)
"$VENV/bin/python" - <<'PY'
from __future__ import annotations
import sys, numpy as np
sys.path.insert(0, ".")

import torch
import onnxruntime as ort
from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding,
    ONNXWeSpeakerPretrainedSpeakerEmbedding,
)

ONNX_PATH = "models/wespeaker_resnet34.onnx"

# reference: pyannote с HF (может использовать ONNX под капотом)
ref = PretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")
# target: наш локальный ONNX
ext = ONNXWeSpeakerPretrainedSpeakerEmbedding(ONNX_PATH)

# прямой onnxruntime для проверки входа/выхода
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print(f"[i] ONNX input: name={inp.name} shape={inp.shape} dtype={inp.type}")

fails = 0
for i in range(5):
    wav = torch.randn(1, 1, 16000 * 5)  # (batch, channel, time)
    with torch.no_grad():
        e_ref = ref(wav)
        e_ref = e_ref.cpu().numpy() if hasattr(e_ref, "cpu") else np.asarray(e_ref)
        e_loc = ext(wav)
        e_loc = e_loc.cpu().numpy() if hasattr(e_loc, "cpu") else np.asarray(e_loc)

    rn = e_ref / (np.linalg.norm(e_ref, axis=-1, keepdims=True) + 1e-9)
    gn = e_loc / (np.linalg.norm(e_loc, axis=-1, keepdims=True) + 1e-9)
    cos = float((rn * gn).sum())
    close = bool(np.allclose(e_ref, e_loc, atol=1e-4))
    print(f"  win {i}: cos={cos:.6f}  allclose(atol=1e-4)={close}")
    if not close and cos < 0.9999:
        fails += 1

if fails:
    print(f"ERR: {fails}/5 windows outside tolerance")
    sys.exit(3)
print("[OK] validation passed: local ONNX == pyannote reference")
PY
