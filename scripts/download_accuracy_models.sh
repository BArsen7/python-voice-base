#!/usr/bin/env bash
# Загрузка ONNX-моделей для Accuracy Mode в models_bundle/.
# Использование: bash scripts/download_accuracy_models.sh [TARGET_DIR]
set -euo pipefail

MODELS_DIR="${1:-models_bundle}"
WHISPER_DIR="$MODELS_DIR/whisper-medium"
WESPEAKER_DIR="$MODELS_DIR/wespeaker"
mkdir -p "$WHISPER_DIR" "$WESPEAKER_DIR"

# --- Определяем, чем качать: hf (новый) -> huggingface-cli (старый) -> wget/curl
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
fi

have() { command -v "$1" >/dev/null 2>&1; }

# Универсальный загрузчик HF-файла
# $1 = repo, $2 = file, $3 = dest
download_hf_file() {
    local repo="$1" file="$2" dest="$3"
    if [ -f "$dest" ]; then
        echo "  [skip] $file (уже есть)"
        return 0
    fi
    echo "  -> $file"
    if [ "$HF_CMD" = "hf" ]; then
        hf download "$repo" "$file" --local-dir "$(dirname "$dest")" >/dev/null
    elif [ "$HF_CMD" = "huggingface-cli" ]; then
        huggingface-cli download "$repo" "$file" --local-dir "$(dirname "$dest")" >/dev/null
    elif have wget; then
        wget -q -O "$dest" "https://huggingface.co/$repo/resolve/main/$file"
    elif have curl; then
        curl -fsSL -o "$dest" "https://huggingface.co/$repo/resolve/main/$file"
    else
        echo "  [ERROR] нет ни hf, ни wget, ни curl" >&2
        return 1
    fi
    if [ ! -s "$dest" ]; then
        echo "  [ERROR] $file не появился или пуст после скачивания" >&2
        return 1
    fi
    local size
    size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
    echo "     $file: $size bytes"
}

# --- Whisper medium (INT8)
echo "==> Whisper medium (INT8)"
WHISPER_REPO="soniqo/Whisper-Medium-ONNX"
for f in medium-encoder.int8.onnx medium-decoder.int8.onnx medium-tokens.txt; do
    download_hf_file "$WHISPER_REPO" "$f" "$WHISPER_DIR/$f"
done

# --- WeSpeaker ResNet34-LM
echo "==> WeSpeaker ResNet34-LM"
WESPEAKER_REPO="Wespeaker/wespeaker-voxceleb-resnet34-LM"
WESPEAKER_FILE="voxceleb_resnet34_LM.onnx"
download_hf_file "$WESPEAKER_REPO" "$WESPEAKER_FILE" "$WESPEAKER_DIR/$WESPEAKER_FILE"

# --- Валидация
echo
echo "==> Проверка"
EXPECTED=(
    "$WHISPER_DIR/medium-encoder.int8.onnx"
    "$WHISPER_DIR/medium-decoder.int8.onnx"
    "$WHISPER_DIR/medium-tokens.txt"
    "$WESPEAKER_DIR/$WESPEAKER_FILE"
)
MISSING=0
for f in "${EXPECTED[@]}"; do
    if [ ! -s "$f" ]; then
        echo "  [MISS] $f"
        MISSING=$((MISSING + 1))
    else
        printf "  [OK]   %s (%s bytes)\n" "$f" "$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")"
    fi
done

if [ "$MISSING" -ne 0 ]; then
    echo
    echo "Не хватает $MISSING файлов."
    echo "Установите современный CLI: pip install -U 'huggingface_hub[cli]'"
    echo "или убедитесь, что установлен wget/curl."
    exit 1
fi

echo
echo "Готово. Все модели на месте."
