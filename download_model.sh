#!/bin/bash
# download_model.sh - Download DeepSeek-OCR model weights from HuggingFace
#
# Usage: ./download_model.sh [output_dir]
# Default output_dir: ./deepseek-ocr

set -e

MODEL_ID="deepseek-ai/DeepSeek-OCR"
OUT_DIR="${1:-./deepseek-ocr}"

echo "Downloading DeepSeek-OCR model to: $OUT_DIR"
echo "Model: $MODEL_ID"
echo ""

# Check for huggingface-cli or wget
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$MODEL_ID" --local-dir "$OUT_DIR"
elif command -v pip &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
    echo "Using huggingface-cli..."
    huggingface-cli download "$MODEL_ID" --local-dir "$OUT_DIR"
else
    echo "Neither huggingface-cli nor pip found."
    echo "Please install huggingface_hub: pip install huggingface_hub"
    echo "Then run: huggingface-cli download $MODEL_ID --local-dir $OUT_DIR"
    exit 1
fi

echo ""
echo "Download complete. Model files in: $OUT_DIR"
echo "Run inference with: ./ds_ocr -d $OUT_DIR -i document.png"
