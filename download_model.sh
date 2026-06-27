#!/bin/bash
# download_model.sh - Download OCR model weights from HuggingFace
#
# Usage:
#   ./download_model.sh <model_name> [output_dir]
#   ./download_model.sh all [output_dir]        # Download all three models
#
# Supported models:
#   v1       deepseek-ai/DeepSeek-OCR        (CLIP ViT-L/14 encoder)
#   v2       deepseek-ai/DeepSeek-OCR-2      (DeepEncoder V2, recommended)
#   v3       baidu/Unlimited-OCR             (CLIP + R-SWA decoder)
#   all      All three models
#
# Examples:
#   ./download_model.sh v2 ./models/DeepSeek-OCR-2
#   ./download_model.sh v3 ./models/Unlimited-OCR
#   ./download_model.sh all ./models

set -e

# ── Model definitions (bash 3.2 compatible — no associative arrays) ──
# Format: MODEL_<KEY>_HF_ID, MODEL_<KEY>_DESC, MODEL_<KEY>_DIR
MODEL_V1_HF_ID="deepseek-ai/DeepSeek-OCR"
MODEL_V2_HF_ID="deepseek-ai/DeepSeek-OCR-2"
MODEL_V3_HF_ID="baidu/Unlimited-OCR"

MODEL_V1_DESC="DeepSeek-OCR V1 (CLIP ViT-L/14 encoder, 1024x1024)"
MODEL_V2_DESC="DeepSeek-OCR V2 (DeepEncoder V2, multi-crop, recommended)"
MODEL_V3_DESC="Unlimited-OCR V3 (CLIP + R-SWA decoder, sliding window)"

MODEL_V1_DIR="DeepSeek-OCR"
MODEL_V2_DIR="DeepSeek-OCR-2"
MODEL_V3_DIR="Unlimited-OCR"

# ── Helpers ─────────────────────────────────────────────────────────
log_info()  { echo "[INFO] $1"; }
log_warn()  { echo "[WARN] $1"; }
log_error() { echo "[ERROR] $1"; }
log_title() { echo ""; echo "====== $1 ======"; }

# Get model property by key and field
# Usage: get_model_prop <key> <field>
# e.g. get_model_prop v2 HF_ID → deepseek-ai/DeepSeek-OCR-2
get_model_prop() {
    eval echo "\"\$MODEL_${1}_${2}\""
}

# Download a single model
# Usage: download_model <version_key> <output_dir>
download_model() {
    local key=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    local out_dir="$2"
    local hf_id=$(get_model_prop "$key" HF_ID)
    local desc=$(get_model_prop "$key" DESC)

    log_title "Downloading $1: $desc"
    echo "  HuggingFace ID : $hf_id"
    echo "  Output dir     : $out_dir"
    echo ""

    # Check if already downloaded
    if [ -f "$out_dir/config.json" ]; then
        log_warn "Model already exists at $out_dir/config.json"
        echo "  Use a different output dir or remove the existing one to re-download."
        log_info "Skipping $1 (already downloaded)"
        return 0
    fi

    # Create output directory
    mkdir -p "$out_dir"

    # Try download tools in order
    if command -v hf &> /dev/null; then
        log_info "Using hf CLI..."
        hf download "$hf_id" --local-dir "$out_dir"
    elif command -v huggingface-cli &> /dev/null; then
        log_info "Using huggingface-cli..."
        huggingface-cli download "$hf_id" --local-dir "$out_dir"
    elif command -v pip &> /dev/null; then
        log_info "Installing huggingface_hub..."
        pip install -q huggingface_hub
        log_info "Using hf..."
        hf download "$hf_id" --local-dir "$out_dir"
    else
        log_error "Neither hf, huggingface-cli, nor pip found."
        echo "  Please install: pip install huggingface_hub"
        echo "  Then run: hf download $hf_id --local-dir $out_dir"
        return 1
    fi

    log_info "Done: $1 downloaded to $out_dir"
}

# Print usage
usage() {
    echo "Usage: $0 <model_name> [output_dir]"
    echo ""
    echo "Models:"
    echo "  v1    DeepSeek-OCR V1       (deepseek-ai/DeepSeek-OCR)"
    echo "  v2    DeepSeek-OCR V2       (deepseek-ai/DeepSeek-OCR-2, recommended)"
    echo "  v3    Unlimited-OCR         (baidu/Unlimited-OCR)"
    echo "  all   Download all three models"
    echo ""
    echo "Examples:"
    echo "  $0 v2                        # Download V2 to ./models/DeepSeek-OCR-2"
    echo "  $0 v3 ./my-models/v3         # Download V3 to custom directory"
    echo "  $0 all ./models              # Download all to ./models/{DeepSeek-OCR,...}"
    echo ""
    echo "After download, run:"
    echo "  ./ds_ocr -d <model_dir> -i document.png --rp 1.03"
}

# ── Main ────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

MODEL_KEY="$1"
OUT_DIR="${2:-}"

# Validate model key
case "$MODEL_KEY" in
    v1|v2|v3|all) ;;
    *)
        log_error "Unknown model: '$MODEL_KEY'"
        echo ""
        usage
        exit 1
        ;;
esac

if [ "$MODEL_KEY" = "all" ]; then
    # Download all models
    BASE_DIR="${OUT_DIR:-./models}"
    FAILED=0
    for key in v1 v2 v3; do
        dir_name=$(get_model_prop "$key" DIR)
        dir="$BASE_DIR/$dir_name"
        if ! download_model "$key" "$dir"; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        log_error "$FAILED model(s) failed to download"
        exit 1
    fi

    log_title "All models downloaded"
    echo ""
    echo "Model directories:"
    for key in v1 v2 v3; do
        dir_name=$(get_model_prop "$key" DIR)
        dir="$BASE_DIR/$dir_name"
        echo "  $key: $dir"
    done
    echo ""
    echo "Run OCR with:"
    echo "  ./ds_ocr -d $BASE_DIR/DeepSeek-OCR-2 -i document.png --rp 1.03"
else
    # Download single model
    dir_name=$(get_model_prop "$MODEL_KEY" DIR)
    dir="${OUT_DIR:-./models/$dir_name}"
    download_model "$MODEL_KEY" "$dir"
    echo ""
    echo "Run OCR with:"
    echo "  ./ds_ocr -d $dir -i document.png --rp 1.03"
fi
