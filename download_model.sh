#!/bin/bash
# download_model.sh - Download OCR model weights from HuggingFace
#
# Usage:
#   ./download_model.sh <model_name> [output_dir]
#   ./download_model.sh all [output_dir]        # Download all four models
#
# Supported models:
#   v1       deepseek-ai/DeepSeek-OCR        (CLIP ViT-L/14 encoder)
#   v2       deepseek-ai/DeepSeek-OCR-2      (DeepEncoder V2, recommended)
#   v3       baidu/Unlimited-OCR             (CLIP + R-SWA decoder)
#   v4       baidu/Unlimited-OCR-2           (DeepLIP-B/L + R-SWA + DASH sparse attention)
#   all      All four models
#
# Network:
#   Direct download first; on failure auto-retries via the s5 proxy wrapper
#   (sets ALL_PROXY=socks5://127.0.0.1:1080). s5 is looked up at:
#     1. $S5_PATH (explicit override)
#     2. $HOME/work/bin/s5
#     3. `command -v s5`
#
# Examples:
#   ./download_model.sh v2 ./models/DeepSeek-OCR-2
#   ./download_model.sh v4 ./models/Unlimited-OCR-2
#   ./download_model.sh all ./models

set -e

# ── Model definitions (bash 3.2 compatible — no associative arrays) ──
# Format: MODEL_<KEY>_HF_ID, MODEL_<KEY>_DESC, MODEL_<KEY>_DIR
MODEL_V1_HF_ID="deepseek-ai/DeepSeek-OCR"
MODEL_V2_HF_ID="deepseek-ai/DeepSeek-OCR-2"
MODEL_V3_HF_ID="baidu/Unlimited-OCR"
MODEL_V4_HF_ID="baidu/Unlimited-OCR-2"

MODEL_V1_DESC="DeepSeek-OCR V1 (CLIP ViT-L/14 encoder, 1024x1024)"
MODEL_V2_DESC="DeepSeek-OCR V2 (DeepEncoder V2, multi-crop, recommended)"
MODEL_V3_DESC="Unlimited-OCR V3 (CLIP + R-SWA decoder, sliding window)"
MODEL_V4_DESC="Unlimited-OCR-2 V4 (DeepLIP-B/L vision + R-SWA + DASH sparse attention)"

MODEL_V1_DIR="DeepSeek-OCR"
MODEL_V2_DIR="DeepSeek-OCR-2"
MODEL_V3_DIR="Unlimited-OCR"
MODEL_V4_DIR="Unlimited-OCR-2"

# s5 proxy wrapper (no shebang — invoke as: sh "$S5_BIN" <cmd...>)
S5_BIN=""
if [ -n "$S5_PATH" ] && [ -f "$S5_PATH" ]; then
    S5_BIN="$S5_PATH"
elif [ -f "$HOME/work/bin/s5" ]; then
    S5_BIN="$HOME/work/bin/s5"
elif p=$(command -v s5 2>/dev/null) && [ -f "$p" ]; then
    S5_BIN="$p"
fi

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

    # Run hf download, optionally through the s5 SOCKS5 proxy wrapper.
    # Usage: run_hf_download <hf_id> <out_dir> <direct|s5>
    run_hf_download() {
        local hf_id="$1" dir="$2" mode="$3"
        local tool=""
        if command -v hf &> /dev/null; then
            tool="hf"
        elif command -v huggingface-cli &> /dev/null; then
            tool="huggingface-cli"
        fi
        if [ -z "$tool" ] && command -v pip &> /dev/null; then
            log_info "Installing huggingface_hub..."
            pip install -q huggingface_hub
            tool="hf"
        fi
        if [ -z "$tool" ]; then
            log_error "Neither hf, huggingface-cli, nor pip found."
            echo "  Please install: pip install huggingface_hub"
            echo "  Then run: hf download $hf_id --local-dir $dir"
            return 1
        fi
        if [ "$mode" = "s5" ]; then
            # s5 has no shebang — run it under sh so it exports ALL_PROXY
            # and execs the download command through the SOCKS5 proxy.
            log_info "Using $tool via s5 proxy (ALL_PROXY=socks5://127.0.0.1:1080)..."
            sh "$S5_BIN" "$tool" download "$hf_id" --local-dir "$dir" \
                --exclude "assets/*" "*.pdf" "wheel/*" "*.py" "*.whl"
        else
            log_info "Using $tool CLI..."
            "$tool" download "$hf_id" --local-dir "$dir" \
                --exclude "assets/*" "*.pdf" "wheel/*" "*.py" "*.whl"
        fi
    }

    # Try direct download first; fall back to s5 proxy on failure
    if run_hf_download "$hf_id" "$out_dir" direct; then
        :
    elif [ -n "$S5_BIN" ]; then
        log_warn "Direct download failed — retrying via s5 proxy ($S5_BIN)..."
        run_hf_download "$hf_id" "$out_dir" s5 || return 1
    else
        log_error "Download failed and no s5 proxy wrapper found."
        echo "  Set S5_PATH to your s5 wrapper and retry."
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
    echo "  v4    Unlimited-OCR-2       (baidu/Unlimited-OCR-2)"
    echo "  all   Download all four models"
    echo ""
    echo "Examples:"
    echo "  $0 v2                        # Download V2 to ./models/DeepSeek-OCR-2"
    echo "  $0 v4 ./my-models/v4         # Download V4 to custom directory"
    echo "  $0 all ./models              # Download all to ./models/{DeepSeek-OCR,...}"
    echo ""
    echo "Network: auto-falls back to s5 SOCKS5 proxy on failure"
    echo "         (S5_PATH=/path/to/s5 to override proxy wrapper location)"
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
    v1|v2|v3|v4|all) ;;
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
    for key in v1 v2 v3 v4; do
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
    for key in v1 v2 v3 v4; do
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
