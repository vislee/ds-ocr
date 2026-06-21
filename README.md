# ds_ocr — DeepSeek-OCR Pure C Inference Engine

A pure C implementation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) inference, following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

**Zero external dependencies** — only BLAS (Accelerate/OpenBLAS) + stb_image.h.

## Architecture

```
                          DeepSeek-OCR V1
                    ┌─────────────────────────┐
  Image ──────────► │ SAM ViT-B Encoder       │
     1024×1024      │  ├─ Patch Embed (16×16) │
                    │  ├─ 12 ViT Blocks       │──► patch_embeds [768-dim]
                    │  │  (window+global attn) │         │
                    │  ├─ Neck (768→256→256)  │         ▼
                    │  └─ Downsample          │    CLIP ViT-L/14
                    │     (256→512→1024)      │    ├─ Conv2d(768→1024)
                    │         │               │    ├─ 24 Transformer Blocks
                    │         │ sam_features  │    └─ Remove CLS
                    │         ▼               │         │
                    │   ┌─────┴──────┐        │         │ clip_output
                    │   │  Concat    │◄───────┘         │
                    │   │(CLIP+SAM)  │                   │
                    │   │  → 2048-dim│                   │
                    │   └─────┬──────┘                   │
                    └─────────┼──────────────────────────┘
                              ▼
                      Projector (2048→1280)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MoE Decoder        │
                    │  DeepSeek3B-MoE     │
                    │  12 layers          │
                    │  Layer 0: Dense FFN │
                    │  Layer 1-11: MoE    │
                    │  64 experts, top-6  │
                    │  + 2 shared experts │
                    └─────────┬───────────┘
                              ▼
                          Text Output
```

```
                          DeepSeek-OCR V2
                    ┌───────────────────────────────┐
  Any Image ────►   │ Dynamic Preprocess             │
  (small/large)    │  ├─ N crops × 768×768          │
                    │  │   (1 crop if both dims≤768) │
                    │  └─ 1 global view 1024×1024   │
                    └───────┬───────────┬───────────┘
                            │           │
                ┌───────────▼──┐  ┌─────▼──────────┐
                │ SAM(768×768) │  │ SAM(1024×1024) │
                │ ×6 crops     │  │ ×1 global      │
                │ →[896,12,12] │  │ →[896,16,16]   │
                └───────┬──────┘  └───────┬────────┘
                        │                 │
                        ▼                 ▼
               ┌────────────────────────────────┐
               │ DeepEncoder V2 (Qwen2-0.5B)    │
               │  ├─ 24 Transformer Layers       │
               │  ├─ Causal Flow Queries         │
               │  └─ local:144 tok + global:256  │
               │     + view_separator(1)         │
               │     = 1121 encoder tokens       │
               │  → masked_scatter → 857 tokens  │
               │     (image_size=640 layout)     │
               └───────────────┬────────────────┘
                               │ 1280-dim
                               ▼
                    ┌─────────────────────┐
                    │  MoE Decoder        │
                    │  DeepSeek3B-MoE     │
                    │  12 layers          │
                    │  Standard MHA       │
                    │  (LLaMA-style RoPE) │
                    │  10 heads, 10 KV    │
                    └─────────┬───────────┘
                              ▼
                          Text Output
```

### Model Parameters

| Component | V1 | V2 | Parameters |
|-----------|----|----|-----------|
| **SAM Vision Tokenizer** | ViT-B | ViT-B | ~86M |
| **Encoder** | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | ~300M / ~500M |
| **Projector** | 2048→1280 | 896→1280 (linear) | ~2.6M / ~1.1M |
| **MoE Decoder** | DeepSeek3B-MoE | DeepSeek3B-MoE | ~3B (570M active) |

### V2 Multi-Crop Preprocessing

DeepSeek-OCR V2 uses a multi-crop strategy for **all images** (including small ones):

1. **Dynamic preprocess**: `find_closest_aspect_ratio()` selects optimal crop ratio (min aspect diff, area tie-breaker)
   - Large image, e.g., 1938×1210 → ratio (3,2) → 6 local crops of 768×768
   - Small image (both dims ≤ 768) → ratio (1,1) → 1 local crop of 768×768
2. **Global view**: `ImageOps.pad()` to 1024×1024 (centered with gray padding) — always present
3. **SAM processes**: N local (768×768 → 896×12×12) + 1 global (1024×1024 → 896×16×16)
   - Small image: 1 local + 1 global → 144 + 256 + 1 sep = 401 encoder tokens
   - Large image (6 crops): 6 local + 1 global → 864 + 256 + 1 sep = 1121 encoder tokens
4. **Token layout** (image_size=640): num_queries=10 → 857 image positions
5. **masked_scatter**: encoder tokens → 857 image slots (truncates excess)
6. **Prefix**: BOS(1) + 857 image + 4 text ("\nFree OCR.") = 862 tokens

### Key Architecture Details

#### SAM Vision Tokenizer (ViT-B)
- **12 transformer blocks** with window attention (window_size=14)
- **Global attention** at layers [2, 5, 8, 11] (full bidirectional)
- **Fused QKV** projection (not separate Q/K/V)
- **Window attention**: pad→partition→QKV→attn→unpartition (matching Python's order; padded positions get non-zero QKV from bias)
- **Relative position embeddings** (`rel_pos_h`, `rel_pos_w`)
- **Neck**:
  - V1: 2×(Conv2d 1×1 + LayerNorm2d): 768→256→256
  - V2: Conv2d(768→256, 1×1) + LN + Conv2d(256→256, 3×3, s1, p1) + LN
- **Downsample (V1)**: net_2 Conv2d(256→512, k3, s2, p1) + net_3 Conv2d(512→1024, k3, s2, p1)
- **Downsample (V2)**: net_2 Conv2d(256→512, k3, s2, p1) + net_3 Conv2d(512→896, k3, s2, p1)

#### DeepEncoder V2 (Qwen2-0.5B based)
- **24 transformer layers**, hidden=896, 14 MHA heads, head_dim=64
- **Causal flow queries**: 144 queries for V1 image_size, or per-crop count
- **Mixed attention**: visual tokens (bidirectional) + causal queries (causal mask)
- **Position embeddings**: learned absolute (not RoPE) in Qwen2 style
- **Encoder prefix**: `model.qwen2_model.model.model.layers.*`

#### MoE Decoder (DeepSeek3B-MoE-A570M)
- **12 transformer layers**, hidden=1280, 10 MHA heads, head_dim=128
- **Layer 0**: Dense FFN (SwiGLU, intermediate=6848)
- **Layers 1-11**: MoE — 64 routed experts (top-6) + 2 shared experts
- MoE expert intermediate size = 896 (NOT 1536)
- **Attention**: Standard MHA with LLaMA-style RoPE (use_mla=False, despite config suggesting MLA)
- **KV heads = Q heads = 10** (NOT GQA; kv_lora_rank=None in config)
- BOS=0, EOS=1 (not 1/2)

## Building

### Prerequisites

- GCC or Clang (C11+)
- BLAS: Apple Accelerate (macOS) or OpenBLAS (Linux)
- Make

### Build

```bash
# With BLAS acceleration (recommended)
make blas

# Debug build (with AddressSanitizer)
make debug

# Clean build artifacts
make clean

# Show build info
make info
```

### Cross-Compilation

```bash
# ARM64 (Apple Silicon)
make blas CC=clang CFLAGS="-Wall -O3 -arch arm64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"

# x86_64 (Intel)
make blas CC=clang CFLAGS="-Wall -O3 -arch x86_64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"
```

## Usage

### Download Model Weights

```bash
# Requires huggingface_hub (pip install huggingface_hub)
./download_model.sh ./deepseek-ocr
```

### CLI

```bash
# Basic OCR
./ds_ocr -d ./deepseek-ocr -i document.png

# With repetition penalty (recommended for OCR: 1.01-1.03)
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03

# Silent mode — only OCR text on stdout (for piping/skill integration)
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --silent

# Profile mode — show phase timing breakdown
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --profile

# Debug mode — verbose per-layer output
./ds_ocr -d ./deepseek-ocr -i doc.png --debug

# macOS Vision backend (no model weights required)
./ds_ocr -i doc.png --vision --silent
./ds_ocr -i doc.png --vision-fast --silent
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input image (required) | — |
| `-t <n>` | Number of threads | all CPUs |
| `-n <n>` | Max new tokens | 4096 |
| `--temp <f>` | Sampling temperature | 0 (greedy) |
| `--rp <f>` | Repetition penalty (1.0 = off, try 1.01-1.1) | 1.0 |
| `--ngram <n>` | No-repeat n-gram size (0 = off, try 20-35) | 0 |
| `--min-tokens <n>` | Min tokens before allowing EOS | 256 |
| `--vision` | Use macOS Vision OCR backend | off |
| `--vision-fast` | Use macOS Vision OCR backend (fast) | off |
| `--profile` | Show phase-level timing breakdown | off |
| `--debug` | Verbose debug output | off |
| `--silent` | Only output OCR text (for piping) | off |
| `-h` | Show help | — |

### C API

```c
#include "ds_ocr.h"

/* Load model */
ds_ctx_t *ctx = ds_load("./deepseek-ocr");
if (!ctx) { /* handle error */ }

/* Configure */
ctx->max_new_tokens = 2048;
ctx->temperature = 0.0f;  /* greedy decoding */

/* Set streaming callback (optional) */
ds_set_token_callback(ctx, my_callback, userdata);

/* Recognize image */
char *text = ds_recognize(ctx, "document.png");
printf("Result: %s\n", text);
free(text);

/* Or use raw pixels directly */
char *text2 = ds_recognize_image(ctx, rgb_pixels, width, height, 3);
free(text2);

/* Free resources */
ds_free(ctx);
```

#### Streaming Callback

```c
void my_callback(const char *piece, void *userdata) {
    /* piece is a UTF-8 token fragment, print immediately */
    fputs(piece, stdout);
    fflush(stdout);
}

ds_set_token_callback(ctx, my_callback, NULL);
```

## Project Structure

```
ds-ocr/
├── ds_ocr.h                 # Public API + config + all type definitions
├── ds_ocr.c                 # Model loading, orchestration, recognition
├── ds_visual_tokenizer.h/c  # SAM ViT-B (window attn, rel pos, neck, downsample)
├── ds_deep_encoder.h/c      # CLIP ViT-L/14 (V1) + DeepEncoder V2 (Qwen2-0.5B)
├── ds_moe_decoder.h/c       # MoE decoder (dense layer 0 + MoE layers 1-11)
├── ds_kernels.h              # Math kernel API declarations
├── ds_kernels.c              # Kernel dispatcher + threading
├── ds_kernels_impl.h         # Architecture dispatch macros (NEON/AVX/generic)
├── ds_kernels_generic.c      # Scalar fallback (no SIMD)
├── ds_kernels_neon.c         # ARM NEON optimized kernels
├── ds_kernels_avx.c          # x86 AVX2/AVX-512 optimized kernels
├── ds_safetensors.h/c        # Multi-shard safetensors reader (BF16 + FP32)
├── ds_image.h/c              # Image loading + preprocessing (via stb_image)
├── ds_tokenizer.h/c          # Qwen2 BPE tokenizer (GPT-2 byte-level) + added_tokens
├── ds_platform_ocr.h/c/m   # macOS Vision OCR bridge (.m=ObjC, .c=Linux stub)
├── ds_dump.h                 # Tensor dump utilities (debug, env: DS_DUMP_TENSORS)
├── main.c                    # CLI entry point
├── test.c                    # Test suite (run: make test)
├── stb_image.h               # Single-header image loader (public domain)
├── Makefile                  # Build system
├── download_model.sh         # Model download script
└── README.md
```

### Debug Environment Variables

The engine supports various `DS_*` environment variables for debugging and development:

| Variable | Description |
|----------|-------------|
| `DS_DUMP_TENSORS` | Enable tensor dumping at key pipeline stages |
| `DS_DUMP_FIRST_CROP` | Dump only the first crop's SAM output |
| `DS_DUMP_PATCH_EMBED` | Dump SAM patch embedding output |
| `DS_DUMP_SAM_LAYERS` | Dump per-layer SAM attention/output + neck + downsample (crop 0 only) |
| `DS_DUMP_CONV2_IM2COL` | Dump im2col buffers from ds_conv2d (for neck conv debug) |
| `DS_DUMP_DIR` | Custom directory for SAM/encoder dump output |
| `DS_SAM_POSEMBED_FILE` | Override SAM position embedding (skip interpolation) |
| `DS_DUMP_ENCODER` | Dump encoder output |
| `DS_DUMP_INPUT_EMBEDS` | Dump final input embeddings (after projector) |
| `DS_DUMP_DECODER` | Dump decoder layer 0 internals |
| `DS_DUMP_DECODER_LAYERS` | Dump all decoder layers |
| `DS_DUMP_LAYERS` | Dump MoE expert routing details |
| `DS_PERFECT_ENCODER` | Override encoder output with Python reference .npy |
| `DS_SKIP_ENCODER` | Skip encoder, load embeddings from file |
| `DS_LOAD_INPUT_EMBEDS` | Load Python reference inputs_embeds for decoder debugging |
| `DS_LOAD_ENCODER_OUTPUT` | Load Python reference encoder output (skip SAM+encoder) |
| `DS_LOAD_ENC_INPUT` | Load Python reference encoder input per crop (skip SAM, keep encoder) |
| `DS_LOAD_SAM_TOKENS` | Load Python reference SAM tokens (skip SAM, keep encoder) |
| `DS_LOAD_PIL_PIXELS` | Load PIL-preprocessed pixels from dump/ (bypass C resize) |
| `DS_BF16_CACHE_MB` | Set BF16 weight cache size (MB) |

## Weight Loading

The engine loads weights directly from HuggingFace safetensors format:

| Component | Tensor Name Pattern | Format |
|-----------|-------------------|--------|
| SAM patch embed | `model.sam_model.patch_embed.proj.*` | FP32 |
| SAM blocks | `model.sam_model.blocks.{l}.*` | FP32 |
| SAM neck | `model.sam_model.neck.*` | FP32 |
| SAM downsample | `model.sam_model.net_2.*`, `net_3.*` | FP32 |
| CLIP (V1) | `model.vision_model.*` | FP32 |
| DeepEncoder V2 | `model.qwen2_model.model.model.layers.{l}.*` | BF16 |
| DeepEncoder V2 norm | `model.qwen2_model.model.model.norm.weight` | BF16 |
| DeepEncoder V2 queries | `model.qwen2_model.query_{768,1024}.weight` | BF16 |
| Projector V1 | `model.projector.layers.*` | FP32 |
| Projector V2 | `model.projector.weight` | BF16 (linear 896→1280) |
| Decoder embed | `model.embed_tokens.weight` | BF16 |
| Decoder layers | `model.layers.{l}.*` | BF16 |
| LM head | `lm_head.weight` | BF16 |

Weights are **memory-mapped** (mmap) for instant loading — no full-weight copy into RAM.

## Platform Optimization

| Platform | Kernel Set | Key Operations |
|----------|-----------|----------------|
| **ARM (Apple Silicon)** | NEON | BF16→FP32 convert, RMSNorm, matmul, SwiGLU |
| **x86 (Intel/AMD)** | AVX2+FMA / AVX-512 | BF16→FP32, RMSNorm, matmul, SwiGLU |
| **Other** | Generic (scalar) | Portable C fallback |

## Testing

```bash
# Run all tests
make test

# Run with verbose output
make test V=1

# Run specific test
./test_ds_ocr test_kernels
```

See [test.c](test.c) for available test suites:
- **test_kernels** — Math kernel correctness (LayerNorm, RMSNorm, matmul, SwiGLU, attention)
- **test_safetensors** — Safetensors reader (index parsing, BF16/FP32 conversion)
- **test_tokenizer** — BPE tokenizer encode/decode roundtrip
- **test_config** — Config initialization and constant verification
- **test_integration** — End-to-end pipeline (requires model weights)

## Performance

Typical inference on Apple M2 Pro (8 threads, BLAS accelerated):

### End-to-End (ds-ocr学习.png, 6-crop V2 image)

| Stage | v0.5 | v0.7 | **v0.8** | Optimization |
|-------|------|------|----------|-------------|
| Model loading (mmap) | <1s | <1s | <1s | — |
| SAM+Encoder | ~50s | ~12s | **8.8s** | Parallel global crop |
| Prefill (862 tokens) | ~30s | ~12s | **1.1s** | Batched MoE sgemm |
| Decode (280 tokens) | ~19s | ~6.3s | **6.2s** | F32 KV cache + fused ops |
| **Total** | **~97s** | **~30s** | **16s** | |

> **v0.8 is 6× faster than v0.5 and 45× faster than Python PyTorch (CPU, BF16, ~736s).**

### Optimization Highlights

| Phase | Optimization | Speedup |
|-------|-------------|---------|
| **Prefill** | Batched gate scoring (sgemm all tokens × 64 experts) | 11× |
| **Prefill** | Batched shared experts (sgemm all tokens × gate/up/down) | |
| **Prefill** | Grouped routed experts (batch tokens per expert → sgemm) | |
| **Encoding** | Parallel global crop with local crops (7 threads) | 1.36× |
| **Decode** | F32 KV cache (aligned + prefetch, no BF16→F32 per step) | 1.5× |
| **Decode** | Fused residual + RMS norm (NEON/AVX2/AVX-512) | |
| **Decode** | Direct SwiGLU (no interleaving), fast expf | |

### Profiler Output

```
$ ./ds_ocr -d model_dir -i image.png --profile
...
Inference: 16061 ms, 280 text tokens (45.21 tok/s decode)
  Encoding: 8783 ms | Prefill: 1092 ms | Decode: 6185 ms
```

### Small Image (1-crop V2)

| Stage | Time |
|-------|------|
| SAM+Encoder (1 crop + global, parallel) | ~5s |
| Prefill (401 tokens) | ~0.3s |
| Decode | ~5-6s |
| **Total** | **~11s** |

## Current Status

| Component | V1 | V2 |
|-----------|----|----|
| **SAM Vision Tokenizer** | ✅ Working | ✅ Verified |
| **Encoder** | ✅ Working | ✅ Working (corr ~0.995 vs Python) |
| **Projector** | ✅ Working | ✅ Working (linear 896→1280) |
| **MoE Decoder** | ✅ Working | ✅ **Verified** (gate softmax fix: corr 0.82→0.9998) |
| **Tokenizer** | ✅ Working | ✅ Working |
| **Multi-crop** | N/A | ✅ Working |
| **End-to-end OCR** | ✅ | ✅ **Verified** (99.29% text similarity vs Python) |

### v0.8 — Batched MoE + Parallel Encoding

**16s end-to-end on Apple M2 Pro** (6× faster than v0.5, 45× faster than Python):

- **Batched MoE prefill** (11× speedup): All tokens processed via sgemm — gate scoring,
  shared experts, and grouped routed experts replace per-token BF16 matvec loops
- **Parallel global crop**: Global image encoding overlaps with local crops, saving ~4s
- **F32 KV cache** (v0.7): Aligned + prefetch, eliminated BF16↔F32 conversion per decode step
- **Fused kernels** (v0.7): Residual + RMS norm, direct SwiGLU, NEON RMS norm, batch decode

### v0.5 — Decoder Correctness Verified

The MoE decoder is now **fully verified** against Python. Using identical `inputs_embeds`, C and Python produce matching decode logits (差异 <0.1, BF16 precision range).

**Key fix**: MoE gate softmax order — Python softmaxes over all 64 experts first, then selects top-K. C was selecting top-K first then softmaxing only those K. This caused different expert routing and 0.82 correlation at Layer 11. After fix: 0.9998 correlation.

### Known Issues

1. **SAM encoder precision drift**: C's SAM+Encoder produces slightly different output from Python (corr ~0.995) due to FP32 accumulation across 12 SAM + 24 DeepEncoder layers. This causes minor OCR text differences but does not affect overall quality.
2. **lm_head not tied**: `lm_head.weight` differs from `embed_tokens.weight` — C correctly loads the separate weight.

## Model Support

| Model | Version | Encoder | Notes |
|-------|---------|---------|-------|
| DeepSeek-OCR | v1 | CLIP ViT-L/14 | CLIP takes SAM patch_embeds as input |
| DeepSeek-OCR-2 | v2 | DeepEncoder V2 (Qwen2-0.5B) | Multi-crop: 6×768 + 1×1024, causal flow queries, split-half RoPE |

## Differences from Python Implementation

| Feature | Python (PyTorch) | This Implementation (C) |
|---------|-------------------|------------------------|
| Weight format | Full FP32/BF16 tensors | Memory-mapped BF16 (zero-copy) |
| Attention | FlashAttention / SDPA | Online softmax (no O(seq²) memory), window attn: pad→QKV→attn (matches Python) |
| MoE routing | Scatter/gather on GPU | Batched sgemm: grouped expert batching, shared experts in one pass |
| Position embeddings | Dynamic computation | Precomputed RoPE tables (LLaMA rotate_half style) |
| SAM Neck conv2 | 1×1 conv (V1) / 3×3 conv (V2) | V2: 3×3 s1 p1 conv (controlled by `g_sam_is_v2`) |
| Image resize | PIL BICUBIC (antialias) | Antialias bicubic (filter expansion for downsampling) |
| Tokenizer | HuggingFace tokenizers | Custom BPE (GPT-2 byte-level) + added_tokens (HTML table tags) |
| Dependencies | PyTorch, transformers, etc. | Only BLAS + stb_image |

## Credits

- Architecture inspired by [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — pure C ASR inference
- Model by [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Image loading by [stb_image](https://github.com/nothings/stb) (public domain)

## License

MIT
