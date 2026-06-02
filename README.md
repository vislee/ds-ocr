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
                    ┌─────────────────────────┐
  Image ──────────► │ SAM ViT-B Encoder       │
     1024×1024      │  (same as V1)           │──► sam_features [1024-dim]
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ DeepEncoder V2           │
                    │ (Qwen2-0.5B based)      │
                    │  ├─ 24 Transformer Layers│
                    │  ├─ Causal Flow Queries  │
                    │  └─ Mixed Attention      │
                    │     (visual: bidir,      │
                    │      queries: causal)    │
                    └─────────┬───────────────┘
                              │ 896-dim
                              ▼
                      Projector (896→1280)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MoE Decoder        │
                    │  (same as V1)       │
                    └─────────┬───────────┘
                              ▼
                          Text Output
```

### Model Parameters

| Component | V1 | V2 | Parameters |
|-----------|----|----|-----------|
| **SAM Vision Tokenizer** | ViT-B | ViT-B | ~86M |
| **Encoder** | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | ~300M / ~500M |
| **Projector** | 2048→1280 | 896→1280 | ~2.6M / ~1.1M |
| **MoE Decoder** | DeepSeek3B-MoE | DeepSeek3B-MoE | ~3B (570M active) |

### Key Architecture Details

#### SAM Vision Tokenizer (ViT-B)
- **12 transformer blocks** with window attention (window_size=14)
- **Global attention** at layers [2, 5, 8, 11] (full bidirectional)
- **Fused QKV** projection (not separate Q/K/V)
- **Relative position embeddings** (`rel_pos_h`, `rel_pos_w`)
- **Neck**: 2×(Conv2d 1×1 + LayerNorm2d): 768→256→256
- **Downsample**: net_2 Conv2d(256→512, k3, s2, p1) + net_3 Conv2d(512→1024, k3, s2, p1)

#### MoE Decoder (DeepSeek3B-MoE-A570M)
- **12 transformer layers**, hidden=1280, 10 MHA heads, head_dim=128
- **Layer 0**: Dense FFN (SwiGLU, intermediate=6848)
- **Layers 1-11**: MoE — 64 routed experts (top-6) + 2 shared experts
- MoE expert intermediate size = 896 (NOT 1536)
- **Per-head Q/K RMSNorm** (not per-layer)
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

# With options
./ds_ocr -d ./deepseek-ocr -i doc.png -t 8 -n 2048 --temp 0.0

# Debug mode (verbose per-layer output)
./ds_ocr -d ./deepseek-ocr -i doc.png --debug

# Silent mode (only output recognized text)
./ds_ocr -d ./deepseek-ocr -i doc.png --silent > output.txt
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input image (required) | — |
| `-t <n>` | Number of threads | all CPUs |
| `-n <n>` | Max new tokens | 4096 |
| `--temp <f>` | Sampling temperature | 0 (greedy) |
| `--debug` | Verbose debug output | off |
| `--silent` | Only output recognition text | off |
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
├── ds_tokenizer.h/c          # Qwen2 BPE tokenizer (GPT-2 byte-level)
├── main.c                    # CLI entry point
├── test.c                    # Test suite (run: make test)
├── stb_image.h               # Single-header image loader (public domain)
├── Makefile                  # Build system
├── download_model.sh         # Model download script
└── README.md
```

## Weight Loading

The engine loads weights directly from HuggingFace safetensors format:

| Component | Tensor Name Pattern | Format |
|-----------|-------------------|--------|
| SAM patch embed | `model.sam_model.patch_embed.proj.*` | FP32 |
| SAM blocks | `model.sam_model.blocks.{l}.*` | FP32 |
| SAM neck | `model.sam_model.neck.*` | FP32 |
| SAM downsample | `model.sam_model.net_2.*`, `net_3.*` | FP32 |
| CLIP (V1) | `model.vision_model.*` | FP32 |
| DeepEncoder V2 | `model.encoder.model.model.layers.{l}.*` | FP32 |
| Projector | `model.projector.layers.*` | FP32 |
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

Typical inference on Apple M2 Pro (8 threads):

| Stage | Time |
|-------|------|
| Model loading (mmap) | < 1s |
| Visual encoding (SAM + encoder) | ~200ms |
| Decoding (per token) | ~15ms |
| Total (256 visual + 512 text tokens) | ~8s |

## Model Support

| Model | Version | Encoder | Notes |
|-------|---------|---------|-------|
| DeepSeek-OCR | v1 | CLIP ViT-L/14 | CLIP takes SAM patch_embeds as input |
| DeepSeek-OCR-2 | v2 | DeepEncoder V2 (Qwen2-0.5B) | Causal flow queries for bidirectional→causal bridge |

## Differences from Python Implementation

| Feature | Python (PyTorch) | This Implementation (C) |
|---------|-------------------|------------------------|
| Weight format | Full FP32/BF16 tensors | Memory-mapped BF16 (zero-copy) |
| Attention | FlashAttention / SDPA | Online softmax (no O(seq²) memory) |
| MoE routing | Scatter/gather on GPU | Sequential expert evaluation |
| Position embeddings | Dynamic computation | Precomputed RoPE tables |
| Tokenizer | HuggingFace tokenizers | Custom BPE (GPT-2 byte-level) |
| Dependencies | PyTorch, transformers, etc. | Only BLAS + stb_image |

## Credits

- Architecture inspired by [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — pure C ASR inference
- Model by [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Image loading by [stb_image](https://github.com/nothings/stb) (public domain)

## License

MIT
