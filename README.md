# ds-ocr — DeepSeek-OCR Pure C Inference Engine

[中文文档](README_zh.md)

A pure C inference implementation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- **Zero external dependencies** — only BLAS (Accelerate/OpenBLAS) + stb_image.h
- **Zero-copy weight loading** — mmap BF16 safetensors, on-the-fly F32 conversion
- **Platform-optimized kernels** — NEON / AVX2 / AVX-512 / scalar auto-dispatch

## Quick Start

```bash
# 1. Download model weights (requires huggingface_hub)
./download_model.sh ./deepseek-ocr

# 2. Build (macOS defaults to Accelerate BLAS)
make blas

# 3. Run OCR
./ds_ocr -d ./deepseek-ocr -i document.png --rp 1.03
```

> Linux: `make blas` auto-detects OpenBLAS. See [Building](#building) for cross-compilation.

## Performance

Apple M2 Pro (8 threads, BLAS), 6-crop V2 image:

| Stage | v0.5 | v0.8 | v0.9 | Optimization |
|-------|------|------|------|-------------|
| SAM+Encoder | ~50s | **8.8s** | **8.8s** | Parallel global crop |
| Prefill (862 tokens) | ~30s | **1.1s** | **1.1s** | Batched MoE sgemm |
| Decode (280 tokens) | ~19s | **6.2s** | **~2s** | Argmax LM head + contiguous experts |
| **Total** | **~97s** | **16s** | **~12s** | |

Key optimizations in v0.9:
- **Argmax LM head**: Uses `ds_argmax_matvec_bf16` instead of full sgemm for
  greedy decoding — avoids 631MB BF16→F32 weight conversion, computes dot
  products on-the-fly while tracking only the best token (~8ms vs ~60ms/step).
- **Selective repetition penalty**: Only recomputes logits for history tokens
  (~100) via `ds_bf16_dot_row`, not all 129280 vocabulary entries.
- **Contiguous expert blocks (P2)**: All 64 experts' gate_up_fused + shared
  weights in a single contiguous allocation per layer — adjacent experts share
  pages, reducing page faults 2.4× during MoE decode (~351ms → ~121ms/step).
- **madvise prefetch**: Issues `MADV_WILLNEED` for upcoming expert weights,
  further reducing page fault stalls from random expert address jumps.
- **Fused expert forward**: `ds_expert_forward_fused()` combines gate+up
  projection into a single matvec, improving L2 cache reuse of the input vector.
- **8-thread decode**: All cores used for decode (argmax LM head benefits from
  8T; small expert matvecs are marginally slower but overall faster).

Small images (1-crop V2, global-only): ~40s total (~3-4 tok/s decode).

**v0.9 = 8× v0.5 = 61× Python PyTorch (CPU BF16 ~736s)**

```
$ ./ds_ocr -d model_dir -i image.png --profile
Inference: 12135 ms, 280 text tokens (14.3 tok/s decode)
  Encoding: 8783 ms | Prefill: 1092 ms | Decode: 2260 ms
```

## Architecture

```
DeepSeek-OCR V1                           DeepSeek-OCR V2
─────────────────                          ─────────────────
Image (1024×1024)                          Any Image
    │                                          │
    ▼                                          ▼
SAM ViT-B (12 blocks)                     Dynamic Preprocess
    │                                      ├─ N crops × 768×768
    │                                      └─ 1 global 1024×1024
    ▼                                          │
┌───┴───┐                              ┌───────┴───────┐
│ CLIP  │  ← SAM patch_embeds           │ SAM × (N+1)   │
│ViT-L/14│                              │ local→896×12² │
│24 blocks│                              │ global→896×16²│
└───┬───┘                               └───────┬───────┘
    │                                           │
    ▼                                           ▼
Concat(SAM,CLIP) → 2048-dim          DeepEncoder V2 (Qwen2-0.5B)
    │                                  24 layers + causal flow queries
    ▼                                  1121→857 tokens (masked_scatter)
Projector (2048→1280)                     │
    │                                     ▼
    │                               Projector (896→1280)
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
           MoE Decoder (DeepSeek3B-MoE)
           12 layers: L0 dense + L1-11 MoE
           64 routed experts (top-6) + 2 shared
                   │
                   ▼
               Text Output
```

### Model Parameters

| Component | V1 | V2 | Params |
|-----------|----|----|--------|
| SAM Vision Tokenizer | ViT-B | ViT-B | ~86M |
| Encoder | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | ~300M / ~500M |
| Projector | 2048→1280 | 896→1280 (linear) | ~2.6M / ~1.1M |
| MoE Decoder | DeepSeek3B-MoE | DeepSeek3B-MoE | ~3B (570M active) |

### Key Architecture Details

<details>
<summary>SAM Vision Tokenizer</summary>

- 12 transformer blocks, window attention (window_size=14), global attention at layers [2,5,8,11]
- Fused QKV projection, relative position embeddings (rel_pos_h, rel_pos_w)
- Neck: V1 uses 2×(Conv1×1+LN), V2 uses Conv1×1+LN+Conv3×3+LN
- Downsample: net_2(256→512, k3, s2) + net_3(512→1024/896, k3, s2)

</details>

<details>
<summary>DeepEncoder V2</summary>

- Qwen2-0.5B architecture, 24 layers, hidden=896, 14 MHA heads, 2 KV heads (GQA)
- Causal flow queries: 144 queries per crop, masked_scatter to image_size=640's 857 positions
- Learned absolute position embeddings (not RoPE)
- Weight prefix: `model.qwen2_model.model.model.layers.*`

</details>

<details>
<summary>MoE Decoder</summary>

- 12 layers, hidden=1280, 10 heads, head_dim=128
- Layer 0: Dense FFN (SwiGLU, intermediate=6848)
- Layer 1-11: 64 routed experts (top-6) + 2 shared experts, expert intermediate=896
- Standard MHA + LLaMA-style RoPE (not MLA, kv_heads=q_heads=10)
- BOS=0, EOS=1

</details>

<details>
<summary>V2 Multi-Crop Preprocessing</summary>

1. `find_closest_aspect_ratio()` selects optimal crop ratio (min aspect diff, area tie-breaker)
2. Large image (e.g. 1938×1210) → ratio (3,2) → 6 local crops of 768×768
3. Small image (both dims ≤768) → ratio (1,1) → 1 local crop of 768×768
4. Global view: `ImageOps.pad()` → 1024×1024 (always present)
5. SAM: N local + 1 global → [896,12,12] / [896,16,16]
6. Token layout (image_size=640): num_queries=10 → 857 image slots
7. masked_scatter: 1121→857 (truncate overflow)
8. Prefix: BOS(1) + 857 image + 4 text ("\nFree OCR.") = 862 tokens

</details>

## Building

```bash
make blas           # BLAS-accelerated (recommended)
make debug          # AddressSanitizer debug build
make clean          # Clean artifacts
make info           # Show build config
```

### Cross-Compilation

```bash
# ARM64 (Apple Silicon)
make blas CC=clang CFLAGS="-Wall -O3 -arch arm64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"

# x86_64 (Intel)
make blas CC=clang CFLAGS="-Wall -O3 -arch x86_64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"
```

## Usage

### CLI

```bash
# Basic OCR
./ds_ocr -d ./deepseek-ocr -i document.png

# Recommended: repetition penalty 1.01-1.03
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03

# Silent mode: OCR text only (pipe/skill integration)
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --silent

# Profiling
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --profile

# macOS Vision OCR (no model weights needed)
./ds_ocr -i doc.png --vision --silent
./ds_ocr -i doc.png --vision-fast --silent
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory | Required |
| `-i <file>` | Input image | Required |
| `-t <n>` | Thread count | All CPUs |
| `-n <n>` | Max generated tokens | 4096 |
| `--temp <f>` | Sampling temperature | 0 (greedy) |
| `--rp <f>` | Repetition penalty (1.0=off, rec 1.01-1.1) | 1.0 |
| `--ngram <n>` | No-repeat n-gram (0=off, rec 20-35) | 0 |
| `--min-tokens <n>` | Min generated tokens (prevent early EOS) | 256 |
| `--vision` | macOS Vision OCR backend | Off |
| `--vision-fast` | macOS Vision OCR backend (fast) | Off |
| `--profile` | Per-stage timing | Off |
| `--debug` | Verbose debug output | Off |
| `--silent` | OCR text output only | Off |

### C API

```c
#include "ds_ocr.h"

ds_ctx_t *ctx = ds_load("./deepseek-ocr");
if (!ctx) { /* handle error */ }

ctx->max_new_tokens = 2048;
ctx->temperature    = 0.0f;  /* greedy */

/* Stream callback (optional) */
ds_set_token_callback(ctx, my_callback, userdata);

/* Recognize from file */
char *text = ds_recognize(ctx, "document.png");
printf("Result: %s\n", text);
free(text);

/* Recognize from raw pixels */
char *text2 = ds_recognize_image(ctx, rgb_pixels, width, height, 3);
free(text2);

ds_free(ctx);
```

Stream callback:

```c
void my_callback(const char *piece, void *userdata) {
    fputs(piece, stdout);  /* UTF-8 token fragment */
    fflush(stdout);
}
```

## Project Structure

```
ds-ocr/
├── ds_ocr.h/c                 # Public API + model loading + recognition pipeline
├── ds_visual_tokenizer.h/c    # SAM ViT-B (window attn, rel pos, neck, downsample)
├── ds_deep_encoder.h/c        # CLIP ViT-L/14 (V1) + DeepEncoder V2 (Qwen2-0.5B)
├── ds_moe_decoder.h/c         # MoE decoder (dense L0 + MoE L1-11)
├── ds_kernels.h/c              # Math kernel API + thread pool + dispatch
├── ds_kernels_impl.h           # Architecture dispatch macros (NEON/AVX/generic)
├── ds_kernels_generic.c        # Scalar fallback
├── ds_kernels_neon.c           # ARM NEON optimizations (incl. BF16 dot product)
├── ds_kernels_avx.c            # x86 AVX2/AVX-512 optimizations
├── ds_safetensors.h/c          # Multi-shard safetensors reader (BF16 + FP32)
├── ds_image.h/c                # Image loading + preprocessing (stb_image + bicubic resize)
├── ds_tokenizer.h/c            # Qwen2 BPE tokenizer + added_tokens
├── ds_platform_ocr.h/c/m       # macOS Vision OCR bridge (.m=ObjC, .c=Linux stub)
├── ds_dump.h                    # Debug tensor dump utilities
├── main.c                       # CLI entry point
├── test.c                       # Test suite
├── stb_image.h                  # Single-header image loader (public domain)
├── Makefile                     # Build system
├── download_model.sh            # Model download script
├── README.md                    # This file (English)
└── README_zh.md                 # Chinese documentation
```

**Code size**: ~12K lines of custom code (excluding stb_image.h), compiled binary ~331KB.

## Testing

```bash
make test              # Build and run all tests (BLAS backend)
make test_debug        # AddressSanitizer mode
./test_ds_ocr test_kernels   # Run specific test suite
```

Test suites:
- **test_kernels** — Math kernel correctness (LayerNorm, RMSNorm, matmul, SwiGLU, attention)
- **test_safetensors** — Safetensors reader (index parsing, BF16/FP32 conversion)
- **test_tokenizer** — BPE encode/decode round-trip
- **test_config** — Configuration init and constant validation
- **test_integration** — End-to-end pipeline (requires model weights)

## Debug Environment Variables

For development debugging. Set to dump intermediate tensors or skip pipeline stages (see `ds_dump.h`):

| Variable | Description |
|----------|-------------|
| `DS_DUMP_TENSORS` | Enable tensor dumping |
| `DS_DUMP_PATCH_EMBED` | Dump SAM patch embedding output |
| `DS_DUMP_SAM_LAYERS` | Per-layer SAM attention/output + neck + downsample |
| `DS_DUMP_ENCODER` | Dump encoder output |
| `DS_DUMP_INPUT_EMBEDS` | Dump projected input embeddings |
| `DS_DUMP_DECODER` | Dump decoder layer 0 internals |
| `DS_DUMP_DECODER_LAYERS` | Dump all decoder layers |
| `DS_DUMP_DECODE_STEPS` | Dump step-by-step decode process |
| `DS_DUMP_LAYERS` | Dump MoE expert routing details |
| `DS_DUMP_CONV2_IM2COL` | Dump ds_conv2d im2col buffer |
| `DS_DUMP_DIR` | Custom dump output directory |
| `DS_SKIP_ENCODER` | Skip encoder, load embeddings from file |
| `DS_PERFECT_ENCODER` | Override encoder output with Python reference .npy |
| `DS_LOAD_ENCODER_OUTPUT` | Load Python encoder output (skip SAM+encoder) |
| `DS_LOAD_ENC_INPUT` | Load Python encoder input (skip SAM, keep encoder) |
| `DS_LOAD_SAM_TOKENS` | Load Python SAM tokens (skip SAM, keep encoder) |
| `DS_LOAD_SAM_ALL` | Load full Python SAM tokens (global + all crops) |
| `DS_LOAD_PIL_PIXELS` | Load PIL preprocessed pixels (bypass C resize) |
| `DS_LOAD_INPUT_EMBEDS` | Load Python inputs_embeds (decoder debugging) |
| `DS_BF16_CACHE_MB` | BF16 weight cache size (MB) |
| `DS_BF16_SIMULATE_PYTHON` | Truncate intermediate values to BF16 precision (match Python) |
| `DS_SLOW_PREFILL` | Use per-token prefill (debug, not batched sgemm) |

## Weight Loading

Reads HuggingFace safetensors format directly, mmap zero-copy loading:

| Component | Tensor naming pattern | Format |
|-----------|----------------------|--------|
| SAM patch embed | `model.sam_model.patch_embed.proj.*` | FP32 |
| SAM blocks | `model.sam_model.blocks.{l}.*` | FP32 |
| SAM neck/downsample | `model.sam_model.neck.*`, `net_2.*`, `net_3.*` | FP32 |
| CLIP (V1) | `model.vision_model.*` | FP32 |
| DeepEncoder V2 | `model.qwen2_model.model.model.layers.{l}.*` | BF16 |
| DeepEncoder V2 norm/queries | `model.qwen2_model.model.model.norm.weight`, `query_*.weight` | BF16 |
| Projector V1 | `model.projector.layers.*` | FP32 |
| Projector V2 | `model.projector.weight` | BF16 |
| Decoder embed | `model.embed_tokens.weight` | BF16 |
| Decoder layers | `model.layers.{l}.*` | BF16 |
| LM head | `lm_head.weight` | BF16 |

## Platform Optimizations

| Platform | Kernel set | Key operations |
|----------|-----------|----------------|
| **ARM (Apple Silicon)** | NEON + BF16 dot product | BF16→F32, RMSNorm, matmul, SwiGLU |
| **x86 (Intel/AMD)** | AVX2+FMA / AVX-512 | BF16→F32, RMSNorm, matmul, SwiGLU |
| **Other** | Generic (scalar) | Portable C fallback |

## Status

| Component | V1 | V2 |
|-----------|----|----|
| SAM Vision Tokenizer | ✅ | ✅ |
| Encoder | ✅ | ✅ (corr ~0.995 vs Python) |
| Projector | ✅ | ✅ |
| MoE Decoder | ✅ | ✅ **Verified** (gate softmax fix: corr 0.82→0.9998) |
| Tokenizer | ✅ | ✅ |
| Multi-crop | N/A | ✅ |
| End-to-end OCR | ✅ | ✅ **99.29% text agreement** |

### Version History

- **v0.8** — Batched MoE prefill + parallel encoding: 16s end-to-end (6× v0.5)
- **v0.7** — F32 KV cache + fused residual+norm + direct SwiGLU + batched decode
- **v0.6** — sgemm LM head + BF16 KV cache + fused decode attention + fast exp
- **v0.5** — MoE gate softmax fix, decoder correctness verification, 7.5× vs Python

### Known Issues

1. **SAM encoder precision drift**: C's SAM+Encoder output has minor differences from Python (corr ~0.995), caused by FP32 accumulation error amplified through 12+24 layers. Does not affect OCR quality.
2. **Independent lm_head weights**: `lm_head.weight` ≠ `embed_tokens.weight`; C correctly loads the independent weights.

## Differences from Python Implementation

| Feature | Python (PyTorch) | This Implementation (C) |
|---------|-------------------|--------------------------|
| Weight format | Full FP32/BF16 tensors | mmap BF16 (zero-copy) |
| Attention | FlashAttention / SDPA | Online softmax (O(1) memory) |
| MoE routing | GPU scatter/gather | Batched sgemm: grouped expert + shared in one pass |
| Position encoding | Dynamic computation | Pre-computed RoPE table |
| Image resize | PIL BICUBIC (antialias) | Antialias bicubic |
| Tokenizer | HuggingFace tokenizers | Custom BPE (GPT-2 byte-level) + added_tokens |
| Dependencies | PyTorch, transformers... | Only BLAS + stb_image |

## Acknowledgments

- Architecture inspiration: [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — pure C ASR inference
- Model: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Image loading: [stb_image](https://github.com/nothings/stb) (public domain)

## License

MIT