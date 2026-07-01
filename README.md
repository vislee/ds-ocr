# ds-ocr — DeepSeek-OCR Pure C Inference Engine

[中文文档](README_zh.md)

A pure C inference implementation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) and [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- **3 model variants** — DeepSeek-OCR V1, V2, and Unlimited-OCR (V3)
- **Zero external dependencies** — only BLAS (Accelerate/OpenBLAS) + stb_image.h
- **Zero-copy weight loading** — mmap BF16 safetensors, on-the-fly F32 conversion
- **Platform-optimized kernels** — NEON / AVX2 / AVX-512 / scalar auto-dispatch

## Quick Start

```bash
# 1. Download model weights (requires huggingface_hub)
#    v1 = DeepSeek-OCR, v2 = DeepSeek-OCR-2 (recommended), v3 = Unlimited-OCR
./download_model.sh v2 ./models/DeepSeek-OCR-2

# 2. Build (macOS defaults to Accelerate BLAS)
make blas

# 3. Run OCR
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03
```

> Linux: `make blas` auto-detects OpenBLAS. See [Building](#building) for cross-compilation.

## Model Comparison

| | DeepSeek-OCR V1 | DeepSeek-OCR V2 | Unlimited-OCR V3 |
|---|---|---|---|
| **HuggingFace ID** | `deepseek-ai/DeepSeek-OCR` | `deepseek-ai/DeepSeek-OCR-2` | `baidu/Unlimited-OCR` |
| **Encoder** | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | CLIP ViT-L/14 + R-SWA |
| **Decoder** | DeepSeek3B-MoE | DeepSeek3B-MoE | DeepSeek3B-MoE + R-SWA |
| **Input** | 1024×1024 (stretch) | Dynamic multi-crop | 640×640 (pad) |
| **Visual tokens** | 256 | 857 (6-crop) | 273 |
| **Prompt** | `\nFree OCR.` | `\nFree OCR.` | `\ndocument parsing.` |
| **C support** | ✅ Full | ✅ Full | ✅ Full + EOS fix |
| **Model size** | ~6.3 GB | ~6.7 GB | ~6.2 GB |

### Benchmark

#### M2 Pro (10-core, 16 GB, BLAS + Metal GPU), 6-crop V2 image (595×841)

| Metric | V2 (BF16) | V2 (INT8 `--int4`) | V2 (CPU-only BLAS) |
|--------|-----------|---------------------|---------------------|
| **Total time** | 71.1s | 71.1s | 15.0s* |
| **Encoding** | 17.5s | 17.5s | 9.0s |
| **Prefill** | 18.0s | 18.0s | 0.93s |
| **Decode** | 35.5s (362 tok) | 35.5s (362 tok) | 5.1s (226 tok) |
| **Decode speed** | 10.2 tok/s | 10.2 tok/s | 44.4 tok/s |

> \* CPU-only BLAS from v0.9 measurements (pre-Metal). Metal GPU currently accelerates
> encoding but has a regression in decode prefill path on M2 Pro. INT8 quantization
> benefits memory-constrained devices and x86; on M2 Pro with Metal, decode speed is
> similar to BF16 due to GPU overhead.

#### M4 Max (16-core, 48 GB, CPU BLAS), 1794×1578 large image

| Metric | V1 | V2 (6-crop) | V3 (Unlimited-OCR) |
|--------|----|----|---------------------|
| **Total time** | 11.6s | ~18s* | 76.4s (30 crops) |
| **Encoding** | 6.0s | ~10s* | 42.6s (30 crops × SAM+CLIP) |
| **Prefill** | 0.7s (286 tokens) | ~1.0s* | 3.9s (3328 tokens) |
| **Decode** | 5.0s (233 tokens) | ~6.5s* | 30.0s (499 tokens) |
| **Decode speed** | 47.0 tok/s | ~44 tok/s* | 16.6 tok/s |
| **Output quality** | ⚠️ Minor typos | ✅ Correct | ✅ Correct (det tags stripped) |

> \* V2 M4 Max estimated from V1 scaling ratio (V2 encoding ~1.6× V1 due to multi-crop).

#### M2 Pro vs M4 Max Comparison (CPU BLAS, V2 6-crop)

| Stage | M2 Pro | M4 Max (est.) | Ratio |
|-------|--------|---------------|-------|
| SAM+Encoder | 9.0s | ~5.8s | 1.55× |
| Prefill (~660 tok) | 0.93s | ~0.6s | 1.55× |
| Decode (226 tok) | 5.1s | ~3.3s | 1.55× |
| **Total** | **15.0s** | **~9.7s** | **1.55×** |

> M4 Max scaling: CPU single-thread ~1.35×, multi-thread ~1.55×, memory BW ~1.37×.

#### Model Quality Comparison

| Model | Strengths | Weaknesses | Best for |
|-------|-----------|------------|----------|
| **V2** | Dynamic multi-crop, best accuracy, flexible input sizes | Slower encoding (6+ crops) | **General use (recommended)** |
| **V1** | Fastest single-crop, simplest pipeline | 1024×1024 stretch, BF16 precision typos | Quick scans, small images |
| **V3** | Detailed output, sliding window attention | Very slow for large images (30 crops), repetition issues | Small images ≤640px, structured docs |

> V3's decode speed is ~3× slower than V1/V2 due to LlamaAttention (10 heads × 128 dim)
> vs DeepSeek-V2 MLA compression. For small images (≤640px, 1 crop), V3 matches V1
> decode speed at ~47 tok/s.

## Performance

Apple M2 Pro (8 threads, BLAS), 6-crop V2 image:

| Stage | v0.5 | v0.8 | v0.9 | Optimization |
|-------|------|------|------|-------------|
| SAM+Encoder | ~50s | **8.8s** | **8.8s** | Parallel global crop |
| Prefill (~660 tokens) | ~30s | **1.1s** | **0.93s** | Batched MoE sgemm |
| Decode (226 tokens) | ~19s | **6.2s** | **5.1s** | Argmax LM head + contiguous experts |
| **Total** | **~97s** | **16s** | **15.0s** | |

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

**v0.9 = 8× v0.5 = 61× Python PyTorch (CPU BF16 ~736s)**

### INT8 Quantization (`--int4`)

Per-row asymmetric INT8 quantization for MoE expert weights, reducing memory
bandwidth ~2× while maintaining OCR accuracy (RMS < 0.01 vs BF16).

| | BF16 | INT8 (`--int4`) |
|---|---|---|
| **Expert weight size** | 3176 MB | 2399 MB (1.3× smaller) |
| **Quantization RMS** | — | 0.007–0.010 |
| **OCR accuracy** | ✅ Reference | ✅ Matches BF16 |
| **Quantization time** | — | ~3–8s (one-time at load) |

```bash
# INT8 quantization is applied at load time (~3-8s one-time cost)
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --int4
```

> **Note**: On Apple Silicon (M2+), hardware BF16 dot-product instructions are
> faster than software INT8 dequantize+MLA for single-token decode on CPU-only BLAS.
> However, with Metal GPU acceleration, INT8 can be competitive due to reduced
> memory bandwidth. INT8 is recommended for memory-constrained devices and x86
> platforms without BF16 hardware.

```
$ ./ds_ocr -d model_dir -i image.png --profile
# M2 Pro (CPU BLAS), V2 6-crop:
Inference: 15025 ms, 226 text tokens (44.4 tok/s decode)
  Encoding: 8989 ms | Prefill: 934 ms | Decode: 5102 ms

# M2 Pro (Metal GPU), V2 6-crop:
Inference: 71116 ms, 362 text tokens (10.2 tok/s decode)
  Encoding: 17537 ms | Prefill: 18028 ms | Decode: 35551 ms
```

## Architecture

```
  DeepSeek-OCR V1          DeepSeek-OCR V2          Unlimited-OCR (V3)
  ─────────────────         ─────────────────         ─────────────────
  Image (1024×1024)         Any Image                 Image (640×640 padded)
       │                         │                          │
       ▼                         ▼                          ▼
  SAM ViT-B (12 blocks)    Dynamic Preprocess         SAM ViT-B (12 blocks)
       │                    ├─ N crops × 768×768            │
       │                    └─ 1 global 1024×1024            ▼
       ▼                         │                     ┌────┴────┐
  ┌────┴────┐              ┌─────┴─────┐               │  CLIP   │ ← SAM features
  │  CLIP   │ ← SAM feat  │ SAM ×(N+1)│               │ ViT-L/14│   (bypass Conv2d)
  │ ViT-L/14│              │local→896×12²              │ 24 blocks│
  │ 24 blocks│              │global→896×16²              └────┬────┘
  └────┬────┘              └─────┬─────┘                     │
       │                         │                           ▼
       ▼                         ▼                    Concat(SAM,CLIP)
  Concat(SAM,CLIP)         DeepEncoder V2             → 2048-dim
   → 2048-dim              (Qwen2-0.5B)                    │
       │                   24 layers + causal flow          ▼
       ▼                   1121→857 tokens            Projector
  Projector                   (masked_scatter)         (2048→1280)
  (2048→1280)                     │                         │
       │                           ▼                         │
       │                     Projector (896→1280)             │
       │                           │                         │
       └──────────┬────────────────┴────────────┬────────────┘
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                      MoE Decoder (DeepSeek3B-MoE)
                      12 layers: L0 dense + L1-11 MoE
                      64 routed experts (top-6) + 2 shared
                      V3: R-SWA decode attention (sliding_window=128)
                                 │
                                 ▼
                             Text Output
```

### Model Parameters

| Component | V1 | V2 | V3 | Params |
|-----------|----|----|----|--------|
| SAM Vision Tokenizer | ViT-B | ViT-B | ViT-B | ~86M |
| Encoder | CLIP ViT-L/14 | DeepEncoder V2 | CLIP ViT-L/14 (bypass Conv2d) | ~300M / ~500M / ~300M |
| Projector | 2048→1280 | 896→1280 | 2048→1280 | ~2.6M / ~1.1M / ~2.6M |
| MoE Decoder | DeepSeek3B-MoE | DeepSeek3B-MoE | DeepSeek3B-MoE + R-SWA | ~3B |

### Key Architecture Details

<details>
<summary>SAM Vision Tokenizer</summary>

- 12 transformer blocks, window attention (window_size=14), global attention at layers [2,5,8,11]
- Fused QKV projection, relative position embeddings (rel_pos_h, rel_pos_w)
- Neck: 2×(Conv1×1+LN)
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
<summary>R-SWA Decoder (V3)</summary>

- Reference Sliding Window Attention during decode
- During prefill: full causal attention (unchanged)
- During decode: attend to [0..prefill_len-1] (visual/reference) + [kv_cache_len-128..kv_cache_len] (recent text)
- sliding_window_size=128, reduces KV cache growth for long outputs
- Token ID 128815 for all image positions (no separate start/end/newline tokens)

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
# DeepSeek-OCR V2 (recommended)
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03

# Unlimited-OCR V3
./ds_ocr -d ./models/Unlimited-OCR -i document.png

# DeepSeek-OCR V1
./ds_ocr -d ./models/DeepSeek-OCR -i document.png --rp 1.03

# Silent mode: OCR text only (pipe/skill integration)
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --silent

# Profiling
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --profile

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
| `--min-tokens <n>` | Min generated tokens (prevent early EOS) | 32 (V3: 32) |
| `--vision` | macOS Vision OCR backend | Off |
| `--vision-fast` | macOS Vision OCR backend (fast) | Off |
| `--profile` | Per-stage timing | Off |
| `--int4` | INT8 quantize MoE expert weights (2× less memory) | Off |
| `--debug` | Verbose debug output | Off |
| `--silent` | OCR text output only | Off |

### Recommended Settings per Model

| Model | `--rp` | `--ngram` | Notes |
|-------|--------|-----------|-------|
| V2 | 1.03 | 0 | Best quality, fast |
| V3 | 1.01 | 35 | ngram required to prevent repetition |
| V1 | 1.03 | 0 | Minor precision-induced typos |

### C API

```c
#include "ds_ocr.h"

ds_ctx_t *ctx = ds_load("./models/DeepSeek-OCR-2");
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
├── ds_deep_encoder.h/c        # CLIP ViT-L/14 (V1/V3) + DeepEncoder V2 (Qwen2-0.5B)
├── ds_moe_decoder.h/c         # MoE decoder (dense L0 + MoE L1-11 + R-SWA V3)
├── ds_kernels.h/c              # Math kernel API + thread pool + dispatch
├── ds_kernels_impl.h           # Architecture dispatch macros (NEON/AVX/generic)
├── ds_kernels_generic.c        # Scalar fallback
├── ds_kernels_neon.c           # ARM NEON optimizations (incl. BF16 dot product)
├── ds_kernels_avx.c            # x86 AVX2/AVX-512 optimizations
├── ds_safetensors.h/c          # Multi-shard safetensors reader (BF16 + FP32)
├── ds_image.h/c                # Image loading + preprocessing (stb_image + bicubic resize)
├── ds_tokenizer.h/c            # BPE tokenizer (GPT-2 byte-level + DeepSeek ▁format)
├── ds_platform_ocr.h/c/m       # macOS Vision OCR bridge (.m=ObjC, .c=Linux stub)
├── ds_dump.h                    # Debug tensor dump utilities
├── main.c                       # CLI entry point
├── test.c                       # Test suite
├── stb_image.h                  # Single-header image loader (public domain)
├── Makefile                     # Build system
├── download_model.sh            # Model download script (v1/v2/v3/all)
├── README.md                    # This file (English)
└── README_zh.md                 # Chinese documentation
```

**Code size**: ~13K lines of custom code (excluding stb_image.h), compiled binary ~331KB.

## Testing

```bash
make test              # Build and run all tests (BLAS backend)
make test_debug        # AddressSanitizer mode
./test_ds_ocr test_kernels   # Run specific test suite
```

## Downloading Models

```bash
# Download a specific model
./download_model.sh v2                        # V2 to ./models/DeepSeek-OCR-2
./download_model.sh v3 ./my-models/v3         # V3 to custom directory

# Download all three models
./download_model.sh all ./models
```

Requires `huggingface_hub` (`pip install huggingface_hub`). Supports `HF_ENDPOINT` for mirror sites.

## Weight Loading

Reads HuggingFace safetensors format directly, mmap zero-copy loading:

| Component | Tensor naming pattern | Format |
|-----------|----------------------|--------|
| SAM patch embed | `model.sam_model.patch_embed.proj.*` | FP32 |
| SAM blocks | `model.sam_model.blocks.{l}.*` | FP32 |
| SAM neck/downsample | `model.sam_model.neck.*`, `net_2.*`, `net_3.*` | FP32 |
| CLIP (V1/V3) | `model.vision_model.*` | BF16/FP32 |
| DeepEncoder V2 | `model.qwen2_model.model.model.layers.{l}.*` | BF16 |
| Projector V1/V3 | `model.projector.layers.*` | BF16/FP32 |
| Projector V2 | `model.projector.weight` | BF16 |
| Decoder embed | `model.embed_tokens.weight` | BF16 |
| Decoder layers | `model.layers.{l}.*` | BF16 |
| LM head | `lm_head.weight` | BF16 |

Tokenizer loaded from `vocab.json` (V3) or `tokenizer.json` (V1/V2) with automatic fallback.

## Platform Optimizations

| Platform | Kernel set | Key operations |
|----------|-----------|----------------|
| **ARM (Apple Silicon)** | NEON + BF16 dot product | BF16→F32, RMSNorm, matmul, SwiGLU |
| **x86 (Intel/AMD)** | AVX2+FMA / AVX-512 | BF16→F32, RMSNorm, matmul, SwiGLU |
| **Other** | Generic (scalar) | Portable C fallback |

## Status

| Component | V1 | V2 | V3 |
|-----------|----|----|----|
| SAM Vision Tokenizer | ✅ | ✅ | ✅ |
| Encoder (CLIP/DeepEnc) | ✅ | ✅ | ✅ |
| Projector | ✅ | ✅ | ✅ |
| MoE Decoder | ✅ | ✅ | ✅ |
| R-SWA Decode Attention | N/A | N/A | ✅ |
| Tokenizer | ✅ | ✅ | ✅ |
| Multi-crop | N/A | ✅ | ✅ (single crop) |
| End-to-end OCR | ⚠️ | ✅ | ✅ |

### Version History

- **v0.10** — INT8 per-row MoE expert quantization (`--int4`, RMS<0.01), platform auto-detect (M1/M2+/x86)
- **v0.9** — Unlimited-OCR V3 support (CLIP+R-SWA), tokenizer.json fallback, download_model.sh v1/v2/v3
- **v0.8** — Batched MoE prefill + parallel encoding: 16s end-to-end (6× v0.5)
- **v0.7** — F32 KV cache + fused residual+norm + direct SwiGLU + batched decode
- **v0.6** — sgemm LM head + BF16 KV cache + fused decode attention + fast exp
- **v0.5** — MoE gate softmax fix, decoder correctness verification, 7.5× vs Python

### Known Issues

1. **V1 CLIP encoder**: V1 and V3 share the same CLIP architecture (bypass Conv2d, receive SAM features directly). V1 output has minor BF16 precision-induced typos (e.g. "raletimit" vs "ratelimit") — within normal precision range.
2. **V3 output tags**: Unlimited-OCR produces `<|det|>` and `<|ref|>` detection tags that are automatically stripped in post-processing. The 830 added_tokens (including `<|det|>`, `<|ref|>`, `<|grounding|>`, `<td>`, `<tr>`, etc.) are now correctly loaded from `tokenizer.json` with vocab expansion beyond 128K.
3. **SAM encoder precision drift**: C's SAM+Encoder output has minor differences from Python (corr ~0.995), caused by FP32 accumulation error amplified through 12+24 layers. Does not affect OCR quality.
4. **Independent lm_head weights**: `lm_head.weight` ≠ `embed_tokens.weight`; C correctly loads the independent weights.

## Differences from Python Implementation

| Feature | Python (PyTorch) | This Implementation (C) |
|---------|-------------------|--------------------------|
| Weight format | Full FP32/BF16 tensors | mmap BF16 (zero-copy) |
| Attention | FlashAttention / SDPA | Online softmax (O(1) memory) |
| MoE routing | GPU scatter/gather | Batched sgemm: grouped expert + shared in one pass |
| Position encoding | Dynamic computation | Pre-computed RoPE table |
| Image resize | PIL BICUBIC (antialias) | Antialias bicubic |
| Tokenizer | HuggingFace tokenizers | Custom BPE (GPT-2 byte-level) + added_tokens |
| R-SWA (V3) | Full KV cache with mask | Two-range online softmax |
| Dependencies | PyTorch, transformers... | Only BLAS + stb_image |

## Acknowledgments

- Architecture inspiration: [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — pure C ASR inference
- Model V1/V2: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Model V3: [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
- Image loading: [stb_image](https://github.com/nothings/stb) (public domain)

## License

MIT