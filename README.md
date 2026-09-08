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
| **Input** | 1024×1024 (stretch) | Dynamic multi-crop (768, 2–6 crops) | 640×640 (pad) + multi-crop |
| **Visual tokens** | 256 | 857 (6-crop) | 111 ~ 3323 (1~30 crops) |
| **Prompt** | `\nFree OCR.` | `\nFree OCR.` | `\ndocument parsing.` |
| **C support** | ✅ Full | ✅ Full | ✅ Full + EOS fix |
| **Model size** | ~6.3 GB | ~6.7 GB | ~6.2 GB |

### Benchmark

#### M4 Max (14-core, 36 GB, CPU BLAS), 1600×2264 document (Unlimited-OCR cover, 6 crops)

| Metric | V1 | V2 (6-crop) | V3 (Unlimited-OCR, 6 crops) |
|--------|----|----|-----------------------------|
| **Total time** | 7.5s | 11.4s | 10.1s *(was 20.6s)* |
| **Encoding** | 5.2s (SAM 4.7 + CLIP 0.5) | 8.9s | 7.0s *(was 13.8s)* |
| **Prefill** | 0.7–1.5s (280 tokens) | 1.9s (862 tokens) | 1.9s (908 tokens) *(was 3.5s)* |
| **Decode** | 0.8s (33 tokens) | 0.7s (16 tokens) | 1.3s (52 tokens) |
| **Decode speed** | 42–47 tok/s | 21–40 tok/s | 40 tok/s |
| **Output quality** | ⚠️ Minor typos | ✅ Correct | ✅ Text correct (det/grounding tags + artifacts auto-stripped) |

> V3 speedups vs the previous release: parallel crop encoding (SAM+CLIP run
> concurrently across crops, like V2), parallel BF16→F32 weight conversion in
> prefill, and float32 attention softmax in SAM. V2 prefill gains from the same
> parallel conversion; V1 from the SAM softmax + window-threading changes.

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

#### M2 Pro (8 threads, CPU BLAS), 1794×1578 large image + 400×100 small image (historical)

| Metric | V1 | V2 ⭐ | V3 |
|--------|----|----|-----|
| **Total time (large)** | 12.2s | 14.8s | 79.7s (30 crops) |
| **Encoding** | 6.1s | 8.8s | 43.0s (30 crops × SAM+CLIP) |
| **Prefill** | 1.0s (280 tok) | 1.0s (662 tok) | 6.4s (3328 tok) |
| **Decode** | 5.0s (238 tok) | 5.0s (224 tok) | 30.3s (499 tok) |
| **Decode speed** | 47.3 tok/s | 44.4 tok/s | 16.5 tok/s |
| **Total time (small)** | 8.9s | 10.9s | 2.5s |
| **Decode speed (small)** | 38.8 tok/s | 36.9 tok/s | 48.0 tok/s |
| **Visual tokens (small)** | 256 | 257 | 111 |

> Measured before V3 parallel crop encoding landed; on the current build the
> 30-crop encoding phase runs ~6× faster (see v1.1 optimizations below).

#### Model Quality Comparison

| Model | Strengths | Weaknesses | Best for |
|-------|-----------|------------|----------|
| **V2** | Dynamic multi-crop, best accuracy, flexible input sizes | Slower encoding (6+ crops) | **General use (recommended)** |
| **V1** | Fastest single-crop, simplest pipeline | 1024×1024 stretch, BF16 precision typos | Quick scans, small images |
| **V3** | Detailed output, sliding window attention | More crops for large images (parallel-encoded), occasional structural-tag artifacts (auto-stripped) | Small images ≤640px, structured docs |

> **Recommendation**: V2 is the best all-round choice — multi-crop handles any image size,
> DeepEncoder V2 produces the most accurate encoder output.
> V1 is faster but stretches images to 1024×1024 (aspect distortion) and has minor precision typos.
> V3 excels on small images (2.5s vs 8-11s for V1/V2) and supports language detection.
> On small images (≤640px, 1 crop), V3 matches V1 decode speed at ~47 tok/s.

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

Key optimizations in v1.1:
- **V3 parallel crop encoding**: Unlimited-OCR local crops (640×640) now run
  SAM+CLIP concurrently — one pthread per crop, wave-scheduled at `num_cpus`.
  V3 encoding on a 6-crop image: 13.8s → 7.5s.
- **Parallel BF16→F32 conversion**: The per-call weight conversion used by
  batched prefill (experts ~5GB per pass) is split across the thread pool.
  Prefill for ~900-token prompts: 3.5s → 1.8s.
- **Prefill weight prefetch**: `madvise(MADV_WILLNEED)` on the next layer's
  gate/up/down expert weights overlaps page-in with compute.
- **N-gram exclusion argmax**: banned tokens are collected before the LM-head
  argmax and skipped via `ds_argmax_matvec_bf16_excluding` — one ~8ms pass
  replaces the full-logits sgemm fallback (60ms+ plus a one-time 631MB
  conversion) on banned steps.
- **SAM float32 softmax**: attention softmax switched from double to float32,
  matching Python's FP32 softmax. SAM@1024 on V1: 6.3s → 5.1s.
- **SAM window-attention threading**: when a single SAM forward runs alone
  (V1, small images, V3 global view), the 25 per-layer windows run across
  raw pthreads (`DS_SAM_WIN_THREADS` tunable; auto-disabled during
  multi-crop encoding to avoid oversubscription).
- **V3 crop position embedding fix**: crops (100 tokens) now resample the
  16×16 CLIP position grid with antialiased bicubic interpolation, matching
  Python's `get_abs_pos` (`F.interpolate(mode='bicubic', antialias=True)`),
  instead of copying its first 100 rows — that copy was the root cause of
  V3's hallucinated `</td></tr></table>` prefix on multi-crop images.
- **V3 output cleanup**: orphaned leading HTML closing-tag runs
  (`</td></tr></table>`) and broken leading `<|/det|>` fragments are
  suppressed in both streaming output and final text.
- **N-gram ban capacity**: greedy decode collects up to 32 banned tokens per
  step (det-tag-heavy output can ban many) before falling back to full
  logits.
- **Encoding stage breakdown**: the timing summary now reports
  `SAM X ms + Encoder Y ms` separately.

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
2. `dynamic_preprocess(min_num=2, max_num=6)`: e.g. 1794×1578 → ratio (2,2) → 4 local crops of 768×768
3. Small image (both dims ≤768) → ratio (1,1) → 1 local crop of 768×768
4. Global view: `ImageOps.pad()` → 1024×1024 (always present)
5. SAM: N local + 1 global → [896,12,12] / [896,16,16]
6. DeepEncoder V2: 144 visual + 144 causal flow → 144 out per crop, 256 out for global
7. Token layout (image_size=640): num_queries=10 → 100 slots/crop, 257 global slots
8. masked_scatter: source=[local,global,sep] → fill [global(257),local(400)] positions
   4 crops: 833 source → 657 positions (176 dropped = masked_scatter truncation)
9. Prefix: BOS(1) + 657 image + 4 text ("\nFree OCR.") = 662 tokens

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

### V3 Output Post-Processing

Unlimited-OCR (V3) is trained to emit layout/grounding markup, so raw model
output can contain non-content artifacts. The engine strips them
automatically — both from the streamed display and from the text returned by
`ds_recognize()`:

| Artifact | Example | Handling |
|----------|---------|----------|
| Detection tags | `<\|det\|>title [x1,y1,x2,y2]<\|/det\|>` | removed (`ds_strip_det_tags`) |
| Reference+det pairs | `<\|ref\|>…<\|/ref\|><\|det\|>…<\|/det\|>` | removed, text kept |
| "No text" hallucination prefix | `The image contains no text…[No text detected]` | prefix removed |
| Orphaned HTML closing-tag prefix | `</td></tr></table>` | prefix removed (streaming + final text) |
| Broken leading det fragment | `text [x1,y1,x2,y2]<\|/det\|>` (missing opener) | fragment removed |
| Trailing/leading whitespace | — | trimmed (matches Python `.strip()`) |

Root cause of the closing-tag artifact: with position-embedding interpolation
now matching Python's `get_abs_pos`, the model emits these far less often; the
filter remains as a safety net for diagram/table-heavy pages.

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

- **v1.1** — OCR quality + speed: V3 parallel crop encoding, parallel BF16→F32 prefill conversion, n-gram exclusion argmax, SAM float32 softmax, SAM window-attention threading, V3 crop position-embedding interpolation fix (root cause of hallucinated table tags), V3 orphan closing-tag prefix strip; V2 multi-crop fix (dynamic_preprocess min_num=2), V3 small image fix (640×640 pad→111 tokens)
- **v1.0** — BPE tokenizer merge loading fix (3 bugs: JSON skip, nested array format, strdup key copy), V2 output quality fix, Metal GPU MoE batching, INT8 quantization (`--int4`)
- **v0.10** — INT8 per-row MoE expert quantization (`--int4`, RMS<0.01), platform auto-detect (M1/M2+/x86)
- **v0.9** — Unlimited-OCR V3 support (CLIP+R-SWA), V3 multi-crop, tokenizer.json fallback, download_model.sh v1/v2/v3
- **v0.8** — Batched MoE prefill + parallel encoding: 16s end-to-end (6× v0.5)
- **v0.7** — F32 KV cache + fused residual+norm + direct SwiGLU + batched decode
- **v0.6** — sgemm LM head + BF16 KV cache + fused decode attention + fast exp
- **v0.5** — MoE gate softmax fix, decoder correctness verification, 7.5× vs Python

### Known Issues

1. **V1 CLIP encoder**: V1 and V3 share the same CLIP architecture (bypass Conv2d, receive SAM features directly). V1 output has minor BF16 precision-induced typos (e.g. "raletimit" vs "ratelimit") — within normal precision range.
2. **V3 output tags**: Unlimited-OCR produces `<|det|>` and `<|ref|>` detection tags that are automatically stripped in post-processing. Hallucination prefixes (e.g. "The image contains no text...[No text detected]") and orphaned HTML closing-tag runs (e.g. `</td></tr></table>`) are also auto-stripped. The 830 added_tokens (including `<|det|>`, `<|ref|>`, `<|grounding|>`, `<td>`, `<tr>`, etc.) are now correctly loaded from `tokenizer.json` with vocab expansion beyond 128K.
3. **V3 large-image typos**: V3 on large images (30+ crops) may show minor OCR typos (e.g. "CC"→"CF") from local-crop context loss — a model limitation, not a code bug.
4. **V3 small-image Non-Text**: V3 on very small images (≤640px) may emit `[Non-Text]` markers — the model was trained predominantly on larger document images.
5. **SAM encoder precision drift**: C's SAM+Encoder output has minor differences from Python (corr ~0.995), caused by FP32 accumulation error amplified through 12+24 layers. V3 crop position embeddings use antialiased bicubic interpolation matching Python's `get_abs_pos`. Does not affect OCR quality.
6. **Multi-crop reading order**: on dense multi-crop pages (6+ crops), V2/V3 output line order can be jumbled — the model reads the crop sequence but section order does not always match the visual grid. Pre-existing model behavior, unrelated to encode speed.
7. **Independent lm_head weights**: `lm_head.weight` ≠ `embed_tokens.weight`; C correctly loads the independent weights.


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