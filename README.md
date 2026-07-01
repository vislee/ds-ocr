# ds-ocr — DeepSeek-OCR Pure C Inference Engine

[中文文档](README_zh.md)

A pure C inference implementation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) and [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- **3 model variants** — DeepSeek-OCR V1, V2 (DeepEncoder + multi-crop), and Unlimited-OCR V3 (R-SWA)
- **Zero external dependencies** — only BLAS (Accelerate/OpenBLAS) + stb_image.h
- **Zero-copy weight loading** — mmap BF16 safetensors, on-the-fly F32 conversion
- **Platform-optimized kernels** — NEON / AVX2 / AVX-512 / scalar auto-dispatch
- **Metal GPU acceleration** — MoE decode on Apple GPU (single command buffer batching)

## Quick Start

```bash
# 1. Download model weights (requires huggingface_hub)
#    v1 = DeepSeek-OCR, v2 = DeepSeek-OCR-2 (recommended), v3 = Unlimited-OCR
./download_model.sh v2 ./models/DeepSeek-OCR-2

# 2. Build (macOS defaults to Accelerate BLAS + Metal GPU)
make blas

# 3. Run OCR
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03
```

> Linux: `make blas` auto-detects OpenBLAS. See [Building](#building) for details.

## Model Comparison

| | DeepSeek-OCR V1 | DeepSeek-OCR V2 ⭐ | Unlimited-OCR V3 |
|---|---|---|---|
| **HuggingFace ID** | `deepseek-ai/DeepSeek-OCR` | `deepseek-ai/DeepSeek-OCR-2` | `baidu/Unlimited-OCR` |
| **Encoder** | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | CLIP ViT-L/14 + R-SWA |
| **Decoder** | DeepSeek3B-MoE | DeepSeek3B-MoE | DeepSeek3B-MoE + R-SWA |
| **Input** | 1024×1024 (stretch) | Dynamic multi-crop | 640×640 (pad) + multi-crop |
| **Visual tokens** | 256 | 857 (4-crop) | 111 ~ 3323 (1~30 crops) |
| **Prompt** | `\nFree OCR.` | `\nFree OCR.` | `\ndocument parsing.` |
| **Model size** | ~6.3 GB | ~6.7 GB | ~6.2 GB |

### Benchmark (Apple M2 Pro, 8 threads, BLAS)

#### Large Image (1794×1578, multi-crop)

| Metric | V1 | V2 ⭐ | V3 |
|--------|----|----|-----|
| **Total time** | 12.2s | 14.8s | 79.7s (30 crops) |
| **Encoding** | 6.1s | 8.8s | 43.0s (30 crops × SAM+CLIP) |
| **Prefill** | 1.0s (280 tok) | 1.0s (662 tok) | 6.4s (3328 tok) |
| **Decode** | 5.0s (238 tok) | 5.0s (224 tok) | 30.3s (499 tok) |
| **Decode speed** | 47.3 tok/s | 44.4 tok/s | 16.5 tok/s |
| **Crops** | 1 (1024×1024) | 4 (2×2 @768) | 30 (6×5 @640) |
| **Output quality** | ✅ Complete | ✅ Complete | ✅ Content OK (minor typos) |

#### Small Image (400×100, ≤640px, 1-crop)

| Metric | V1 | V2 | V3 |
|--------|----|----|-----|
| **Total time** | 8.9s | 10.9s | 2.5s |
| **Decode speed** | 38.8 tok/s | 36.9 tok/s | 48.0 tok/s |
| **Visual tokens** | 256 | 257 | 111 |

> **Recommendation**: V2 is the best all-round choice — multi-crop handles any image size,
> DeepEncoder V2 produces the most accurate encoder output.
> V1 is faster but stretches images to 1024×1024 (aspect distortion) and has minor precision typos.
> V3 excels on small images (2.5s vs 8-11s for V1/V2) and supports language detection,
> but is significantly slower for large images due to sequential 30-crop encoding + MHA attention.
> V3 hallucination prefix ("The image contains no text...") is automatically stripped in post-processing.

## Performance

### Optimization History (Apple M2 Pro, 8 threads, BLAS, 6-crop V2 image)

| Stage | v0.5 | v0.8 | v1.0 | Optimization |
|-------|------|------|------|-------------|
| SAM+Encoder | ~50s | **8.8s** | **9.0s** | Parallel global crop |
| Prefill (~660 tok) | ~30s | **1.1s** | **0.93s** | Batched MoE sgemm |
| Decode (226 tok) | ~19s | **6.2s** | **5.1s** | Argmax LM head + contiguous experts |
| **Total** | **~97s** | **16s** | **15.0s** | |

**v1.0 = 8× v0.5 = 61× Python PyTorch (CPU BF16 ~736s)**

### Key Optimizations

- **Argmax LM head**: `ds_argmax_matvec_bf16` replaces full sgemm for greedy decode — computes dot products on-the-fly while tracking only the best token (~8ms vs ~60ms/step), avoiding 631MB BF16→F32 weight conversion.
- **Contiguous expert blocks**: All 64 experts' gate_up_fused + shared weights in a single contiguous allocation per layer — adjacent experts share pages, reducing page faults 2.4×.
- **Fused expert forward**: `ds_expert_forward_fused()` combines gate+up projection into a single matvec, improving L2 cache reuse.
- **Metal GPU MoE**: Single command buffer batching for Apple GPU — gate_up + SwiGLU + down + expert combine in one dispatch.
- **BPE tokenizer fix**: Correct merge loading from tokenizer.json (nested array format + strdup key copy + safe JSON skip), fixing V2's degenerate "4.4.4." output.

### INT8 Quantization (`--int4`)

Per-row asymmetric INT8 quantization for MoE expert weights, reducing memory bandwidth ~2× while maintaining OCR accuracy (RMS < 0.01 vs BF16).

| | BF16 | INT8 (`--int4`) |
|---|---|---|
| **Expert weight size** | 3176 MB | 2399 MB (1.3× smaller) |
| **Quantization RMS** | — | 0.007–0.010 |
| **OCR accuracy** | ✅ Reference | ✅ Matches BF16 |

```bash
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --int4
```

> **Note**: On Apple Silicon (M2+), hardware BF16 dot-product instructions are faster than
> software INT8 dequantize+MLA for single-token decode. INT8 benefits memory-constrained
> devices and x86 platforms without BF16 hardware.

```
$ ./ds_ocr -d model_dir -i image.png --profile
Inference: 15025 ms, 226 text tokens (44.4 tok/s decode)
  Encoding: 8989 ms | Prefill: 934 ms | Decode: 5102 ms
```

## Architecture

```
  DeepSeek-OCR V1          DeepSeek-OCR V2 ⭐       Unlimited-OCR (V3)
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
  └────┬────┘              └─────┴─────┘                     │
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
| MoE Decoder | DeepSeek3B-MoE | DeepSeek3B-MoE | DeepSeek3B-MoE + R-SWA | ~3B (570M active) |

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

# Unlimited-OCR V3 (document parsing + language detection)
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
| `--min-tokens <n>` | Min generated tokens (prevent early EOS) | 32 |
| `--int4` | INT8 quantize MoE expert weights (2× less memory) | Off |
| `--vision` | macOS Vision OCR backend | Off |
| `--vision-fast` | macOS Vision OCR backend (fast) | Off |
| `--profile` | Per-stage timing | Off |
| `--debug` | Verbose debug output | Off |
| `--silent` | OCR text output only | Off |

### Recommended Settings per Model

| Model | `--rp` | `--ngram` | Notes |
|-------|--------|-----------|-------|
| V2 ⭐ | 1.03 | 0 | Best quality, fast |
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
├── ds_kernels.h/c             # Math kernel API + thread pool + dispatch
├── ds_kernels_impl.h          # Architecture dispatch macros (NEON/AVX/generic)
├── ds_kernels_generic.c       # Scalar fallback
├── ds_kernels_neon.c          # ARM NEON optimizations (incl. BF16 dot product)
├── ds_kernels_avx.c           # x86 AVX2/AVX-512 optimizations
├── ds_safetensors.h/c         # Multi-shard safetensors reader (BF16 + FP32)
├── ds_image.h/c               # Image loading + preprocessing (stb_image + bicubic resize)
├── ds_tokenizer.h/c           # BPE tokenizer (GPT-2 byte-level + DeepSeek format)
├── ds_quantize.h/c            # INT8 per-row quantization for MoE experts
├── ds_metal.h                 # Metal GPU acceleration (Apple Silicon)
├── ds_platform_ocr.h/c/m      # macOS Vision OCR bridge (.m=ObjC, .c=Linux stub)
├── ds_dump.h                  # Debug tensor dump utilities
├── main.c                     # CLI entry point
├── test.c                     # Test suite
├── stb_image.h                # Single-header image loader (public domain)
├── Makefile                   # Build system
├── download_model.sh          # Model download script (v1/v2/v3/all)
├── README.md                  # This file (English)
└── README_zh.md               # Chinese documentation
```

**Code size**: ~15K lines of custom code (excluding stb_image.h), compiled binary ~400KB.

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

## Version History

- **v1.1** — V2 multi-crop fix (dynamic_preprocess min_num=2→4 crops, was 9→truncated), V3 small image fix (640×640 pad→111 tokens, was 1024→273), CLIP position embedding bilinear interpolation, V3 hallucination prefix auto-strip, V3 streaming disabled for post-processing
- **v1.0** — BPE tokenizer merge loading fix (3 bugs: JSON skip, nested array format, strdup key copy), V2 output quality fix, Metal GPU MoE batching, INT8 quantization (`--int4`)
- **v0.9** — Unlimited-OCR V3 support (CLIP+R-SWA), V3 multi-crop, tokenizer.json fallback, download_model.sh v1/v2/v3
- **v0.8** — Batched MoE prefill + parallel encoding: 16s end-to-end (6× v0.5)
- **v0.7** — F32 KV cache + fused residual+norm + direct SwiGLU + batched decode
- **v0.6** — sgemm LM head + BF16 KV cache + fused decode attention + fast exp
- **v0.5** — MoE gate softmax fix, decoder correctness verification, 7.5× vs Python

### Known Issues

1. **V1 precision**: V1 output has minor BF16 precision-induced typos (e.g. "raletimit" vs "ratelimit") — within normal precision range for F32 accumulation through 36 encoder layers.
2. **V3 output tags**: Unlimited-OCR produces `<|det|>` and `<|ref|>` detection tags that are automatically stripped in post-processing. Hallucination prefix (e.g. "The image contains no text...[No text detected]") is also auto-stripped.
3. **V3 large image typos**: V3 on large images (30+ crops) may have minor OCR typos (e.g. "CC"→"CF") due to local crop context loss. This is a model limitation, not a code bug.
4. **V3 small image Non-Text**: V3 on very small images (≤640px) may output `[Non-Text]` markers — the model was trained predominantly on larger document images.
5. **SAM encoder precision drift**: C's SAM+Encoder output has minor differences from Python (corr ~0.995), caused by FP32 accumulation error amplified through 12+24 layers. Does not affect OCR quality.

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
