# ds_ocr — DeepSeek-OCR Pure C Inference Engine

A pure C implementation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) inference, following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

**Zero external dependencies** — only BLAS (Accelerate/OpenBLAS) + stb_image.h.

## Architecture

```
Image → SAM Vision Tokenizer → DeepEncoder V2 → MoE Decoder → Text
         (80M params)          (500M params)    (3B params)
```

| Component | Description | Parameters |
|-----------|-------------|-----------|
| **SAM Vision Tokenizer** | ViT-B encoder + 2 Conv compression layers | 80M |
| **DeepEncoder V2** | Qwen2-0.5B with causal flow queries | 500M |
| **MoE Decoder** | DeepSeek-V2 MoE (64 experts, top-6 + 2 shared) | 3B (570M active) |

### Key Features
- **Memory-mapped BF16 weights** from safetensors — instant model loading
- **Platform-optimized kernels**: NEON (ARM) / AVX2+FMA / AVX-512 (x86) / generic fallback
- **Streaming token output** via callback API
- **Mixed attention** in DeepEncoder V2: bidirectional for visual tokens, causal for flow queries
- **MoE routing**: 64 routed experts with top-6 gating + 2 always-active shared experts

## Building

```bash
# With BLAS acceleration (recommended)
make blas

# Debug build
make debug
```

### Requirements
- GCC or Clang
- BLAS: Apple Accelerate (macOS) or OpenBLAS (Linux)
- For image loading: stb_image.h (included)

## Usage

```bash
# Download model weights
./download_model.sh ./deepseek-ocr

# Run OCR on an image
./ds_ocr -d ./deepseek-ocr -i document.png

# Options
./ds_ocr -d ./deepseek-ocr -i doc.png -t 8 -n 2048 --debug
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

## C API

```c
#include "ds_ocr.h"

/* Load model */
ds_ctx_t *ctx = ds_load("./deepseek-ocr");

/* Set streaming callback */
ds_set_token_callback(ctx, my_callback, userdata);

/* Recognize image */
char *text = ds_recognize(ctx, "document.png");
printf("%s\n", text);
free(text);

/* Free resources */
ds_free(ctx);
```

## Project Structure

```
ds-ocr/
├── ds_ocr.h/c              # Public API + coordinator
├── ds_kernels.c/h           # Math kernels + thread pool
├── ds_kernels_impl.h        # Architecture dispatch (NEON/AVX/generic)
├── ds_kernels_generic.c     # Scalar fallback kernels
├── ds_kernels_neon.c        # ARM NEON optimized kernels
├── ds_kernels_avx.c         # x86 AVX2/AVX-512 kernels
├── ds_safetensors.c/h       # Safetensors reader (multi-shard)
├── ds_image.c/h             # Image loading (PNG/JPEG via stb_image)
├── ds_visual_tokenizer.c/h  # SAM ViT-B + Conv compression
├── ds_deep_encoder.c/h      # DeepEncoder V2 (Qwen2-0.5B based)
├── ds_moe_decoder.c/h       # MoE decoder (64 experts, top-6)
├── ds_tokenizer.c/h         # BPE tokenizer (Qwen2-compatible)
├── main.c                   # CLI entry point
├── stb_image.h              # Single-header image loader
├── Makefile                 # Build system
└── download_model.sh        # Model download script
```

## Model Support

| Model | Version | Encoder | Visual Tokens |
|-------|---------|---------|---------------|
| DeepSeek-OCR | v1 | CLIP ViT-L | 256-1120 |
| DeepSeek-OCR-2 | v2 | DeepEncoder V2 (Qwen2-0.5B) | 256-1120 |

## Credits

- Architecture inspired by [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — pure C ASR inference
- Model by [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Image loading by [stb_image](https://github.com/nothings/stb)

## License

MIT
