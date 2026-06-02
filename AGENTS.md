# AGENTS.md

## Project Overview

**ds-ocr** is a pure C inference engine for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), following the architecture patterns of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- **Tech Stack**: C99, BLAS (Accelerate/OpenBLAS), stb_image.h
- **Architecture**: SAM Vision Tokenizer → DeepEncoder V2 → MoE Decoder
- **Key Decisions**: Zero external dependencies, mmap'd BF16 safetensors weights, platform-optimized kernels (NEON/AVX/generic)

## Build & Test

```bash
make blas           # Build with BLAS acceleration
make debug          # Debug build with AddressSanitizer
make clean          # Remove build artifacts
./ds_ocr -d model_dir -i image.png   # Run OCR
```

## Key Conventions

- All public API uses `ds_` prefix
- Weights loaded as BF16 via mmap (zero-copy), converted on-the-fly during matmul
- MoE decoder: 64 routed experts (top-6) + 2 shared experts per layer
- Kernel dispatch via `ds_kernels_impl.h`: NEON/AVX/generic based on compile-time detection
- Follow the same code patterns as qwen-asr: thread pool, online softmax, fused BF16 matvec

## Component Mapping (qwen-asr → ds-ocr)

| qwen-asr | ds-ocr |
|----------|--------|
| Audio → Mel spectrogram | Image → RGB pixels |
| Conv2D stem + Transformer encoder | SAM ViT-B + Conv compression |
| Qwen3 LLM decoder (dense) | DeepSeek-V2 MoE decoder (64 experts) |
| Safetensors reader | Same (reused) |
| Kernels (NEON/AVX/generic) | Same (reused + MoE additions) |

## File Structure

- `ds_ocr.h/c` — Public API + coordinator
- `ds_kernels.c/h` — Math kernels + thread pool
- `ds_kernels_impl.h + _generic.c + _neon.c + _avx.c` — Platform-optimized kernels
- `ds_safetensors.c/h` — Safetensors reader
- `ds_image.c/h` — Image loading via stb_image
- `ds_visual_tokenizer.c/h` — SAM + Conv compression
- `ds_deep_encoder.c/h` — DeepEncoder V2 (Qwen2-0.5B + causal flow queries)
- `ds_moe_decoder.c/h` — MoE decoder with expert routing
- `ds_tokenizer.c/h` — BPE tokenizer
- `main.c` — CLI entry point
