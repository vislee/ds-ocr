# ds-ocr — DeepSeek-OCR 纯 C 推理引擎

[English](README.md)

[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 和 [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 的纯 C 推理实现，架构模式参照 [antirez/qwen-asr](https://github.com/antirez/qwen-asr)。

- **3 种模型** — DeepSeek-OCR V1、V2、Unlimited-OCR (V3)
- **零外部依赖** — 仅需 BLAS（Accelerate/OpenBLAS）+ stb_image.h
- **零拷贝权重加载** — mmap BF16 safetensors，按需转 F32
- **平台优化内核** — NEON / AVX2 / AVX-512 / scalar 自动调度

## 快速开始

```bash
# 1. 下载模型权重（需要 huggingface_hub）
#    v1 = DeepSeek-OCR, v2 = DeepSeek-OCR-2（推荐）, v3 = Unlimited-OCR
./download_model.sh v2 ./models/DeepSeek-OCR-2

# 2. 编译（macOS 默认用 Accelerate BLAS）
make blas

# 3. 运行 OCR
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03
```

## 模型对比

| | DeepSeek-OCR V1 | DeepSeek-OCR V2 | Unlimited-OCR V3 |
|---|---|---|---|
| **HuggingFace** | `deepseek-ai/DeepSeek-OCR` | `deepseek-ai/DeepSeek-OCR-2` | `baidu/Unlimited-OCR` |
| **编码器** | CLIP ViT-L/14 | DeepEncoder V2 | CLIP ViT-L/14 + R-SWA |
| **输入** | 1024×1024 (拉伸) | 动态多裁剪 | 640×640 (填充) |
| **视觉 token** | 256 | 857 (6-裁剪) | 273 |
| **提示词** | `\nFree OCR.` | `\nFree OCR.` | `\ndocument parsing.` |
| **C 支持** | ⚠️ 部分 | ✅ 完整 | ✅ 完整 |

### 性能对比（Apple M2 Pro，同一测试图片）

| 指标 | V1 | V2（推荐） | V3 (Unlimited-OCR) |
|------|----|-----------|---------------------|
| **总耗时** | 14.9s | 14.9s | 20.1s |
| **编码** | 6.7s | 9.2s | 6.5s |
| **Prefill** | 3.0s | 3.3s | 3.0s |
| **解码** | 5.1s (233 token) | 2.5s (106 token) | 10.5s (464 token) |
| **解码速度** | 45.5 tok/s | 43.1 tok/s | 44.1 tok/s |
| **输出质量** | ✅ 少量拼写偏差 | ✅ 正确 | ✅ 正确（额外标签） |

## 性能

Apple M2 Pro（8 线程，BLAS 加速），6-crop V2 图像：

| 阶段 | v0.5 | v0.8 | v0.9 | 优化手段 |
|------|------|------|------|---------|
| SAM+Encoder | ~50s | **8.8s** | **8.8s** | 并行 global crop |
| Prefill（862 tokens） | ~30s | **1.1s** | **1.1s** | 批量 MoE sgemm |
| Decode（280 tokens） | ~19s | **6.2s** | **~2s** | Argmax LM head + 连续 expert 块 |
| **总计** | **~97s** | **16s** | **~12s** | |

v0.9 主要优化：
- **Argmax LM head**：decode 时用 `ds_argmax_matvec_bf16` 替代 sgemm，
  避免 631MB BF16→F32 权重转换，边算边比较只保留最优 token（~8ms vs ~60ms/step）
- **Selective repetition penalty**：只重算 history tokens 的 logit（~100 个 vs 129280 个）
- **连续 expert 块（P2）**：64 个 expert 的 gate_up_fused + shared 权重
  分配在单层一块连续内存中，相邻 expert 共享内存页，page fault 减少 2.4×
  （~351ms → ~121ms/step）
- **madvise 预取**：MoE decode 时预取下一个 expert 权重，进一步减少 page fault
- **Fused expert forward**：`ds_expert_forward_fused()` 合并 gate+up 投影为单次 matvec，
  提升 L2 cache 对输入向量的复用
- **8 线程 decode**：argmax LM head 受益于 8T；小 expert matvec 略慢但总体更快

小图（1-crop V2，global-only）：~40s 总耗时（decode ~3-4 tok/s）

**v0.9 = 8× v0.5 = 61× Python PyTorch（CPU BF16 ~736s）**

```
$ ./ds_ocr -d model_dir -i image.png --profile
Inference: 12135 ms, 280 text tokens (14.3 tok/s decode)
  Encoding: 8783 ms | Prefill: 1092 ms | Decode: 2260 ms
```

## 架构

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

### 模型参数

| 组件 | V1 | V2 | V3 | 参数量 |
|------|----|----|--------|
| SAM Vision Tokenizer | ViT-B | ViT-B | ~86M |
| Encoder | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | CLIP ViT-L/14 (bypass Conv2d) | ~300M / ~500M / ~300M |
| Projector | 2048→1280 | 896→1280 (linear) | ~2.6M / ~1.1M |
| MoE Decoder | DeepSeek3B-MoE | DeepSeek3B-MoE | ~3B (570M active) |

### 关键架构细节

<details>
<summary>SAM Vision Tokenizer</summary>

- 12 transformer blocks，window attention（window_size=14），全局注意力在 layer [2,5,8,11]
- Fused QKV 投影，相对位置嵌入（rel_pos_h, rel_pos_w）
- Neck：V1 为 2×(Conv1×1+LN)，V2 为 Conv1×1+LN+Conv3×3+LN
- Downsample：net_2(256→512, k3, s2) + net_3(512→1024/896, k3, s2)

</details>

<details>
<summary>DeepEncoder V2</summary>

- Qwen2-0.5B 架构，24 层，hidden=896，14 MHA heads，2 KV heads（GQA）
- Causal flow queries：每 crop 144 queries，masked_scatter 到 image_size=640 的 857 个位置
- 学习式绝对位置编码（非 RoPE）
- 权重前缀：`model.qwen2_model.model.model.layers.*`

</details>

<details>
<summary>MoE Decoder</summary>

- 12 层，hidden=1280，10 heads，head_dim=128
- Layer 0: Dense FFN（SwiGLU, intermediate=6848）
- Layer 1-11: 64 routed experts (top-6) + 2 shared experts，expert intermediate=896
- Standard MHA + LLaMA-style RoPE（非 MLA，kv_heads=q_heads=10）
- BOS=0, EOS=1

</details>

<details>
<summary>V2 Multi-Crop 预处理</summary>

1. `find_closest_aspect_ratio()` 选择最优 crop ratio（最小 aspect diff，面积 tie-breaker）
2. 大图（如 1938×1210）→ ratio (3,2) → 6 local crops of 768×768
3. 小图（双维 ≤768）→ ratio (1,1) → 1 local crop of 768×768
4. Global view: `ImageOps.pad()` → 1024×1024（始终存在）
5. SAM: N local + 1 global → [896,12,12] / [896,16,16]
6. Token layout (image_size=640): num_queries=10 → 857 image slots
7. masked_scatter: 1121→857 (截断超出部分)
8. Prefix: BOS(1) + 857 image + 4 text ("\nFree OCR.") = 862 tokens

</details>

## Building

```bash
make blas           # BLAS 加速（推荐）
make debug          # AddressSanitizer 调试构建
make clean          # 清理
make info           # 查看构建配置
```

### 跨平台编译

```bash
# ARM64 (Apple Silicon)
make blas CC=clang CFLAGS="-Wall -O3 -arch arm64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"

# x86_64 (Intel)
make blas CC=clang CFLAGS="-Wall -O3 -arch x86_64 -DUSE_BLAS -DACCELERATE_NEW_LAPACK"
```

## 使用

### CLI

```bash
# 基础 OCR
./ds_ocr -d ./deepseek-ocr -i document.png

# 推荐参数：repetition penalty 1.01-1.03
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03

# 静默模式：仅输出 OCR 文本（适合管道/skill 集成）
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --silent

# 性能分析
./ds_ocr -d ./deepseek-ocr -i doc.png --rp 1.03 --profile

# macOS Vision OCR（不需要模型权重）
./ds_ocr -i doc.png --vision --silent
./ds_ocr -i doc.png --vision-fast --silent
```

### CLI 选项

| 选项 | 说明 | 默认 |
|------|------|------|
| `-d <dir>` | 模型目录 | 必需 |
| `-i <file>` | 输入图片 | 必需 |
| `-t <n>` | 线程数 | 全部 CPU |
| `-n <n>` | 最大生成 token 数 | 4096 |
| `--temp <f>` | 采样温度 | 0（贪心） |
| `--rp <f>` | 重复惩罚（1.0=关闭, 推荐 1.01-1.1） | 1.0 |
| `--ngram <n>` | 禁止重复 n-gram（0=关闭, 推荐 20-35） | 0 |
| `--min-tokens <n>` | 最少生成 token 数（阻止提前 EOS） | 256 |
| `--vision` | macOS Vision OCR 后端 | 关闭 |
| `--vision-fast` | macOS Vision OCR 后端（快速） | 关闭 |
| `--profile` | 阶段级耗时分析 | 关闭 |
| `--debug` | 详细调试输出 | 关闭 |
| `--silent` | 仅输出 OCR 文本 | 关闭 |

### C API

```c
#include "ds_ocr.h"

ds_ctx_t *ctx = ds_load("./deepseek-ocr");
if (!ctx) { /* handle error */ }

ctx->max_new_tokens = 2048;
ctx->temperature    = 0.0f;  /* greedy */

/* 流式回调（可选） */
ds_set_token_callback(ctx, my_callback, userdata);

/* 从文件识别 */
char *text = ds_recognize(ctx, "document.png");
printf("Result: %s\n", text);
free(text);

/* 从原始像素识别 */
char *text2 = ds_recognize_image(ctx, rgb_pixels, width, height, 3);
free(text2);

ds_free(ctx);
```

流式回调：

```c
void my_callback(const char *piece, void *userdata) {
    fputs(piece, stdout);  /* UTF-8 token fragment */
    fflush(stdout);
}
```

## 项目结构

```
ds-ocr/
├── ds_ocr.h/c                 # 公共 API + 模型加载 + 识别主流程
├── ds_visual_tokenizer.h/c    # SAM ViT-B（window attn, rel pos, neck, downsample）
├── ds_deep_encoder.h/c        # CLIP ViT-L/14 (V1) + DeepEncoder V2 (Qwen2-0.5B)
├── ds_moe_decoder.h/c         # MoE decoder（dense L0 + MoE L1-11）
├── ds_kernels.h/c              # 数学内核 API + 线程池 + 分发
├── ds_kernels_impl.h           # 架构分发宏 (NEON/AVX/generic)
├── ds_kernels_generic.c        # 标量回退
├── ds_kernels_neon.c           # ARM NEON 优化（含 BF16 dot product）
├── ds_kernels_avx.c            # x86 AVX2/AVX-512 优化
├── ds_safetensors.h/c          # 多分片 safetensors 读取（BF16 + FP32）
├── ds_image.h/c                # 图片加载 + 预处理（stb_image + bicubic resize）
├── ds_tokenizer.h/c            # Qwen2 BPE tokenizer + added_tokens
├── ds_platform_ocr.h/c/m       # macOS Vision OCR bridge（.m=ObjC, .c=Linux stub）
├── ds_dump.h                    # 调试张量 dump 工具
├── main.c                       # CLI 入口
├── test.c                       # 测试套件
├── stb_image.h                  # 单头图片加载器（public domain）
├── Makefile                     # 构建系统
├── download_model.sh            # 模型下载脚本
├── README.md                    # 英文文档
└── README_zh.md                 # 中文文档
```

**代码规模**：~12K 行自研代码（不含 stb_image.h），编译后二进制 ~331KB。

## 测试

```bash
make test              # 构建并运行全部测试（BLAS 后端）
make test_debug        # AddressSanitizer 模式
./test_ds_ocr test_kernels   # 运行指定测试套件
```

测试套件：
- **test_kernels** — 数学内核正确性（LayerNorm, RMSNorm, matmul, SwiGLU, attention）
- **test_safetensors** — Safetensors 读取（索引解析, BF16/FP32 转换）
- **test_tokenizer** — BPE encode/decode 往返
- **test_config** — 配置初始化和常量验证
- **test_integration** — 端到端流程（需要模型权重）

## 调试环境变量

开发调试用，设置后可 dump 中间张量或跳过管线阶段（详见源码 `ds_dump.h`）：

| 变量 | 说明 |
|------|------|
| `DS_DUMP_TENSORS` | 启用张量 dump |
| `DS_DUMP_PATCH_EMBED` | dump SAM patch embedding 输出 |
| `DS_DUMP_SAM_LAYERS` | 逐层 SAM attention/output + neck + downsample |
| `DS_DUMP_ENCODER` | dump encoder 输出 |
| `DS_DUMP_INPUT_EMBEDS` | dump 投影后输入嵌入 |
| `DS_DUMP_DECODER` | dump decoder layer 0 内部 |
| `DS_DUMP_DECODER_LAYERS` | dump 全部 decoder 层 |
| `DS_DUMP_DECODE_STEPS` | dump 逐步 decode 过程 |
| `DS_DUMP_LAYERS` | dump MoE expert routing 细节 |
| `DS_DUMP_CONV2_IM2COL` | dump ds_conv2d im2col buffer |
| `DS_DUMP_DIR` | 自定义 dump 输出目录 |
| `DS_SKIP_ENCODER` | 跳过 encoder，从文件加载嵌入 |
| `DS_PERFECT_ENCODER` | 用 Python 参考 .npy 覆盖 encoder 输出 |
| `DS_LOAD_ENCODER_OUTPUT` | 加载 Python encoder 输出（跳过 SAM+encoder） |
| `DS_LOAD_ENC_INPUT` | 加载 Python encoder 输入（跳过 SAM，保留 encoder） |
| `DS_LOAD_SAM_TOKENS` | 加载 Python SAM tokens（跳过 SAM，保留 encoder） |
| `DS_LOAD_SAM_ALL` | 加载完整 Python SAM tokens（global + all crops） |
| `DS_LOAD_PIL_PIXELS` | 加载 PIL 预处理像素（绕过 C resize） |
| `DS_LOAD_INPUT_EMBEDS` | 加载 Python inputs_embeds（decoder 调试） |
| `DS_BF16_CACHE_MB` | BF16 权重缓存大小（MB） |
| `DS_BF16_SIMULATE_PYTHON` | 截断中间值为 BF16 精度（匹配 Python） |
| `DS_SLOW_PREFILL` | 使用逐 token prefill（调试用，非批量 sgemm） |

## 权重加载

直接读取 HuggingFace safetensors 格式，mmap 零拷贝加载：

| 组件 | Tensor 命名模式 | 格式 |
|------|----------------|------|
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

## 平台优化

| 平台 | 内核集 | 关键操作 |
|------|--------|---------|
| **ARM (Apple Silicon)** | NEON + BF16 dot product | BF16→F32, RMSNorm, matmul, SwiGLU |
| **x86 (Intel/AMD)** | AVX2+FMA / AVX-512 | BF16→F32, RMSNorm, matmul, SwiGLU |
| **其他** | Generic (scalar) | 可移植 C 回退 |

## 当前状态

| 组件 | V1 | V2 | V3 |
|------|----|----|----|
| SAM Vision Tokenizer | ✅ | ✅ | ✅ |
| Encoder | ⚠️ | ✅（corr ~0.995 vs Python） | ✅ |
| Projector | ✅ | ✅ | ✅ |
| MoE Decoder | ✅ | ✅ | ✅ |
| R-SWA 解码注意力 | N/A | N/A | ✅ |
| Tokenizer | ✅ | ✅ | ✅ |
| Multi-crop | N/A | ✅ | ✅（单裁剪） |
| 端到端 OCR | ✅ | ✅ | ✅ |

### 版本历史

- **v0.9** — Unlimited-OCR V3 支持（CLIP+R-SWA），tokenizer.json 回退，download_model.sh v1/v2/v3
- **v0.8** — 批量 MoE prefill + 并行 encoding：16s 端到端（6× v0.5）
- **v0.7** — F32 KV cache + 融合 residual+norm + 直接 SwiGLU + 批量 decode
- **v0.6** — sgemm LM head + BF16 KV cache + 融合 decode attention + fast exp
- **v0.5** — MoE gate softmax 修复，decoder 正确性验证，7.5× vs Python

### 已知问题

1. **V1 CLIP 编码器**：V1 和 V3 使用相同的 CLIP 架构，CLIP bypass Conv2d 直接接收 SAM features。V1 输出存在少量 BF16 精度引起的拼写偏差（如 "raletimit" vs "ratelimit"），属于正常精度范围。
2. **V3 输出标签**：Unlimited-OCR 产生 `<|det|>` 和 `<|ref|>` 检测标签，后处理中已去除。部分格式（子项缩进）可能不同于 Python 输出。
3. **SAM encoder 精度漂移**：C 的 SAM+Encoder 输出与 Python 有微小差异（corr ~0.995），源于 FP32 累积误差经 12+24 层放大。不影响 OCR 质量。
4. **lm_head 权重独立**：`lm_head.weight` ≠ `embed_tokens.weight`，C 正确加载了独立权重。

## 与 Python 实现的差异

| 特性 | Python (PyTorch) | 本实现 (C) |
|------|-------------------|-----------|
| 权重格式 | 完整 FP32/BF16 张量 | mmap BF16（零拷贝） |
| Attention | FlashAttention / SDPA | Online softmax（O(1) 内存） |
| MoE routing | GPU scatter/gather | 批量 sgemm: grouped expert + shared 一步完成 |
| 位置编码 | 动态计算 | 预计算 RoPE 表 |
| 图片 resize | PIL BICUBIC (antialias) | Antialias bicubic |
| Tokenizer | HuggingFace tokenizers | 自定义 BPE (GPT-2 byte-level) + added_tokens |
| 依赖 | PyTorch, transformers... | 仅 BLAS + stb_image |

## 致谢

- 架构灵感：[antirez/qwen-asr](https://github.com/antirez/qwen-asr) — 纯 C ASR 推理
- 模型：[deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- 图片加载：[stb_image](https://github.com/nothings/stb)（public domain）

## License

MIT