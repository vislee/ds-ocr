# ds-ocr — DeepSeek-OCR 纯 C 推理引擎

[English](README.md)

[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 和 [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 的纯 C 推理实现，架构模式参照 [antirez/qwen-asr](https://github.com/antirez/qwen-asr)。

- **3 种模型** — DeepSeek-OCR V1、V2 (DeepEncoder + 多裁剪)、Unlimited-OCR V3 (R-SWA)
- **零外部依赖** — 仅需 BLAS（Accelerate/OpenBLAS）+ stb_image.h
- **零拷贝权重加载** — mmap BF16 safetensors，按需转 F32
- **平台优化内核** — NEON / AVX2 / AVX-512 / scalar 自动调度
- **Metal GPU 加速** — Apple GPU MoE 解码（单命令缓冲区批处理）

## 快速开始

```bash
# 1. 下载模型权重（需要 huggingface_hub）
#    v1 = DeepSeek-OCR, v2 = DeepSeek-OCR-2（推荐）, v3 = Unlimited-OCR
./download_model.sh v2 ./models/DeepSeek-OCR-2

# 2. 编译（macOS 默认用 Accelerate BLAS + Metal GPU）
make blas

# 3. 运行 OCR
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03
```

## 模型对比

| | DeepSeek-OCR V1 | DeepSeek-OCR V2 ⭐ | Unlimited-OCR V3 |
|---|---|---|---|
| **HuggingFace** | `deepseek-ai/DeepSeek-OCR` | `deepseek-ai/DeepSeek-OCR-2` | `baidu/Unlimited-OCR` |
| **编码器** | CLIP ViT-L/14 | DeepEncoder V2 (Qwen2-0.5B) | CLIP ViT-L/14 + R-SWA |
| **解码器** | DeepSeek3B-MoE | DeepSeek3B-MoE | DeepSeek3B-MoE + R-SWA |
| **输入** | 1024×1024 (拉伸) | 动态多裁剪 | 640×640 (填充) + 多裁剪 |
| **视觉 token** | 256 | 857 (4-裁剪) | 111 ~ 3323 (1~30 裁剪) |
| **提示词** | `\nFree OCR.` | `\nFree OCR.` | `\ndocument parsing.` |
| **模型大小** | ~6.3 GB | ~6.7 GB | ~6.2 GB |

### 性能对比（Apple M2 Pro，8 线程，BLAS）

#### 大图（1794×1578，多裁剪）

| 指标 | V1 | V2 ⭐ | V3 |
|------|----|----|-----|
| **总耗时** | 12.2s | 14.8s | 79.7s (30 裁剪) |
| **编码** | 6.1s | 8.8s | 43.0s (30 裁剪 × SAM+CLIP) |
| **Prefill** | 1.0s (280 tok) | 1.0s (662 tok) | 6.4s (3328 tok) |
| **解码** | 5.0s (238 tok) | 5.0s (224 tok) | 30.3s (499 tok) |
| **解码速度** | 47.3 tok/s | 44.4 tok/s | 16.5 tok/s |
| **裁剪数** | 1 (1024×1024) | 4 (2×2 @768) | 30 (6×5 @640) |
| **输出质量** | ✅ 完整 | ✅ 完整 | ✅ 内容正确（轻微拼写偏差） |

#### 小图（400×100，≤640px，单裁剪）

| 指标 | V1 | V2 | V3 |
|------|----|----|-----|
| **总耗时** | 8.9s | 10.9s | 2.5s |
| **解码速度** | 38.8 tok/s | 36.9 tok/s | 48.0 tok/s |
| **视觉 token** | 256 | 257 | 111 |

> **推荐**：V2 是最佳全能选择 — 多裁剪适配任意尺寸图片，DeepEncoder V2 编码精度最高。
> V1 更快但会把图片拉伸到 1024×1024（宽高比失真），且有小概率精度拼写偏差。
> V3 小图表现优异（2.5s vs V1/V2 的 8-11s），支持语种检测和详细文档解析，
> 但大图因 30 裁剪顺序编码 + MHA 注意力速度显著更慢。
> V3 幻觉前缀（如 "The image contains no text..."）会自动在后处理中清除。

## 性能

### 优化历程（Apple M2 Pro，8 线程，BLAS，6-crop V2 图像）

| 阶段 | v0.5 | v0.8 | v1.0 | 优化手段 |
|------|------|------|------|---------|
| SAM+Encoder | ~50s | **8.8s** | **9.0s** | 并行 global crop |
| Prefill（~660 tok） | ~30s | **1.1s** | **0.93s** | 批量 MoE sgemm |
| Decode（226 tok） | ~19s | **6.2s** | **5.1s** | Argmax LM head + 连续 expert 块 |
| **总计** | **~97s** | **16s** | **15.0s** | |

**v1.0 = 8× v0.5 = 61× Python PyTorch（CPU BF16 ~736s）**

### 核心优化

- **Argmax LM head**：decode 时用 `ds_argmax_matvec_bf16` 替代 sgemm，边算边比较只保留最优 token（~8ms vs ~60ms/step），避免 631MB 权重转换。
- **连续 expert 块**：64 个 expert 权重在单层一块连续内存中，page fault 减少 2.4×。
- **Fused expert forward**：gate+up 投影合并为单次 matvec，提升 L2 cache 复用。
- **Metal GPU MoE**：Apple GPU 单命令缓冲区批处理。
- **BPE tokenizer 修复**：merge 正确加载（嵌套数组格式 + strdup key 拷贝 + 安全 JSON 跳过），修复 V2 退化输出。

### INT8 量化（`--int4`）

| | BF16 | INT8 (`--int4`) |
|---|---|---|
| **Expert 权重** | 3176 MB | 2399 MB (1.3× 更小) |
| **量化 RMS** | — | 0.007–0.010 |
| **OCR 精度** | ✅ 参考 | ✅ 匹配 BF16 |

```bash
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --int4
```

## 编译

```bash
make blas           # BLAS 加速（推荐）
make debug          # AddressSanitizer 调试构建
make clean          # 清理
make info           # 查看构建配置
```

## 使用

### CLI

```bash
# DeepSeek-OCR V2（推荐）
./ds_ocr -d ./models/DeepSeek-OCR-2 -i document.png --rp 1.03

# Unlimited-OCR V3（文档解析 + 语种检测）
./ds_ocr -d ./models/Unlimited-OCR -i document.png

# 静默模式：仅输出 OCR 文本
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --silent

# 性能分析
./ds_ocr -d ./models/DeepSeek-OCR-2 -i doc.png --rp 1.03 --profile

# macOS Vision OCR（不需要模型权重）
./ds_ocr -i doc.png --vision --silent
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
| `--min-tokens <n>` | 最少生成 token 数 | 32 |
| `--int4` | INT8 量化 MoE expert 权重 | 关闭 |
| `--vision` | macOS Vision OCR 后端 | 关闭 |
| `--vision-fast` | macOS Vision OCR 快速模式 | 关闭 |
| `--profile` | 阶段级耗时分析 | 关闭 |
| `--debug` | 详细调试输出 | 关闭 |
| `--silent` | 仅输出 OCR 文本 | 关闭 |

### 各模型推荐参数

| 模型 | `--rp` | `--ngram` | 备注 |
|------|--------|-----------|------|
| V2 ⭐ | 1.03 | 0 | 最佳质量，快速 |
| V3 | 1.01 | 35 | ngram 必须开启防重复 |
| V1 | 1.03 | 0 | 少量精度拼写偏差 |

### C API

```c
#include "ds_ocr.h"

ds_ctx_t *ctx = ds_load("./models/DeepSeek-OCR-2");
if (!ctx) { /* handle error */ }

ctx->max_new_tokens = 2048;
ctx->temperature    = 0.0f;

ds_set_token_callback(ctx, my_callback, userdata);

char *text = ds_recognize(ctx, "document.png");
printf("Result: %s\n", text);
free(text);

ds_free(ctx);
```

## 版本历史

- **v1.1** — V2 多裁剪修复（dynamic_preprocess min_num=2→4 裁剪，之前 9 裁剪导致截断），V3 小图修复（640×640→111 tokens，之前 1024→273），CLIP 位置编码双线性插值，V3 幻觉前缀自动清除，V3 禁用流式输出
- **v1.0** — BPE tokenizer merge 加载修复（3个 bug），V2 输出质量修复，Metal GPU MoE 批处理，INT8 量化
- **v0.9** — Unlimited-OCR V3 支持（CLIP+R-SWA），V3 多裁剪，tokenizer.json 回退
- **v0.8** — 批量 MoE prefill + 并行 encoding：16s 端到端（6× v0.5）
- **v0.7** — F32 KV cache + 融合 residual+norm + 直接 SwiGLU + 批量 decode
- **v0.6** — sgemm LM head + BF16 KV cache + 融合 decode attention + fast exp
- **v0.5** — MoE gate softmax 修复，decoder 正确性验证，7.5× vs Python

### 已知问题

1. **V1 精度**：V1 输出有少量 BF16 精度引起的拼写偏差（如 "raletimit" vs "ratelimit"），属正常精度范围。
2. **V3 输出标签**：`<|det|>` 和 `<|ref|>` 检测标签已在后处理中自动去除。幻觉前缀（如 "The image contains no text...[No text detected]"）也会自动清除。
3. **V3 大图拼写偏差**：V3 对大图（30+ 裁剪）可能出现轻微 OCR 拼写偏差（如 "CC"→"CF"），因局部裁剪丢失上下文，属模型限制。
4. **V3 小图 Non-Text**：V3 对极小图片（≤640px）可能输出 `[Non-Text]` 标记，模型主要在大文档图上训练。
5. **SAM encoder 精度漂移**：C 的 SAM+Encoder 输出与 Python 有微小差异（corr ~0.995），不影响 OCR 质量。

## 致谢

- 架构灵感：[antirez/qwen-asr](https://github.com/antirez/qwen-asr) — 纯 C ASR 推理
- 模型：[deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
- 图片加载：[stb_image](https://github.com/nothings/stb)（public domain）

## License

MIT
