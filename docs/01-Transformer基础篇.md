# 01 - Transformer 基础篇：从零理解大模型的骨架

> **阅读目标**：读完本篇，你将理解 Transformer 的每一个核心组件——注意力机制、多头注意力、前馈网络、位置编码、归一化、残差连接——并能在 ds-ocr 和 qwen-asr 的 C 代码中找到对应的实现。

---

## 目录

1. [为什么需要 Transformer？](#1-为什么需要-transformer)
2. [整体架构鸟瞰](#2-整体架构鸟瞰)
3. [嵌入层：把文字变成数字](#3-嵌入层把文字变成数字)
4. [位置编码：让模型知道"谁在前面"](#4-位置编码让模型知道谁在前面)
5. [自注意力机制：模型的核心引擎](#5-自注意力机制模型的核心引擎)
6. [多头注意力：多个视角看世界](#6-多头注意力多个视角看世界)
7. [前馈网络(FFN)：逐位置的非线性变换](#7-前馈网络ffn逐位置的非线性变换)
8. [层归一化与残差连接：稳定训练的秘诀](#8-层归一化与残差连接稳定训练的秘诀)
9. [编码器 vs 解码器：双向与单向的区别](#9-编码器-vs-解码器双向与单向的区别)
10. [在代码中找到一切](#10-在代码中找到一切)

---

## 1. 为什么需要 Transformer？

### 1.1 从 RNN 到 Transformer

想象你正在读一本书。RNN（循环神经网络）就像一个**逐字逐句**读书的人——读完第一个字才能读第二个字，读完第二个才能读第三个。这种"串行"方式有两个致命问题：

- **慢**：无法并行计算，100个词必须按顺序处理100步
- **健忘**：读到第100个词时，第1个词的信息已经被"稀释"了（梯度消失）

Transformer 的革命性在于：**它同时看所有词**，就像一个人一眼扫过整页书，然后用"注意力"决定哪些词之间有关系。

### 1.2 核心思想：注意力机制

> **一句话理解**：注意力机制就是让每个词"问"所有词一个问题，根据回答的相关性分配权重，然后综合所有回答得到自己的新表示。

这就像开会时，每个人都可以同时关注所有人，而不是只能听前一个人说话。

---

## 2. 整体架构鸟瞰

原始 Transformer（2017 年论文 *Attention Is All You Need*）由**编码器**和**解码器**两大部分组成：

```
输入序列 → [编码器 ×N层] → 编码表示
                                         ↘
目标序列 → [解码器 ×N层] → 输出概率
```

但现代大模型的架构更加多样：

| 架构 | 代表模型 | 特点 | 本项目对应 |
|------|---------|------|-----------|
| **Encoder-Only** | BERT | 双向注意力，擅长理解 | ds-ocr 的 SAM ViT 编码器 |
| **Decoder-Only** | GPT, LLaMA, Qwen | 因果注意力（只看左边），擅长生成 | ds-ocr 的 MoE 解码器, qwen-asr 的解码器 |
| **Encoder-Decoder** | T5, 原始 Transformer | 编码器理解，解码器生成 | ds-ocr 的整体：编码器理解图像，解码器生成文字 |

**在 ds-ocr 中**：
- **SAM ViT-B** = 视觉编码器（理解图像）
- **DeepEncoder V2** = 文本编码器（深化视觉特征）
- **MoE Decoder** = 解码器（生成 OCR 文本）

**在 qwen-asr 中**：
- **音频编码器** = 理解音频（Conv2D stem + Transformer encoder）
- **LLM 解码器** = 生成转录文字（Qwen3 decoder）

---

## 3. 嵌入层：把文字变成数字

### 3.1 什么是嵌入？

计算机只懂数字，不懂文字。**嵌入（Embedding）** 就是把每个词元（token）映射成一个高维向量（通常 768~4096 维）。

可以把它想象成一本"数字词典"：
- "猫" → [0.23, -0.15, 0.89, ...]
- "狗" → [0.21, -0.12, 0.85, ...]
- "桌子" → [-0.50, 0.33, -0.22, ...]

语义相近的词，向量也相近（"猫"和"狗"的向量比"猫"和"桌子"更接近）。

### 3.2 在代码中的实现

嵌入本质上就是一个**查找表**（lookup table）：一个 `[vocab_size, hidden_dim]` 的矩阵。给定 token ID，直接取对应行。

**ds-ocr 中的嵌入**（`ds_ocr.c`）：

```c
/* 嵌入矩阵：[129280, 1280] 的 BF16 权重 */
uint16_t *tok_embeddings_bf16;  /* BF16 格式，零拷贝 mmap */

/* 查表：token ID → 向量 */
const uint16_t *emb = tok_embeddings_bf16 + (size_t)token_id * hidden;
/* BF16 → F32 转换 */
for (int i = 0; i < hidden; i++) {
    uint32_t f32_bits = ((uint32_t)emb[i]) << 16;
    memcpy(&output[i], &f32_bits, sizeof(float));
}
```

**qwen-asr 中的嵌入**（`qwen_asr_decoder.c`）：

```c
/* 嵌入矩阵：[151936, 2048] 的 BF16 权重 */
uint16_t *tok_embeddings_bf16;
/* 同样的查表方式 */
```

> **关键细节**：ds-ocr 和 qwen-asr 都将嵌入权重存储为 **BF16**（Brain Float 16），通过 mmap 直接映射磁盘文件，避免加载到内存。推理时按需转为 F32 计算。这是一种**零拷贝**优化。

### 3.3 Tied Embeddings（嵌入共享）

两个项目都使用了**嵌入共享**（tied embeddings）：输入嵌入矩阵 = 输出投影矩阵（lm_head）。

```c
/* 如果没有单独的 lm_head 权重，就用嵌入矩阵做输出投影 */
if (dec->lm_head_bf16)
    ds_bf16_matvec_pub(logits, x, dec->lm_head_bf16, ...);
else
    ds_bf16_matvec_pub(logits, x, dec->tok_embeddings_bf16, ...);  /* 共享 */
```

这就像：认路（输入嵌入）和指路（输出投影）用的是同一张地图，节省了约 25% 的参数量。

---

## 4. 位置编码：让模型知道"谁在前面"

### 4.1 为什么需要位置编码？

自注意力机制是**顺序无关**的——"猫追老鼠"和"老鼠追猫"对它来说完全一样！位置编码就是给每个位置"打标签"，让模型区分先后。

### 4.2 两种主要的位置编码

#### 正弦位置编码（原始 Transformer、SAM ViT）

用 sin/cos 函数生成固定位置向量：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**在 ds-ocr SAM 中的实现**（`ds_visual_tokenizer.c`）：

```c
/* SAM 的位置嵌入是预训练好的固定向量，直接从权重文件加载 */
float *sam_pos_embed;  /* [577, 768] — 1个CLS + 576个patch */
```

#### RoPE 旋转位置编码（现代大模型的主流选择）

RoPE 的核心思想：**不把位置信息加到向量上，而是把位置信息"旋转"进注意力计算中**。

具体来说，对 Q 和 K 向量的每对相邻维度施加旋转变换：

```
[q₁, q₂] → [q₁cos(θ) - q₂sin(θ), q₁sin(θ) + q₂cos(θ)]
```

其中 θ = pos × (1/10000^(2i/d))。

> **为什么 RoPE 更好？**
> - Q₁·K₁ 的内积自然包含了**相对位置**信息（pos_m - pos_n）
> - 支持外推到更长的序列
> - 不增加额外参数

**在 ds-ocr 解码器中的实现**（`ds_moe_decoder.c`）：

```c
/* 预计算 RoPE 缓存 */
for (int i = 0; i < max_seq; i++) positions[i] = i;
ds_compute_rope_neox(rope_cache_cos, rope_cache_sin,
                      positions, max_seq, head_dim, rope_theta);

/* 推理时应用 RoPE（解码器每一步） */
ds_apply_rope_neox(q, cos_vals, sin_vals, 1, n_heads, head_dim);
ds_apply_rope_neox(k, cos_vals, sin_vals, 1, n_kv_heads, head_dim);
```

**在 qwen-asr 解码器中的实现**（`qwen_asr_decoder.c`）：

```c
/* 同样的 NeoX 风格 RoPE */
qwen_compute_rope_neox(cos_out, sin_out, positions, seq, head_dim, theta);
qwen_apply_rope_neox(q, rope_cos, rope_sin, seq, n_heads, head_dim);
```

> **两个项目的 RoPE 参数不同**：
> - ds-ocr 解码器：theta = 10000.0
> - qwen-asr 解码器：theta = 1,000,000.0（支持更长的音频序列）

### 4.3 2D 位置编码（视觉模型专用）

SAM 处理图像时，每个 patch 有**行列坐标**，需要 2D 位置编码：

```c
/* ds_visual_tokenizer.c 中的 2D 位置嵌入 */
void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols, int embed_dim);
```

---

## 5. 自注意力机制：模型的核心引擎

### 5.1 通俗理解

自注意力（Self-Attention）就像一个**智能讨论会**：

1. 每个 token 提出**问题**（Query）："我在找和谁有关系？"
2. 每个 token 展示**标签**（Key）："我有这样的特征"
3. 每个 token 携带**信息**（Value）："我的实际内容是"

Query 和 Key 的**点积**衡量"问题和标签的匹配度"，匹配度高就分配更多注意力。

### 5.2 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

分解步骤：
1. **打分**：$S = QK^T$（每个 query 和每个 key 做点积，得到注意力分数矩阵）
2. **缩放**：$S = S / \sqrt{d_k}$（防止数值过大导致 softmax 梯度消失）
3. **归一化**：$A = \text{softmax}(S)$（分数转为概率分布，每行和为 1）
4. **聚合**：$O = AV$（用注意力权重对 Value 做加权求和）

### 5.3 在代码中的实现

**ds-ocr 编码器中的双向注意力**（`ds_kernels.c`）：

```c
/* 编码器：所有 token 可以看到所有 token（双向） */
void ds_bidirectional_attention(float *out, const float *Q, const float *K,
                                const float *V, int seq, int n_heads,
                                int head_dim, float scale);
```

**ds-ocr/qwen-asr 解码器中的因果注意力**（`ds_kernels.c` / `qwen_asr_kernels.c`）：

```c
/* 解码器：每个 token 只能看到自己和之前的 token（因果掩码） */
void ds_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int q_offset);
```

> **因果掩码（Causal Mask）**：解码器生成文字时，不能"偷看"未来的词。因果掩码就是一个上三角矩阵，把未来位置的注意力分数设为 -∞（softmax 后变 0）。

### 5.4 QKV 投影：从输入到 Q、K、V

输入向量 x 通过三个**线性变换**（矩阵乘法）得到 Q、K、V：

```
Q = x @ Wq^T    (Wq: [n_heads × head_dim, hidden])
K = x @ Wk^T    (Wk: [n_kv_heads × head_dim, hidden])
V = x @ Wv^T    (Wv: [n_kv_heads × head_dim, hidden])
```

**在解码器的单 token 推理中**（`ds_moe_decoder.c`）：

```c
/* 优化的 QKV 融合计算：一次线程调度同时算 Q、K、V */
ds_linear_nobias_bf16_qkv(q, k, v, x_norm,
                            layer->wq_weight_bf16,
                            layer->wk_weight_bf16,
                            layer->wv_weight_bf16,
                            hidden, q_dim, kv_dim);
```

> **为什么融合 QKV？** 单 token 推理时，三次矩阵×向量运算可以合并为一次循环遍历权重，减少内存访问次数——这就是 **kernel fusion** 的思想。

---

## 6. 多头注意力：多个视角看世界

### 6.1 为什么需要多头？

单头注意力只学一种"关注模式"。多头注意力让模型**同时从多个角度**关注不同的关系：

- 头 1 可能关注语法关系（主语→谓语）
- 头 2 可能关注指代关系（"它"→"猫"）
- 头 3 可能关注位置关系（相邻词）
- ......

### 6.2 计算方式

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) @ Wo

其中 headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

每个头独立计算注意力，最后拼接并做一次输出投影。

### 6.3 MHA → GQA 的演进

| 类型 | Q头数 | K/V头数 | 内存 | 代表模型 |
|------|-------|---------|------|---------|
| **MHA**（多头注意力） | h | h | 大 | 原始 Transformer, ds-ocr 解码器 |
| **GQA**（分组查询注意力） | h | h/g | 中 | LLaMA-2, qwen-asr 解码器, ds-ocr V2 编码器 |
| **MQA**（多查询注意力） | h | 1 | 小 | PaLM |

**MHA**：每个 Q 头有独立的 K、V 头。
**GQA**：g 个 Q 头共享一组 K、V 头。既保留了多头的多样性，又减少了 KV 缓存的内存。
**MQA**：所有 Q 头共享一组 K、V。最省内存，但表达能力稍弱。

### 6.4 在代码中

**ds-ocr 解码器**：标准 MHA
```c
int n_heads = 10;     /* Q 头数 */
int n_kv_heads = 10;  /* K/V 头数 = Q 头数（标准 MHA） */
```

**ds-ocr V2 编码器**：GQA
```c
int enc_heads = 14;     /* Q 头数 */
int enc_kv_heads = 2;   /* K/V 头数（7个Q头共享1组KV） */
```

**qwen-asr 解码器**：GQA
```c
int n_heads = 16;       /* Q 头数 */
int n_kv_heads = 8;     /* K/V 头数（2个Q头共享1组KV） */
```

---

## 7. 前馈网络(FFN)：逐位置的非线性变换

### 7.1 标准前馈网络

```
FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
```

两层线性变换，中间夹一个激活函数。每个位置独立计算（不涉及 token 间交互）。

> **注意力负责"交流"（token间），FFN负责"思考"（单token）**。

### 7.2 SwiGLU：现代大模型的选择

LLaMA、DeepSeek、Qwen 等现代模型使用 **SwiGLU** 替代传统 FFN：

```
SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

- `SiLU(x) = x · σ(x)`（自门控激活函数，也叫 Swish）
- `⊙` 是逐元素乘法（Hadamard 乘积）
- 三个权重矩阵（gate、up、down）替代了原来的两个

**在 ds-ocr 解码器中**（`ds_moe_decoder.c`）：

```c
/* Gate + Up 投影 */
ds_linear_nobias_bf16(gate_buf, x, layer->dense_gate_weight_bf16, ...);
ds_linear_nobias_bf16(up_buf, x, layer->dense_up_weight_bf16, ...);

/* SwiGLU: SiLU(gate) * up */
ds_swiglu_multiply(swiglu_buf, gate_up_buf, seq_len, intermediate);

/* Down 投影 */
ds_linear_nobias_bf16(output, swiglu_buf, layer->dense_down_weight_bf16, ...);
```

**在 qwen-asr 解码器中**（`qwen_asr_decoder.c`）：

```c
/* 融合 gate+up：一次矩阵乘法，输出交错排列 [g0,u0,g1,u1,...] */
qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16, ...);
/* 一步完成 SiLU(gate) * up */
qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
/* Down 投影 */
qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, ...);
```

> **优化技巧**：qwen-asr 将 gate 和 up 权重**预融合**为一个矩阵（行交错排列），推理时只需一次矩阵乘法，然后 `swiglu_multiply` 一步完成 SiLU + 逐元素乘法。这种 **kernel fusion** 大大减少了内存访问。

### 7.3 编码器中的 GELU FFN

两个项目的编码器使用传统 GELU FFN（而非 SwiGLU）：

- **SAM ViT**：GELU FFN，有 bias
- **qwen-asr 音频编码器**：GELU FFN，有 bias
- **DeepEncoder V2**：SwiGLU（与解码器一致）

---

## 8. 层归一化与残差连接：稳定训练的秘诀

### 8.1 LayerNorm vs RMSNorm

| 归一化方式 | 公式 | 有 bias | 用在哪 |
|-----------|------|---------|--------|
| **LayerNorm** | (x - μ) / √(σ² + ε) * γ + β | 有 | SAM, CLIP, qwen-asr 编码器 |
| **RMSNorm** | x / RMS(x) * γ | 无 | ds-ocr 解码器, qwen-asr 解码器 |

RMSNorm 是 LayerNorm 的简化版——去掉了均值中心化和偏置项，计算更快，效果基本相当。

**在 ds-ocr 解码器中**（`ds_kernels.c`）：

```c
/* RMSNorm：不需要计算均值，只算 RMS */
void ds_rms_norm(float *out, const float *x, const float *weight,
                 int seq_len, int hidden, float eps);

/* Per-head RMSNorm：对 Q/K 的每个头单独归一化（DeepSeek-V2 特色） */
void ds_rms_norm_per_head(float *x, const float *weight,
                           int seq_len, int n_heads, int head_dim, float eps);
```

### 8.2 Pre-Norm vs Post-Norm

```
Pre-Norm (现代模型):    x → Norm → Attn → +x → Norm → FFN → +x
Post-Norm (原始论文):   x → Attn → +x → Norm → FFN → +x → Norm
```

现代大模型几乎都使用 **Pre-Norm**（先归一化再计算），因为它训练更稳定。

**在 ds-ocr 解码器中**（`ds_moe_decoder.c`）：

```c
/* Pre-Norm 模式 */
ds_rms_norm(x_norm, x, layer->input_norm, ...);        /* 先归一化 */
ds_linear_nobias_bf16_qkv(q, k, v, x_norm, ...);       /* 再计算注意力 */
/* ... 注意力计算 ... */
for (int i = 0; i < hidden; i++) out[i] = x[i] + proj_out[i];  /* 残差连接 */
```

### 8.3 残差连接

残差连接（Residual Connection）就是把**输入直接加到输出上**：

```
output = x + SubLayer(x)
```

> **为什么有效？** 残差连接创建了"捷径"，让梯度可以直接流过深层网络，避免梯度消失。就像高速公路的应急车道——即使主路堵车（梯度消失），信息也能通过应急车道流通。

### 8.4 Per-head Q/K RMSNorm（DeepSeek 特色）

DeepSeek-V2 引入了对 Q 和 K **每个头单独做 RMSNorm** 的做法，这是防止注意力分数爆炸的关键技巧。

**在 ds-ocr 解码器中**（`ds_moe_decoder.c`）：

```c
/* 对 Q 的每个头单独归一化 */
if (layer->q_norm_weight)
    ds_rms_norm_per_head(q, layer->q_norm_weight, 1, n_heads, head_dim, eps);
/* 对 K 的每个头单独归一化 */
if (layer->k_norm_weight)
    ds_rms_norm_per_head(k, layer->k_norm_weight, 1, n_kv_heads, head_dim, eps);
```

---

## 9. 编码器 vs 解码器：双向与单向的区别

### 9.1 编码器：双向注意力

编码器用于**理解**输入（图像、音频），所有 token 可以互相看到：

```
token 1: 可以看 [1,2,3,4,5]  ← 全部
token 2: 可以看 [1,2,3,4,5]  ← 全部
...
```

**典型应用**：SAM ViT（图像理解）、CLIP（图文对齐）、qwen-asr 音频编码器

### 9.2 解码器：因果注意力

解码器用于**生成**输出，只能看到当前和之前的 token：

```
token 1: 只能看 [1]          ← 自己
token 2: 只能看 [1,2]        ← 自己和之前
token 3: 只能看 [1,2,3]      ← 自己和之前
...
```

这通过**因果掩码**（下三角矩阵）实现。

**典型应用**：ds-ocr MoE 解码器、qwen-asr LLM 解码器

### 9.3 混合注意力（DeepEncoder V2 特色）

ds-ocr 的 DeepEncoder V2 引入了**混合注意力**：视觉 token 用双向注意力，因果流查询（causal flow queries）用因果注意力。

```c
/* ds_kernels.c */
void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale);
```

> 这是一种巧妙的设计：视觉 token 之间可以充分交流（双向），而因果流查询则按顺序"收集"信息（因果），实现了从并行理解到序列生成的过渡。

---

## 10. 在代码中找到一切

### ds-ocr 项目中 Transformer 组件的对应关系

| 组件 | 代码位置 | 函数/变量 |
|------|---------|----------|
| 嵌入查表 | `ds_ocr.c` | `tok_embeddings_bf16` + BF16→F32 转换 |
| RoPE 位置编码 | `ds_kernels.c/h` | `ds_compute_rope_neox`, `ds_apply_rope_neox` |
| 2D 位置编码 | `ds_visual_tokenizer.c` | `ds_compute_2d_position_embeddings` |
| 双向注意力 | `ds_kernels.c` | `ds_bidirectional_attention` |
| 因果注意力 | `ds_kernels.c` | `ds_causal_attention` |
| 混合注意力 | `ds_kernels.c` | `ds_mixed_attention` |
| RMSNorm | `ds_kernels.c` | `ds_rms_norm`, `ds_rms_norm_per_head` |
| LayerNorm | `ds_kernels.c` | `ds_layer_norm` |
| SwiGLU FFN | `ds_kernels.c` | `ds_swiglu_multiply` |
| GELU | `ds_kernels.c` | `ds_gelu` |
| 线性变换(BF16) | `ds_kernels.c` | `ds_linear_nobias_bf16`, `ds_linear_nobias_bf16_qkv` |
| 残差连接 | `ds_moe_decoder.c` | `out[i] = x[i] + proj_out[i]` (内联) |
| Softmax | `ds_kernels.c` | `ds_softmax` |

### qwen-asr 项目中 Transformer 组件的对应关系

| 组件 | 代码位置 | 函数/变量 |
|------|---------|----------|
| 嵌入查表 | `qwen_asr_decoder.c` | `tok_embeddings_bf16` |
| RoPE | `qwen_asr_kernels.c` | `qwen_compute_rope_neox`, `qwen_apply_rope_neox` |
| 窗口化双向注意力 | `qwen_asr_kernels.c` | `qwen_bidirectional_attention` |
| 因果注意力(GQA) | `qwen_asr_kernels.c` | `qwen_causal_attention` |
| RMSNorm | `qwen_asr_kernels.c` | `qwen_rms_norm`, `qwen_rms_norm_per_head` |
| LayerNorm | `qwen_asr_kernels.c` | `qwen_layer_norm` |
| SwiGLU FFN(融合) | `qwen_asr_decoder.c` | `gate_up_fused_bf16` + `qwen_swiglu_multiply` |
| GELU FFN | `qwen_asr_encoder.c` | `qwen_gelu` + fc1/fc2 |
| 线性变换 | `qwen_asr_kernels.c` | `qwen_linear_nobias_bf16`, `qwen_linear_nobias_bf16_qkv` |

---

## 小结

你现在理解了 Transformer 的每一个核心组件：

1. **嵌入**：把 token 变成向量（查表）
2. **位置编码**：注入位置信息（RoPE 旋转编码）
3. **自注意力**：token 之间交流信息（QKV + softmax）
4. **多头注意力**：多角度关注（并行多头 + 输出投影）
5. **FFN**：逐位置非线性变换（SwiGLU）
6. **归一化**：稳定训练（RMSNorm/LayerNorm）
7. **残差连接**：梯度捷径（x + SubLayer(x)）
8. **因果掩码**：生成时不能偷看未来

**下一步**：在 [02-MoE混合专家篇](./02-MoE混合专家篇.md) 中，我们将深入 MoE 架构——看 ds-ocr 的 64 个专家如何分工合作。