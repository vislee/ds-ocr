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

RNN（Recurrent Neural Network，循环神经网络）处理序列的方式是**逐个推进**——读第 1 个词才能处理第 2 个词，处理第 2 个才能到第 3 个。这带来两个根本缺陷：

- **无法并行**：100 个词必须串行处理 100 步，GPU 的大规模并行能力被浪费
- **长程遗忘**：读到第 100 个词时，第 1 个词的信息经过 99 步传递已经极度衰减（**梯度消失**——反向传播时每经过一层梯度就乘以一个小于 1 的系数，多层后梯度趋近于零，模型无法学习远距离依赖）

> **梯度消失**：深度网络训练时，误差信号从输出层逐层回传到输入层，每经过一层梯度就被乘以一个系数；当网络很深时，经过几十层乘法后梯度可能趋近于零，导致靠近输入的层几乎无法更新权重。RNN 的逐步传递使这个问题尤为严重。

Transformer 的解决方案：**所有位置同时计算**，用注意力机制直接建立任意两个位置之间的联系，不受距离影响。

### 1.2 核心创新：注意力机制取代循环

> 注意力机制的本质是让每个位置**直接**访问所有位置的信息，而不是像 RNN 那样一步步传递。

2017 年论文 *Attention Is All You Need* 的标题意思是"靠注意力就够了，RNN/**CNN**（Convolutional Neural Network，卷积神经网络——通过局部卷积核提取特征的网络，曾广泛用于序列和图像处理）都不需要了"——这是架构层面的替换，不是"只用注意力不加别的东西"。Transformer 仍然需要位置编码、前馈网络、归一化等组件，但信息流转的核心机制从循环变成了注意力。

---

## 2. 整体架构鸟瞰

原始 Transformer（2017 论文）由**编码器**和**解码器**两大部分组成：

```
输入序列 → [编码器 ×N层] → 编码表示
                                         ↘
目标序列 → [解码器 ×N层] → 输出概率
```

现代大模型的架构更加多样：

| 架构 | 代表模型 | 注意力模式 | 本项目对应 |
|------|---------|-----------|-----------|
| **Encoder-Only** | BERT（Bidirectional Encoder Representations from Transformers，双向编码表示） | 双向，擅长理解 | ds-ocr 的 SAM ViT 编码器 |
| **Decoder-Only** | GPT（Generative Pre-trained Transformer，生成式预训练 Transformer）, LLaMA, Qwen | 因果（只看左边），擅长生成 | ds-ocr 的 MoE 解码器, qwen-asr 的解码器 |
| **Encoder-Decoder** | T5（Text-to-Text Transfer Transformer，文本到文本迁移 Transformer，将所有 **NLP**（Natural Language Processing，自然语言处理）任务统一为"输入文本→输出文本"的格式）, 原始 Transformer | 编码器双向 + 解码器因果 | ds-ocr 整体：编码器理解图像，解码器生成文字 |

**在 ds-ocr 中**：
- **SAM ViT-B**（Segment Anything Model - Vision Transformer Base，分割万物模型的视觉 Transformer 基础版） = 视觉编码器（理解图像）
- **DeepEncoder V2** = 特征编码器（深化视觉特征）
- **MoE Decoder**（Mixture of Experts Decoder，混合专家解码器） = 解码器（生成 OCR 文本）

**在 qwen-asr 中**：
- **音频编码器** = 理解音频（**Conv2D**——二维卷积层，从音频频谱图中提取局部特征，作为 Transformer encoder 的前端输入 + Transformer encoder）
- **LLM 解码器** = 生成转录文字（Qwen3 decoder）

---

## 3. 嵌入层：把文字变成数字

### 3.1 什么是嵌入？

计算机只懂数字，不懂文字。**嵌入（Embedding）** 就是把每个**词元（token）**——模型处理的最小单位，可以是一个字、一个子词或一个特殊符号——映射成一个高维向量（通常 768~4096 维的一组浮点数）。

语义相近的词，向量也相近——"猫"和"狗"的向量距离比"猫"和"桌子"近得多。这是嵌入层被训练后自然涌现的性质，而非人工设定。

> **向量（Vector）**：一组有序的数字，如 `[0.12, -0.34, 0.56, ...]`。在嵌入中，向量的每个维度没有明确的语义含义，但整体方向和距离编码了词的语义。

### 3.2 子词分词：词汇表如何覆盖所有语言？

嵌入层的词汇表存储的不是"单词"，而是**子词（subword）**——这是初学者最常见的误解。

现代 LLM 使用 BPE（Byte Pair Encoding，字节对编码）等子词分词器，词表存的是"零件"而非"成品词"：

> **BPE** 是一种数据压缩算法，在分词场景中的用法是：从单字节开始，反复将最高频的相邻字节对合并为新符号，直到词表达到预设大小。这样高频组合（如 "th"、"er"）会被合并成单一 token，低频词则保持拆分状态。

```
"unbelievable" → ["un", "believ", "able"]      # 低频词拆成已知子词
"I love 机器学习" → ["I", " love", " 机器", "学习"]  # 高频整词保留
```

- 高频整词（"the"、"机器学习"）→ 直接保留在词表中
- 低频词 → 拆成已知子词
- 完全未见的词（生僻字、新造词）→ 拆到**单字节**级别，总有对应

**多语言覆盖的原理**：GPT-2/**RoBERTa**（Robustly Optimized BERT Pretraining Approach，Facebook 对 BERT 的优化版本）等模型把文字看成**字节（bytes）**而非 **Unicode**（统一码，国际标准字符编码方案，为全球所有文字系统分配唯一编号）字符。基础表只有 256 个条目（对应 256 种字节值），通过 BPE 合并扩展。所以只要是 **UTF-8**（Unicode 的变长编码实现，用 1~4 个字节表示一个 Unicode 字符，是互联网的事实标准编码）能表示的文字——中文、日文、阿拉伯语——都能被拆解并找到对应向量。模型不是背下了所有语言的词典，而是学会了拆解任何文字的"基本粒子"。

**本项目词表规模**：

| 项目 | 词表大小 |
|------|---------|
| ds-ocr | 129,280（12.9 万） |
| qwen-asr | 151,936（15.2 万） |

### 3.3 代码实现

嵌入本质上是一个**查找表（Lookup Table）**：`[vocab_size, hidden_dim]` 的矩阵，给定 token ID 取对应行。

> **查找表**：给定一个索引（token ID），直接从矩阵中取出对应的那一行向量，不需要做任何计算。这是最简单的"把离散符号变成连续向量"的方式。

**ds-ocr**（`ds_ocr.c`）：

```c
/* 嵌入矩阵：[129280, 1280] 的 BF16 权重 */
uint16_t *tok_embeddings_bf16;  /* BF16 格式，零拷贝 mmap（内存映射，将磁盘文件直接映射到进程地址空间，无需 read 系统调用） */

/* 查表：token ID → 向量 */
const uint16_t *emb = tok_embeddings_bf16 + (size_t)token_id * hidden;
/* BF16 → F32 转换：BF16 的位模式就是 FP32 截掉低 16 位后的结果，
   所以只需左移 16 位补零即可还原 */
for (int i = 0; i < hidden; i++) {
    uint32_t f32_bits = ((uint32_t)emb[i]) << 16;
    memcpy(&output[i], &f32_bits, sizeof(float));
}
```

**qwen-asr**（`qwen_asr_decoder.c`）：

```c
/* 嵌入矩阵：[151936, 2048] 的 BF16 权重 */
uint16_t *tok_embeddings_bf16;
/* 同样的查表方式 */
```

### 3.4 权重存储格式：BF16

两个项目都将嵌入权重存为 **BF16**（Brain Float 16，Google 为深度学习设计的 16 位浮点数格式，名称源于 Google Brain 团队），通过 mmap 零拷贝映射磁盘文件，推理时按需转为 F32 计算。

**BF16 是什么？** BF16 保留 FP32 的数值范围，省掉 FP32 的一半内存。

| 格式 | 总位数 | 指数位 | 尾数位 | 表示范围 | 有效数字 |
|------|--------|--------|--------|----------|---------|
| **FP32**（单精度浮点数，IEEE 754 标准） | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 位 |
| **BF16** | **16** | **8** | **7** | **±3.4×10³⁸** | **~2.5 位** |
| **FP16**（半精度浮点数） | 16 | 5 | 10 | ±65504 | ~3.5 位 |

> **指数位**决定能表示的数值范围（多大/多小的数），**尾数位**决定精度（有效数字的位数）。

关键设计：BF16 保留了和 FP32 **一样的 8 位指数**，所以数值范围完全相同，不会出现 FP16 那种训练时动辄溢出的问题。代价是尾数从 23 位减到 7 位，精度下降——但深度学习对权重的数值精度有较强容错性，范围比精度重要。

> 为什么 LLM 偏爱 BF16？Google 最初为 **TPU**（Tensor Processing Unit，Google 专为深度学习设计的芯片）设计，现在 NVIDIA **H100**（NVIDIA 2022 年发布的旗舰 GPU，原生支持 BF16 矩阵运算）、AMD MI300 等均原生支持。训练用 BF16 混合精度，推理权重直接 BF16 存储，省一半内存且几乎不掉精度。

### 3.5 Tied Embeddings（嵌入共享）

两个项目都使用了**嵌入共享（Tied Embeddings）**：输入嵌入矩阵 = 输出投影矩阵（lm_head，即语言模型头——将隐藏状态映射回词表空间的线性层，输出每个 token 的原始分数 logits）。

> **Logits**：模型输出的未经 softmax 归一化的原始分数向量，长度等于词表大小。经过 softmax 后变为概率分布，最高概率位置对应预测的下一个 token。

```c
/* 如果没有单独的 lm_head 权重，就用嵌入矩阵做输出投影 */
if (dec->lm_head_bf16)
    ds_bf16_matvec_pub(logits, x, dec->lm_head_bf16, ...);
else
    ds_bf16_matvec_pub(logits, x, dec->tok_embeddings_bf16, ...);  /* 共享 */
```

认路（输入嵌入）和指路（输出投影）用的是同一张地图，节省约 25% 参数量（嵌入矩阵通常是模型中最大的单块参数之一，共享后直接省去一份）。

> **偏置（bias）**：线性变换 y = xW + b 中的 b 项，是一个可学习的常向量，为输出添加平移。现代大模型的注意力投影和 FFN 通常**不使用偏置**（nobias），因为 LayerNorm/RMSNorm 已经包含了可学习的偏移参数 β，额外的 bias 冗余且增加计算量。

---

## 4. 位置编码：让模型知道"谁在前面"

### 4.1 为什么需要位置编码？

自注意力机制对位置**完全无感**——"猫追老鼠"和"老鼠追猫"对它来说输入相同、输出也相同。位置编码就是给每个位置注入位置信息，让模型区分先后顺序。

### 4.2 正弦位置编码（原始 Transformer、SAM ViT）

用 sin/cos 函数生成固定位置向量，加到嵌入向量上：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

不同维度使用不同频率的正弦波——低维变化快（区分相邻位置），高维变化慢（区分远距离位置）。

**在 ds-ocr SAM 中**（`ds_visual_tokenizer.c`）：

```c
/* SAM 的位置嵌入是预训练好的固定向量，直接从权重文件加载 */
float *sam_pos_embed;  /* [577, 768] — 1个CLS（分类 token，ViT 在所有 patch 前添加的特殊 token，用于汇总全局信息） + 576个patch */
```

### 4.3 RoPE 旋转位置编码（现代大模型主流选择）

RoPE（Rotary Position Embedding，旋转位置编码，Su et al. 2021）是当前主流方案，**LLaMA**（Large Language Model Meta AI，Meta 开源的系列大模型）、DeepSeek、Qwen 等均采用。与正弦编码不同，RoPE 不把位置信息加到向量上，而是**把向量本身按位置旋转**，让注意力计算自动包含相对位置信息。

#### 核心原理

对 Q 和 K 向量的每对相邻维度施加旋转变换：

```
[q₁, q₂] → [q₁·cos(θ) - q₂·sin(θ),  q₁·sin(θ) + q₂·cos(θ)]
```

其中旋转角度 θ = pos × ωᵢ，频率 ωᵢ = 1 / 10000^(2i/d)。

> **点积（Dot Product）**：两个向量的对应元素相乘后求和，结果是一个标量。点积越大，两个向量越"方向一致"（越相关）；点积为零，方向正交（无关）；点积为负，方向相反。在注意力机制中，点积衡量 Q 和 K 的匹配程度。

**关键性质**：经过旋转后，位置 m 的 Q 向量和位置 n 的 K 向量的点积，**只取决于它们的相对位置 m-n**：

```
Qₘ · Kₙ = f(m - n)
```

这意味着模型天然地感知相对距离，而非绝对位置——位置 5 和位置 10 的关系，等价于位置 100 和位置 105 的关系。

#### 多频率旋转

高维向量（如 128 维）拆成 64 对，每对独立旋转，但**频率不同**：

- 前几对（维度 i 小）：ω 接近 1，旋转快 → 区分相邻位置的细粒度关系
- 后几对（维度 i 大）：ω 接近 0，旋转慢 → 区分远距离位置的粗粒度关系

这类似于信号处理中的多尺度分析——低频捕捉全局位置，高频捕捉局部位置。

#### RoPE vs 正弦编码

| 对比 | 正弦编码 | RoPE |
|------|---------|------|
| 注入方式 | 加到嵌入向量上 | 旋转 Q/K 向量 |
| 位置类型 | 绝对位置 | 相对位置（点积只依赖 m-n） |
| 外推能力 | 弱（超出训练长度效果差） | 强（可外推到更长序列） |
| 额外参数 | 无（固定公式） | 无（固定公式） |

> **外推能力（Length Extrapolation）**：模型能否处理比训练时更长的序列。正弦编码的位置向量在超出训练范围后，模型从未见过这些位置的编码，表现变差。RoPE 由于编码相对距离，新位置只要相对距离在训练范围内，模型仍能正常工作。

#### 代码实现

**ds-ocr 解码器**（`ds_moe_decoder.c`）：

```c
/* 预计算 RoPE 缓存 */
for (int i = 0; i < max_seq; i++) positions[i] = i;
ds_compute_rope_neox(rope_cache_cos, rope_cache_sin,
                      positions, max_seq, head_dim, rope_theta);

/* 推理时应用 RoPE */
ds_apply_rope_neox(q, cos_vals, sin_vals, 1, n_heads, head_dim);
ds_apply_rope_neox(k, cos_vals, sin_vals, 1, n_kv_heads, head_dim);
```

**qwen-asr 解码器**（`qwen_asr_decoder.c`）：

```c
qwen_compute_rope_neox(cos_out, sin_out, positions, seq, head_dim, theta);
qwen_apply_rope_neox(q, rope_cos, rope_sin, seq, n_heads, head_dim);
```

> **两个项目的 RoPE base θ 不同**：ds-ocr 解码器用 10000.0，qwen-asr 解码器用 1,000,000.0（更大的 θ 让旋转更慢，支持更长的音频序列）。

### 4.4 2D 位置编码（视觉模型专用）

SAM 处理图像时，每个 **patch**（图像块，ViT 将整张图切成固定大小的小方块，如 16×16 像素一块，每块对应一个 token）有**行列坐标**，需要 2D 位置编码：

```c
/* ds_visual_tokenizer.c */
void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols, int embed_dim);
```

---

## 5. 自注意力机制：模型的核心引擎

### 5.1 注意力与 Transformer 的关系

注意力机制是 Transformer 的**核心零件**，但不是 Transformer 的全部。

- **注意力机制**：一种"选择性聚焦"的计算方法——让模型关注最相关的信息。2014 年就被用于机器翻译，不是 Transformer 发明的
- **Transformer**：一整套架构——把注意力作为核心，搭配位置编码、FFN、归一化等组件组成完整的序列处理系统

2017 论文标题 *Attention Is All You Need* 的意思是"靠注意力就能取代 RNN/CNN"，而不是"只需要注意力不加别的"。

### 5.2 自注意力的计算过程

自注意力（Self-Attention）让序列中的每个位置与所有位置交互：

1. 每个 token 通过线性变换产生三个向量：
   - **Query（Q）**："我在找什么信息？"
   - **Key（K）**："我有什么特征可以被匹配？"
   - **Value（V）**："我携带的实际内容"
2. Q 和 K 的点积衡量匹配度，**softmax** 归一化后得到注意力权重
3. 用权重对 V 做加权求和，得到每个位置的新表示

> **Softmax**：将一组任意实数转化为概率分布（所有值在 0~1 之间，总和为 1）的函数。公式为 softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ。值越大的元素，softmax 后占比越大，从而实现"选择性聚焦"。

### 5.3 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

分解步骤：
1. **打分**：$S = QK^T$（每个 query 和每个 key 做点积，得到注意力分数矩阵）
2. **缩放**：$S = S / \sqrt{d_k}$（防止数值过大导致 softmax 梯度消失）
3. **归一化**：$A = \text{softmax}(S)$（分数转为概率分布，每行和为 1）
4. **聚合**：$O = AV$（用注意力权重对 Value 做加权求和）

> **缩放的必要性**：当维度 $d_k$ 较大时，QK^T 的值会相应增大（直观理解：点积是 $d_k$ 个乘积项的求和，维度越高项越多，结果的绝对值越大），把 softmax 推入梯度极小的饱和区（输入很大时 softmax 趋近 one-hot，梯度接近零）。除以 $\sqrt{d_k}$ 使点积的**方差**（衡量数值波动幅度的统计量，方差大意味着数值波动剧烈）维持在 1 附近，保证梯度有效传播。

### 5.4 代码实现

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

> **因果掩码（Causal Mask）**：解码器生成文字时，不能"偷看"未来的词。因果掩码把未来位置的注意力分数设为 -∞（softmax 后变为 0），确保每个位置只能关注自身及之前的位置。

### 5.5 QKV 投影

输入向量 x 通过三个**线性变换**（即矩阵乘法，也叫线性层/全连接层）得到 Q、K、V：

> **线性变换**：y = xW + b，其中 W 是权重矩阵，b 是偏置向量。它对输入做仿射变换（旋转+缩放+平移），是神经网络最基本的操作。Transformer 中多数线性变换不使用偏置（nobias）。

```
Q = x @ Wq^T    (Wq: [n_heads × head_dim, hidden])
K = x @ Wk^T    (Wk: [n_kv_heads × head_dim, hidden])
V = x @ Wv^T    (Wv: [n_kv_heads × head_dim, hidden])
```

> **维度说明**：`head_dim` 是每个注意力头独立计算的向量长度（如 128），`hidden` 是模型内部表示的总维度（如 1280），`n_heads × head_dim = hidden`（所有头的输出拼接后恢复到 hidden 维度）。

**单 token 推理中的 kernel fusion**（`ds_moe_decoder.c`）：

```c
/* QKV 融合计算：一次遍历权重同时算 Q、K、V，减少内存访问 */
ds_linear_nobias_bf16_qkv(q, k, v, x_norm,
                            layer->wq_weight_bf16,
                            layer->wk_weight_bf16,
                            layer->wv_weight_bf16,
                            hidden, q_dim, kv_dim);
```

---

## 6. 多头注意力：多个视角看世界

### 6.1 为什么需要多头？

单头注意力只学一种关注模式。多头注意力让模型**同时从多个子空间**关注不同的关系——某个头可能关注语法依赖（主语→谓语），另一个头关注指代关系（"它"→"猫"），还有一个关注位置邻近关系。每个头独立计算注意力，结果拼接后做输出投影，融合多角度信息。

### 6.2 计算方式

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) @ Wo

其中 headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

每个头有独立的 Q/K/V 投影矩阵，注意力结果拼接后通过输出投影矩阵 Wo 融合。

### 6.3 MHA → GQA 的演进

**自回归推理**（Autoregressive Inference）时，模型逐个生成 token——每一步的输出取决于之前所有已生成的 token。这意味着每生成一个新 token，都需要读取之前所有位置的 K、V 向量。这些缓存的 K、V 向量就是 **KV Cache**，其大小与序列长度和 K/V 头数成正比，当序列很长时成为内存瓶颈。

| 类型 | Q 头数 | K/V 头数 | KV 缓存 | 代表模型 |
|------|-------|---------|---------|---------|
| **MHA**（多头注意力） | h | h | 大 | 原始 Transformer, ds-ocr 解码器 |
| **GQA**（分组查询注意力，Grouped-Query Attention） | h | h/g | 中 | **LLaMA-2**（Meta 的开源大模型第二代，首个大规模采用 GQA 的主流模型）, qwen-asr 解码器, ds-ocr V2 编码器 |
| **MQA**（多查询注意力，Multi-Query Attention） | h | 1 | 小 | PaLM（Pathways Language Model，Google 的 540B 参数大模型） |

- **MHA**：每个 Q 头有独立的 K、V 头，表达能力最强但 KV 缓存最大
- **GQA**：g 个 Q 头共享一组 K、V 头。既保留多头的多样性，又减少 KV 缓存——是 MHA 和 MQA 的折中
- **MQA**：所有 Q 头共享一组 K、V。最省内存，但表达能力稍弱

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

### 7.1 注意力负责交互，FFN 负责变换

注意力机制让 token 之间交换信息，但交换完之后，每个 token 自身还需要做**非线性变换**（**激活函数**引入的非线性是关键——纯线性变换的堆叠等价于一个线性变换，必须加入非线性才能让网络学习复杂模式）来提炼和扩展自己的表示——这就是 **FFN（Feed-Forward Network，前馈网络）** 的工作。

每个位置独立计算 FFN，不涉及 token 间交互（这与注意力恰好互补）。

### 7.2 标准前馈网络

```
FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
```

两层线性变换（W₁ 将维度从 hidden 扩展到 4×hidden，W₂ 再缩回 hidden），中间夹一个**激活函数**。

> **激活函数**：对输入做非线性映射的函数。如果没有激活函数，多层线性变换的叠加仍然等价于一个线性变换，网络就无法学习复杂模式。常见激活函数：**ReLU**(x) = max(0, x)，**GELU**(x) = x · Φ(x)（高斯误差线性单元，比 ReLU 在零点附近更平滑），**SiLU**(x) = x · σ(x)（Sigmoid 线性单元，也叫 Swish）。

> 其中 **σ(x)** 是 Sigmoid 函数：σ(x) = 1 / (1 + e⁻ˣ)，将任意实数映射到 (0, 1) 区间，常用来产生"门控"概率。**Φ(x)** 是标准正态分布的累积分布函数。

SAM ViT 和 qwen-asr 音频编码器使用 GELU FFN 这种形式。

### 7.3 SwiGLU：现代大模型的选择

LLaMA、DeepSeek、Qwen 等现代模型用 **SwiGLU**（Swish-Gated Linear Unit，基于 Swish 门控的线性单元）替代传统 FFN：

> **GLU（Gated Linear Unit，门控线性单元）**：一种结构而非单一函数，公式为 GLU(x) = (x @ W₁) ⊙ σ(x @ W₂)，其中一条路径的输出经过 Sigmoid 后作为"门"，控制另一条路径的信息通过量。SwiGLU 是 GLU 的变体，用 SiLU 替代 Sigmoid 做门控。

```
SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

- `SiLU(x) = x · σ(x)`（自门控激活函数，也叫 Swish）——在零点附近平滑过渡，不像 ReLU 那样硬截断
- `⊙` 是逐元素乘法
- 三个权重矩阵（gate、up、down）替代了原来的两个

**为什么 SwiGLU 比 ReLU/GELU FFN 更好？**

传统 FFN 的激活函数是"先投影、再激活、再投影"的串行过程。SwiGLU 引入了**门控机制**：gate 路径的 SiLU 输出充当"阀门"，决定 up 路径的哪些信息应该通过。这种结构让 FFN 具备了**选择性传递**的能力——重要的特征被放大，不相关的被抑制——比单纯的非线性激活表达能力更强。

> 参考来源：Shazeer — GLU Variants Improve Transformer (2020)，证明了 GLU 门控结构在固定参数量下一致优于非门控的激活函数。

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

> **优化技巧**：qwen-asr 将 gate 和 up 权重**预融合**为一个矩阵（行交错排列），推理时只需一次矩阵乘法，然后 `swiglu_multiply` 一步完成 SiLU + 逐元素乘法。这种 **kernel fusion**（算子融合，将多个连续计算步骤合并为一个，减少中间结果的内存读写）大大减少了内存访问次数。

### 7.4 编码器中的 FFN

| 编码器 | FFN 类型 |
|-------|---------|
| SAM ViT | GELU FFN，有 bias |
| qwen-asr 音频编码器 | GELU FFN，有 bias |
| DeepEncoder V2 | SwiGLU（与解码器一致） |

---

## 8. 层归一化与残差连接：稳定训练的秘诀

### 8.1 为什么需要归一化和残差？

Transformer 每一层的运算（注意力、FFN）会改变激活值的分布——层数一多，数值可能越来越大或越来越小（**内部协变量偏移**——每一层面对的输入分布都在变化，导致后续层需要不断适应新的分布，训练变得不稳定），导致训练不稳定。**归一化**把每层的输入拉回合理的数值范围。

**残差连接**则让梯度在深层网络中有"捷径"可走——即使某层学不到有用的变换，梯度也能直接流过，避免梯度消失。

两者配合，是深层网络能成功训练的基础。

### 8.2 LayerNorm vs RMSNorm

| 归一化方式 | 公式 | 有 bias | 用在哪 |
|-----------|------|---------|--------|
| **LayerNorm**（层归一化） | (x - μ) / √(σ² + ε) * γ + β | 有 | SAM, CLIP, qwen-asr 编码器 |
| **RMSNorm**（均方根归一化） | x / RMS(x) * γ | 无 | ds-ocr 解码器, qwen-asr 解码器 |

> **μ** 是均值，**σ²** 是方差，**ε** 是防止除零的小常数（如 1e-5），**γ** 和 **β** 是可学习的缩放和偏移参数。**RMS**（Root Mean Square，均方根）= √(mean(x²))，RMSNorm 只用 RMS 做缩放，不做均值中心化。

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

### 8.3 Pre-Norm vs Post-Norm

```
Pre-Norm (现代模型):    x → Norm → Attn → +x → Norm → FFN → +x
Post-Norm (原始论文):   x → Attn → +x → Norm → FFN → +x → Norm
```

现代大模型几乎都使用 **Pre-Norm**（先归一化再计算），因为它训练更稳定——归一化后的输入数值范围可控，注意力分数不会爆炸。

**在 ds-ocr 解码器中**（`ds_moe_decoder.c`）：

```c
/* Pre-Norm 模式 */
ds_rms_norm(x_norm, x, layer->input_norm, ...);        /* 先归一化 */
ds_linear_nobias_bf16_qkv(q, k, v, x_norm, ...);       /* 再计算注意力 */
/* ... 注意力计算 ... */
for (int i = 0; i < hidden; i++) out[i] = x[i] + proj_out[i];  /* 残差连接 */
```

### 8.4 残差连接

残差连接把**输入直接加到输出上**：

```
output = x + SubLayer(x)
```

如果某层学到的变换 SubLayer(x) ≈ 0，输出就等于输入，梯度通过加法直接回传——深度网络中的"信息保底"机制。这就是为什么几十层的 Transformer 也能稳定训练。

### 8.5 Per-head Q/K RMSNorm（DeepSeek 特色）

DeepSeek-V2 引入了对 Q 和 K **每个头单独做 RMSNorm** 的做法，防止注意力分数爆炸。

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

编码器和解码器使用**同一种注意力机制**，但**掩码模式**不同，导致分工完全不同：

| | 编码器 | 解码器 |
|---|-------|-------|
| **注意力模式** | 双向（每个位置看所有位置） | 因果（每个位置只看自身及之前） |
| **核心任务** | 理解完整输入 | 自回归生成输出 |
| **典型模型** | BERT | GPT |

- **编码器用双向**：理解任务需要上下文——"我今天吃了一个__"要猜出"苹果"，必须同时看到前后文
- **解码器用因果**：生成任务是逐字输出——写第 5 个字时，第 6 个字还不存在，不能偷看

### 9.1 编码器：双向注意力

所有 token 可以互相看到：

```
token 1: 可以看 [1,2,3,4,5]  ← 全部
token 2: 可以看 [1,2,3,4,5]  ← 全部
...
```

**典型应用**：SAM ViT（图像理解）、CLIP（Contrastive Language-Image Pre-training，对比语言-图像预训练，将图像和文本映射到同一向量空间）、qwen-asr 音频编码器

### 9.2 解码器：因果注意力

只能看到当前和之前的 token：

```
token 1: 只能看 [1]          ← 自己
token 2: 只能看 [1,2]        ← 自己和之前
token 3: 只能看 [1,2,3]      ← 自己和之前
...
```

通过**因果掩码**（下三角矩阵）实现——未来位置的注意力分数设为 -∞，softmax 后变 0。

**典型应用**：ds-ocr MoE 解码器、qwen-asr LLM 解码器

### 9.3 混合注意力（DeepEncoder V2 特色）

ds-ocr 的 DeepEncoder V2 引入了**混合注意力**：视觉 token 用双向注意力，**因果流查询（Causal Flow Queries）**——一组按因果顺序排列的可学习查询向量，每个查询只能关注自身之前的视觉 token——用因果注意力。

```c
/* ds_kernels.c */
void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale);
```

视觉 token 之间充分交流（双向），因果流查询按顺序"收集"信息（因果），实现从并行理解到序列生成的过渡。

### 9.4 本项目中的编码器-解码器分工

| 组件 | 角色 | 注意力模式 | 代码文件 |
|------|------|-----------|---------|
| SAM ViT | 理解图像 | 双向 | `ds_visual_tokenizer.c` |
| DeepEncoder V2 | 深化视觉特征 | 混合（视觉双向+流查询因果） | `ds_deep_encoder.c` |
| MoE Decoder | 生成 OCR 文字 | 因果 | `ds_moe_decoder.c` |

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
| 线性变换(BF16) | `ds_kernels.c` | `ds_linear_nobias_bf16`（单矩阵乘法）, `ds_linear_nobias_bf16_qkv`（QKV 三矩阵融合乘法） |
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

1. **嵌入**：把 token 变成向量（查表），子词分词覆盖所有语言，BF16 存储省内存
2. **位置编码**：注入位置信息（正弦编码 / RoPE 旋转编码），RoPE 让注意力自动感知相对位置
3. **自注意力**：token 之间交换信息（QKV → 点积 → softmax → 加权聚合）
4. **多头注意力**：多个子空间并行关注不同关系，GQA 减少 KV 缓存
5. **FFN**：逐位置非线性变换，SwiGLU 的门控机制选择性传递信息
6. **归一化**：稳定训练（RMSNorm/LayerNorm），Pre-Norm 更稳定
7. **残差连接**：梯度捷径（x + SubLayer(x)），深度网络能训练的基础
8. **因果掩码**：解码器生成时只看过去，编码器理解时看全部

**下一步**：在 [02-MoE混合专家篇](./02-MoE混合专家篇.md) 中，我们将深入 MoE 架构——看 ds-ocr 的 64 个专家如何分工合作。