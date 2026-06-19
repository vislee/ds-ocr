# 02 - MoE 混合专家篇：让大模型"博采众长"

> **阅读目标**：读完本篇，你将理解 MoE（Mixture of Experts）的原理、路由机制、专家结构，以及 ds-ocr 中 64 个路由专家 + 2 个共享专家如何协同工作，并理解 MoE 与稠密 FFN 的关键区别。

---

## 目录

1. [为什么需要 MoE？](#1-为什么需要-moe)
2. [MoE 的核心思想：分而治之](#2-moe-的核心思想分而治之)
3. [路由机制：谁来决定选哪个专家？](#3-路由机制谁来决定选哪个专家)
4. [专家结构：每个专家长什么样？](#4-专家结构每个专家长什么样)
5. [共享专家：永远在线的"公共知识"](#5-共享专家永远在线的公共知识)
6. [ds-ocr 中的 MoE 解码器详解](#6-ds-ocr-中的-moe-解码器详解)
7. [MoE vs 稠密 FFN：参数量与计算量的权衡](#7-moe-vs-稠密-ffn参数量与计算量的权衡)
8. [MoE 推理的工程挑战与优化](#8-moe-推理的工程挑战与优化)
9. [在代码中追踪一次完整的 MoE 前向传播](#9-在代码中追踪一次完整的-moe-前向传播)

---

## 1. 为什么需要 MoE？

### 1.1 稠密模型的困境

传统 Transformer 是**稠密模型**（Dense Model）：每个 token 都经过同一个 FFN，**所有参数都被激活**。

问题：
- **参数量↑ = 计算量↑**：想让模型更聪明，就要加参数，但推理也更慢更贵
- GPT-3 有 1750 亿参数，但生成一个 token 要"动用"全部 1750 亿

### 1.2 MoE 的解法：**大模型，小计算**

MoE 的核心理念：**模型可以很大，但每次推理只激活一小部分参数**。

类比：
- 稠密模型 = 一个什么都自己干的全能员工
- MoE = 一个公司有 64 个专家，每次只派 6 个最合适的去干活

这样，模型总参数量可以很大（容量大、知识广），但每次推理的计算量只需激活 top-K 个专家（速度快、成本低）。

---

## 2. MoE 的核心思想：分而治之

### 2.1 从稠密 FFN 到稀疏 MoE

标准 Transformer 层的 FFN 是一个"一刀切"的网络——所有输入用同一组参数处理：

```
FFN(x) = SwiGLU(x)   // 一个 FFN 处理所有输入
```

MoE 将 FFN 替换为**多个并行的专家网络**，通过路由器选择激活哪些专家：

```
MoE(x) = Σₖ wₖ · Expertₖ(x)   // 只激活 top-K 个专家
```

### 2.2 图解 MoE 层

```
输入 x ──→ [路由器 Gate] ──→ 得分 [64个] ──→ Top-K 选择 (K=6)
                │                              │
                ├── Expert 0  ←── w₀ ──┐
                ├── Expert 1  ←── w₁ ──┤
                ├── ...                 ├── 只激活6个
                ├── Expert 63 ←── w₆₃ ─┘
                │
                └── Shared Expert 0, 1  ←── 永远激活
                        │
                        ↓
            路由专家加权求和 + 共享专家输出 = MoE输出
```

---

## 3. 路由机制：谁来决定选哪个专家？

### 3.1 路由器（Gate/Router）

路由器是一个**轻量级的线性层**：

```
scores = x @ W_gate^T    // W_gate: [n_experts, hidden] → scores: [n_experts]
```

然后对得分做 **softmax** 归一化，再选 **top-K**。

### 3.2 Top-K 选择

```python
# 伪代码
scores = gate(x)                  # [64] 每个专家的得分
probs = softmax(scores)           # [64] 归一化概率
top_k_probs, top_k_indices = topk(probs, k=6)  # 选6个
top_k_probs = top_k_probs / sum(top_k_probs)    # 重新归一化
```

**最终输出**：

```
output = Σ (top_k_probs[k] * Expert[top_k_indices[k]](x))
```

### 3.3 在 ds-ocr 代码中的实现

**路由得分计算**（`ds_kernels.c` → `ds_moe_router_bf16`）：

```c
void ds_moe_router_bf16(float *scores, const float *x, const uint16_t *gate_weight_bf16,
                          int hidden, int n_experts) {
    /* x: [hidden], gate_weight_bf16: [n_experts, hidden]
     * scores: [n_experts] = gate @ x (每个专家的得分) */
    ds_linear_nobias_bf16(scores, x, gate_weight_bf16, 1, hidden, n_experts);
    ds_softmax(scores, 1, n_experts);  /* softmax 归一化 */
}
```

**Top-K 选择**（`ds_kernels.c` → `ds_moe_top_k`）：

```c
void ds_moe_top_k(int *top_indices, float *top_weights, const float *scores,
                  int n_experts, int top_k) {
    /* 从 n_experts 个得分中选出 top_k 个最高的
     * 返回: top_indices[0..top_k-1] = 被选中的专家ID
     *       top_weights[0..top_k-1] = 归一化权重 */
}
```

### 3.4 为什么路由权重用 BF16？

在 ds-ocr 中，路由器的权重也存储为 BF16：

```c
/* 使用 BF16 gate weight 来匹配 Python 的 BF16 精度
 * F32 gate weight 可能选出与 Python 不同的专家，
 * 导致长序列准确度下降 */
ds_moe_router_bf16(scores, x, layer->gate_weight_bf16, hidden, n_experts);
```

> **精度细节**：BF16 的精度只有 F32 的 1/256，但对于路由决策来说，通常足够了。关键是要与训练/推理框架（Python PyTorch）保持一致，否则选出的专家不同，输出会逐渐偏移。

---

## 4. 专家结构：每个专家长什么样？

### 4.1 专家 = 小型 SwiGLU FFN

每个专家就是一个**独立的 SwiGLU FFN**，结构完全相同，但参数不同：

```
Expert(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

| 参数 | 维度 | 说明 |
|------|------|------|
| W_gate | [moe_inter, hidden] = [896, 1280] | 门控投影 |
| W_up | [moe_inter, hidden] = [896, 1280] | 上投影 |
| W_down | [hidden, moe_inter] = [1280, 896] | 下投影 |

### 4.2 为什么专家比稠密 FFN 小？

ds-ocr 中：
- **稠密 FFN**（layer 0）：intermediate = 6848
- **每个 MoE 专家**：moe_inter = 896

单个专家只有稠密 FFN 的 ~13%，但 6 个专家加起来 = 6 × 896 = 5376，接近稠密 FFN。再加上 2 个共享专家（2 × 896 = 1792），总激活参数 = 5376 + 1792 = 7168，比稠密 FFN 略大。

> **关键洞察**：MoE 的优势不是减少单次计算量，而是**增加模型总参数量**（64 个专家 = 64 × 896 = 57344 的容量），同时**每次只激活一小部分**（6 × 896 = 5376）。

### 4.3 在代码中的专家结构

**ds-ocr 中每个专家的权重定义**（`ds_ocr.h`）：

```c
struct {
    uint16_t *gate_weight_bf16;    /* [896, 1280] */
    uint16_t *up_weight_bf16;      /* [896, 1280] */
    uint16_t *down_weight_bf16;    /* [1280, 896] */
} experts[64];  /* 64 个路由专家 */
```

**专家前向传播**（`ds_kernels.c` → `ds_expert_forward`）：

```c
void ds_expert_forward(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate,
                       float *gate_buf, float *up_buf,
                       float *gate_up_buf, float *hidden_buf) {
    /* 1. Gate + Up 投影 */
    ds_linear_nobias_bf16(gate_buf, x, gate_bf16, 1, hidden, intermediate);
    ds_linear_nobias_bf16(up_buf, x, up_bf16, 1, hidden, intermediate);

    /* 2. 交错排列 [g0,u0,g1,u1,...] 便于 SwiGLU 计算 */
    for (int i = 0; i < intermediate; i++) {
        gate_up_buf[2*i] = gate_buf[i];
        gate_up_buf[2*i+1] = up_buf[i];
    }

    /* 3. SwiGLU: SiLU(gate) * up */
    ds_swiglu_multiply(hidden_buf, gate_up_buf, 1, intermediate);

    /* 4. Down 投影 */
    ds_linear_nobias_bf16(out, hidden_buf, down_bf16, 1, intermediate, hidden);
}
```

---

## 5. 共享专家：永远在线的"公共知识"

### 5.1 为什么需要共享专家？

DeepSeek-V2 的创新设计：除了 64 个路由专家，还加入 **2 个共享专家**，它们**永远被激活**，不经过路由选择。

> **类比**：路由专家是"专科医生"（只在需要时调用），共享专家是"全科医生"（每次都在场，处理通用知识）。

### 5.2 共享专家的结构

共享专家和路由专家结构相同，但 2 个共享专家的权重**拼接**在一起，作为一个整体计算：

```c
/* 共享专家权重：2个专家拼接 */
uint16_t *shared_gate_weight_bf16;  /* [2*896, 1280] = [1792, 1280] */
uint16_t *shared_up_weight_bf16;    /* [2*896, 1280] = [1792, 1280] */
uint16_t *shared_down_weight_bf16;  /* [1280, 2*896] = [1280, 1792] */
```

### 5.3 在代码中的实现

**ds-ocr 中共享专家的前向传播**（`ds_moe_decoder.c` → `moe_forward`）：

```c
/* Step 5: Add shared expert outputs (always active) */
int shared_inter = n_shared * moe_inter;  /* 2 * 896 = 1792 */

/* 一次性计算 2 个共享专家的 gate 和 up */
ds_linear_nobias_bf16(shared_gate_buf, x, layer->shared_gate_weight_bf16,
                       1, hidden, shared_inter);
ds_linear_nobias_bf16(shared_up_buf, x, layer->shared_up_weight_bf16,
                       1, hidden, shared_inter);

/* SwiGLU */
ds_swiglu_multiply(shared_swiglu_buf, shared_gate_up_buf, 1, shared_inter);

/* Down 投影 */
ds_linear_nobias_bf16(shared_out_buf, shared_swiglu_buf,
                       layer->shared_down_weight_bf16, 1, shared_inter, hidden);

/* 加到路由专家的输出上 */
for (int i = 0; i < hidden; i++) {
    output[i] += shared_out_buf[i];
}
```

---

## 6. ds-ocr 中的 MoE 解码器详解

### 6.1 整体架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| dec_hidden | 1280 | 隐藏维度 |
| dec_layers | 12 | Transformer 层数 |
| dec_heads | 10 | Q 注意力头数 |
| dec_kv_heads | 10 | K/V 头数（标准 MHA） |
| dec_head_dim | 128 | 每个头的维度 |
| dec_intermediate | 6848 | 稠密 FFN 的中间维度 |
| dec_moe_inter | 896 | MoE 每个专家的中间维度 |
| dec_n_routed_experts | 64 | 路由专家数量 |
| dec_n_shared_experts | 2 | 共享专家数量 |
| dec_top_k | 6 | 每次激活的路由专家数 |
| dec_first_k_dense | 1 | 前几层用稠密 FFN |
| vocab_size | 129280 | 词表大小 |

### 6.2 层分工：稠密层 + MoE 层

```
Layer 0: 稠密 FFN (intermediate=6848)    ← 通用特征提取
Layer 1~11: MoE (64 routed + 2 shared)   ← 专业化分工
```

> **为什么第 0 层用稠密 FFN？** 第 0 层处理的是最基础的嵌入变换，不需要"专家分工"。从第 1 层开始，语义逐渐丰富，MoE 的优势才能体现。

### 6.3 参数量对比

| 组件 | 参数量 | 占比 |
|------|--------|------|
| tok_embeddings | 129280 × 1280 ≈ 165M | 最大 |
| 12 × Attention 层 | ~120M | |
| Layer 0 稠密 FFN | ~18.7M | |
| 11 × MoE 层（路由专家） | 11 × 64 × 3 × 896 × 1280 ≈ 2.4B | 主体 |
| 11 × MoE 层（共享专家） | 11 × 2 × 3 × 896 × 1280 ≈ 75M | |
| **总参数** | **~3B** | |

> 虽然总参数约 3B，但**每次推理只激活约 570M 参数**（6/64 的路由专家 + 2 共享专家 + 注意力），这就是 MoE 的威力——"3B 的知识，570M 的计算"。

---

## 7. MoE vs 稠密 FFN：参数量与计算量的权衡

### 7.1 稠密 FFN 的计算

假设 hidden=1280, intermediate=6848：

```
计算量 ≈ 2 × 1280 × 6848 = 17.5M FLOPs/token
参数量 ≈ 3 × 1280 × 6848 = 26.3M
```

### 7.2 MoE 的计算

hidden=1280, moe_inter=896, top_k=6, n_shared=2：

```
路由器: 2 × 1280 × 64 = 0.16M FLOPs
路由专家: top_k × 2 × 1280 × 896 × 3 = 6 × 5.5M = 33.2M FLOPs
共享专家: 2 × 2 × 1280 × 896 × 3 = 11.0M FLOPs
总计算量 ≈ 44.4M FLOPs/token
参数量 ≈ 64 × 3 × 1280 × 896 + 2 × 3 × 1280 × 896 = 218M + 6.8M ≈ 225M
```

### 7.3 对比总结

| | 稠密 FFN (1层) | MoE (1层) |
|--|----------------|-----------|
| **参数量** | 26.3M | 225M (8.6×) |
| **激活参数** | 26.3M (100%) | ~44M (19.6%) |
| **计算量** | 17.5M | 44.4M (2.5×) |
| **知识容量** | 低 | 高 |

> MoE 的核心权衡：**用 8.6 倍的参数量获得 8.6 倍的知识容量，但只用 2.5 倍的计算量**。参数多不代表计算多——因为大部分参数（没被选中的 58 个专家）根本没参与计算。

---

## 8. MoE 推理的工程挑战与优化

### 8.1 内存访问瓶颈

MoE 推理最大的瓶颈是**内存带宽**。每次推理要读取 top_k=6 个专家的权重，这些权重分散在内存中：

```
Expert 3: gate [896×1280] + up [896×1280] + down [1280×896]
Expert 17: gate [896×1280] + up [896×1280] + down [1280×896]
Expert 23: ...
Expert 41: ...
Expert 55: ...
Expert 60: ...
```

每个专家约 4.9MB（BF16），6 个专家约 29.4MB 需要从内存加载。

### 8.2 ds-ocr 的优化策略

#### 1. BF16 零拷贝 mmap

```c
/* 权重直接映射磁盘文件，不加载到内存 */
uint16_t *gate_weight_bf16 = safetensors_get_bf16_direct(sf, tensor);
/* 返回的是 mmap 区域内的指针，零拷贝 */
```

专家权重存储在 safetensors 文件中，通过 mmap 映射到虚拟地址空间。只有实际被访问的专家权重才会被操作系统加载到物理内存（页式调度）。

#### 2. 预分配复用缓冲区

每次 MoE 前向传播需要多个临时缓冲区。ds-ocr 在初始化时一次性分配，推理时**复用**，避免反复 malloc/free：

```c
/* 初始化时预分配 */
ctx->moe_expert_gate_buf = (float *)malloc(moe_inter * sizeof(float));
ctx->moe_expert_up_buf = (float *)malloc(moe_inter * sizeof(float));
ctx->moe_expert_gate_up_buf = (float *)malloc(2 * moe_inter * sizeof(float));
ctx->moe_expert_hidden_buf = (float *)malloc(moe_inter * sizeof(float));
ctx->moe_expert_outputs = (float *)malloc(top_k * hidden * sizeof(float));
/* 共享专家同理 */
ctx->moe_shared_gate_buf = ...;
ctx->moe_shared_up_buf = ...;
/* ... */
```

> **为什么共享缓冲区安全？** 因为 MoE 的专家是**串行处理**的（一个接一个），同一个缓冲区可以被不同专家复用。

#### 3. Prefill 批量路由

在 prefill（多 token 并行处理）阶段，路由计算可以**批量化**：

```c
/* 批量计算所有 token 的 gate 得分 [seq_len, 64] */
float *gate_scores = (float *)malloc(seq_len * n_experts * sizeof(float));
ds_linear_nobias_bf16(gate_scores, x_norm, layer->gate_weight_bf16,
                       seq_len, hidden, n_experts);

/* 每个 token 独立做 top-K 和专家前向 */
for (int s = 0; s < seq_len; s++) {
    ds_moe_top_k(top_indices, top_weights,
                 gate_scores + s * n_experts, n_experts, top_k);
    /* ... 每个 token 的专家前向 ... */
}
```

批量路由用一次 sgemm 计算所有 token 的 gate 得分，比逐 token 计算快得多。

---

## 9. 在代码中追踪一次完整的 MoE 前向传播

让我们逐步追踪 `ds_moe_decoder.c` 中 `moe_forward()` 函数的执行流程：

### Step 1: 路由得分

```c
/* 输入: x[1280] — 当前 token 的归一化后表示 */
float scores[64];  /* 64 个专家的得分 */

/* Gate: x @ W_gate^T → softmax → scores */
ds_moe_router_bf16(scores, x, layer->gate_weight_bf16, 1280, 64);
```

**结果**：得到 64 个浮点数，表示每个专家的"匹配度"。

### Step 2: Top-K 选择

```c
int top_indices[6];   /* 选中的 6 个专家 ID */
float top_weights[6]; /* 对应的归一化权重 */

ds_moe_top_k(top_indices, top_weights, scores, 64, 6);
```

**结果**：例如 `top_indices = [3, 17, 23, 41, 55, 60]`，`top_weights = [0.25, 0.20, 0.18, 0.15, 0.13, 0.09]`。

### Step 3: 路由专家前向

```c
float expert_outputs[6 * 1280];  /* 6 个专家各输出 1280 维 */

for (int k = 0; k < 6; k++) {
    int expert_id = top_indices[k];
    ds_expert_forward(
        expert_outputs + k * 1280,  /* 输出位置 */
        x,                          /* 输入 */
        layer->experts[expert_id].gate_weight_bf16,
        layer->experts[expert_id].up_weight_bf16,
        layer->experts[expert_id].down_weight_bf16,
        1280, 896,                  /* hidden, intermediate */
        gate_buf, up_buf, gate_up_buf, hidden_buf  /* 复用缓冲区 */
    );
}
```

**结果**：6 个专家各自独立处理输入 x，产生 6 个 1280 维的输出向量。

### Step 4: 路由专家加权组合

```c
float output[1280];  /* 路由专家的组合输出 */

ds_expert_combine(output, expert_outputs, top_indices, top_weights, 6, 1280);
/* 内部实现:
 * for (k = 0; k < 6; k++)
 *     for (i = 0; i < 1280; i++)
 *         output[i] += top_weights[k] * expert_outputs[k * 1280 + i];
 */
```

**结果**：6 个专家的输出按路由权重加权求和，得到路由专家的总输出。

### Step 5: 共享专家前向

```c
float shared_out[1280];

/* 2 个共享专家一起计算（权重拼接） */
ds_linear_nobias_bf16(shared_gate_buf, x, layer->shared_gate_weight_bf16,
                       1, 1280, 1792);  /* hidden → 2*896 */
ds_linear_nobias_bf16(shared_up_buf, x, layer->shared_up_weight_bf16,
                       1, 1280, 1792);
/* SwiGLU + Down */
/* ... */
ds_linear_nobias_bf16(shared_out, shared_swiglu_buf,
                       layer->shared_down_weight_bf16, 1, 1792, 1280);

/* 加到路由专家输出上 */
for (int i = 0; i < 1280; i++) {
    output[i] += shared_out[i];
}
```

**结果**：最终 MoE 输出 = 路由专家加权组合 + 共享专家输出。

### 完整数据流图

```
              输入 x [1280]
                   │
          ┌───────┼───────┐
          │       │       │
     [路由器]  [路由专家] [共享专家]
          │       │       │
   scores[64]   ×6 experts  ×2 experts
          │       │       │
     Top-K(6)  各自输出     整体输出
          │       │       │
     weights[6]  expert_   shared_
          │      outputs   out
          │       │       │
          └───→ 加权求和 ──→ + ──→ MoE输出 [1280]
```

---

## 小结

你现在理解了 MoE 架构的每一个细节：

1. **动机**：用稀疏激活实现"大模型容量、小计算量"
2. **路由器**：Gate 线性层 + softmax + top-K 选择
3. **专家**：独立的 SwiGLU FFN，结构相同参数不同
4. **共享专家**：永远激活，处理通用知识
5. **精度**：路由器也用 BF16，确保与 Python 一致
6. **优化**：BF16 mmap、缓冲区复用、批量路由

**下一步**：在 [03-推理引擎实战篇](./03-推理引擎实战篇.md) 中，我们将深入纯 C 推理引擎的实现细节——从 safetensors 加载到 kernel 优化，看两个项目如何把理论变成高效的代码。