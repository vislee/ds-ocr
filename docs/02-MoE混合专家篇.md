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
10. [MoE 的训练：负载均衡、容量因子与推理时看不见的机制](#10-moe-的训练负载均衡容量因子与推理时看不见的机制)
11. [专家并行 EP 与 all-to-all 通信](#11-专家并行-ep-与-all-to-all-通信)

---

## 1. 为什么需要 MoE？

### 1.1 稠密模型的困境

传统 Transformer 是**稠密模型**（Dense Model）：每个 token 都经过同一个 FFN，**所有参数都被激活**。

问题：
- **参数量↑ = 计算量↑**：想让模型更聪明，就要加参数，但推理也更慢更贵
- GPT-3 有 1750 亿参数，但生成一个 token 要"动用"全部 1750 亿

### 1.2 MoE 的解法：**大模型，小计算**

一个组织类比：与其雇一位全才通宵加班，不如建一座**智库**——64 位专家 + 一个选题会（路由器），每道题按题请 6 位专家会诊，另有 2 位常驻秘书（共享专家，§5）处理通用事务。容量靠"多雇人"翻倍，成本只花在"真的请来的专家"身上。

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
     * scores: [n_experts] = W_gate @ x（每个专家的原始得分）
     * 【注意】本函数只做线性层，不含 softmax——softmax 由调用方完成
     * （ds_moe_decoder.c 的 moe_forward 内联做"全体 64 专家 softmax
     *   后再选 top-6"，见 §3.5 手算） */
    ds_linear_nobias_bf16(scores, x, gate_weight_bf16, 1, hidden, n_experts);
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

> **精度细节**：BF16 的尾数只有 7 位（F32 是 23 位），可分辨的最小相对间隔是 F32 的 1/65536（$2^{16}$），但对路由决策来说通常足够了。关键是要与训练/推理框架（Python PyTorch）保持一致，否则选出的专家不同，输出会逐渐偏移。

### 3.5 逐步手算：4 专家 top-2 的一次完整路由

把 64 专家 top-6 缩成 4 专家 top-2，每个数字都能手算。设 hidden=4，当前 token 的归一化表示 x = [1.0, 0.5, −0.5, 0.0]，路由器权重 W_gate 是 [4 专家, 4 维] 的小矩阵（每行 = 一个专家的"提问方向"）：

```
W_gate = [ 2.0  0.0  0.0  0.0 ]   ← 专家 0
         [ 0.0  2.0  0.0  0.0 ]   ← 专家 1
         [ 0.0  0.0  2.0  0.0 ]   ← 专家 2
         [ 0.5  1.0  0.5  0.5 ]   ← 专家 3
```

**Step 1：线性打分**（复刻 `ds_moe_router_bf16`：scores[e] = W_gate[e]·x。注意本函数**只做线性层、不含 softmax**，softmax 由调用方完成）：

```
s₀ = 2.0×1.0                              = 2.0
s₁ = 2.0×0.5                              = 1.0
s₂ = 2.0×(−0.5)                           = −1.0
s₃ = 0.5×1.0 + 1.0×0.5 + 0.5×(−0.5) + 0.5×0 = 0.75
```

**Step 2：全体专家 softmax**（复刻 `ds_moe_decoder.c` moe_forward 的 Step 2：先减最大值 2.0 防 exp 上溢，再对**全部 4 个**专家归一化）：

```
e^(s−max) = [e⁰, e⁻¹, e⁻³, e⁻¹·²⁵] = [1.0000, 0.3679, 0.0498, 0.2865]
分母 Σ = 1.7042
p = [0.5868, 0.2159, 0.0292, 0.1681]      ← 和恰为 1
```

**Step 3：无放回 top-2**（选择法：扫一遍找最大 0.5868 → 专家 0；跳过它再扫一遍找次大 0.2159 → 专家 1，算法细节见 §3.6）：

```
top_indices = [0, 1]
```

**Step 4：路由权重——两种语义，数字不同**：

- **本仓库解码器真实路径（不 renorm）**：top_weights 直接取全局 softmax 概率 `[0.5868, 0.2159]`，两者之和 0.8027 < 1——剩下的 0.1973 概率质量属于没被选中的专家 2/3。这与官方实现的配置一致（`norm_topk_prob=False`、`routed_scaling_factor=1.0`，见 §10.5）。
- **`ds_moe_top_k`（renorm 语义）**：在 top-2 子集内重归一化 `[0.5868, 0.2159] / 0.8027 = [0.7310, 0.2690]`，和恰为 1——即 §3.2 伪代码的 `top_k_probs / sum(top_k_probs)`。

**Step 5：专家前向 + 加权合并**。设两个专家对 x 的输出为 E₀(x) = [0.4, −0.2, 0.1, 0]、E₁(x) = [0, 0.3, −0.1, 0.2]（各自是一整个 SwiGLU FFN 的结果，结构见 §4）：

```
不 renorm（本仓库真实路径）: out = 0.5868·E₀ + 0.2159·E₁
                              = [0.2347, −0.0526, 0.0371, 0.0432]
renorm 语义（ds_moe_top_k）:  out = 0.7310·E₀ + 0.2690·E₁
                              = [0.2924, −0.0655, 0.0462, 0.0538]
```

最后还要**加上共享专家的输出**（永远激活、无路由权重系数，见 §5）才得到完整 MoE 输出。

> **为什么"先全体 softmax、再 top-K"很重要？** softmax 的分母含全部 64 个专家；如果先选后归一（把未选中专家从分母里丢掉），选中专家的权重会**系统性偏大**（上例 0.7310 vs 0.5868），且与 Python 参考实现的输出不可对齐。本仓库踩过这个坑：旧版 `ds_moe_top_k` 的"先选后 softmax"正是因此被改为解码器内联的"先 softmax 后选择"（`ds_moe_decoder.c` moe_forward Step 2 的注释记录了这次修复）；`ds_moe_top_k` 本身保留 renorm 语义，供测试与未来切换使用。

### 3.6 top-k 选择的算法细节

选 top-6 听起来要排序，但 K 很小时**选择法**更划算。`ds_kernels.c` 的 `ds_moe_top_k` 与解码器内联选择（moe_forward Step 3）逻辑相同：

```
重复 K 次：
    扫描全部 n_experts 个分数，跳过已入选的，
    记下当前最大者 → 写入 top_indices[k]
```

- **复杂度**：O(K × n_experts) = 6×64 = 384 次比较，微秒级——不值得为此上堆或 partial sort（代码复杂度翻倍，只省 <1μs）
- **无放回**：内层"是否已入选"检查保证同一专家不会被选两次
- **天然有序**：按名次产出，top_indices[0] 恒为得分最高者
- **数值安全**：若在选中后再做 softmax（renorm 语义），必须先减最大值——expf(89) 就溢出为 Inf（`ds_softmax` 的注释演示过这对 logits 90/89 的反例）

> **工程取舍**：64×6 的规模下，"简单且能与 Python 逐位对齐"比"渐进最优"更重要——路由是每 token 每层都要跑的热路径，与 Python 的输出对齐（CER 回归为 0）优先于微小的性能收益。

> 🗣️ **说人话**：先给"全班 64 人"投票定占比，再挑前 6 名上台——Gate 就是个打分器：64 个专家各给当前 token 打一个分（§3.1）。softmax 把这 64 个分转成"占比"（全为正、加起来恰好 1）。最容易踩的坑是**先挑 6 人、再拿这 6 人的分算占比**：那样分母只剩 6 个人，人人占比被抬高——手算例里专家 0 的权重会从 0.5868 被吹到 0.7310（§3.5）。正确顺序是**先让 64 人都投票定占比，再取占比最高的 6 个**，分母永远是全班。还有 Gate 权重也用 BF16：不是因为它准，而是因为**要和 Python 选出同一批专家**——差一位数换成别的专家，输出就一层层偏下去（§3.4）。
> *边界*：先全班归一还是先选后归一取决于 checkpoint 配置——本仓库 `norm_topk_prob=False`，所以真实路径取全局概率、六者之和 < 1（§10.5）；换成更深模型的配置结果又不一样。

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

> 🗣️ **说人话**：共享专家就是给这个团队留两个"每单都在场"的常识岗。路由专家像派单的专才——这张单恰好分给谁，谁才出手；但有些事是**所有单子都绕不开**的通用常识（基本语法、常见排版），要是也靠路由，一不小心就被拆散到几十个专家那里、每个都学得一知半解。于是在每层固定摆两个共享专家，**不管来什么 token，它俩都跟着算一份**，最后把结果直接加到输出上（上面那段 `output[i] += shared_out_buf[i]`）。专才负责差异化，这两个常识岗负责兜底，模型就不至于为了通用知识再去挤占路由专家的名额。
> *边界*：共享专家不参与路由、永远激活，意味着**它的参数量在每次前向里都被全额支付**——它可以不患“不被选中”的毛病，却给每一层的下限计算量加了固定一笔，专家数越多这笔越大；本仓库是 2 个共享专家、inter 896，所以每层固定多算 2×896 那一块（§6）。

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
| 12 × Attention 层 | 12 × 4×1280×1280 ≈ 78.6M | |
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

> 🗣️ **说人话**：671B 的总账，每次只掏 37B 的现金——把参数想成一支军队：DeepSeek-V3 报了 671B 人的编制（知识储备就这么多），但打一场仗（生成一个 token）只出动 37B 人上场，其余 634B 在营房睡觉。37/671 ≈ 5.5%，等于每 18 个参数里只有 1 个真的参与计算。所以"参数多"不等于"算得慢"——慢不慢看的是**上场人数**，不是**报名人数**。本仓库同理：2.8B 编制，每 token 只激活 574M（§7.4 那个官方名里的 A570M）。
> *边界*：账不能只算 FFN——本仓库的 574M 激活口径把 12 层注意力和 165M 的 lm_head 也算进去了（§7.4），稠密那部分不会因为上了 MoE 而省。

### 7.4 严格实算：用 `ds_ocr.h` 的配置把账算清

§7.1–7.3 是快算（且口径不一：17.5M 只算了一条投影）。现在统一口径，用 `ds_ocr.h` 的真实配置实算：hidden=1280、dense inter=6848、moe_inter=896、64 路由专家、2 共享专家、top-6、12 层（第 0 层稠密）、词表 129280。

**单组件参数**（一个专家 = gate/up/down 三块 [896,1280] 的 BF16 矩阵）：

| 组件 | 公式 | 参数量 | 权重字节（BF16） |
|------|------|--------|----------------|
| 单个专家 | 3×1280×896 | 3.44M | 6.9 MB |
| 64 个路由专家 / 层 | 64 × 3.44M | 220.2M | 441 MB |
| 2 个共享专家 / 层 | 2 × 3.44M | 6.88M | 13.8 MB |
| 路由器 / 层 | 64×1280 | 0.082M | 0.16 MB |
| 注意力 / 层（Wq/Wk/Wv/Wo 各 [1280,1280]） | 4×1280×1280 | 6.55M | 13.1 MB |
| 第 0 层稠密 FFN | 3×1280×6848 | 26.3M | 52.6 MB |

**全模型总参数**：

```
tok_embeddings（=lm_head，tied）    129280×1280         ≈ 165.5M
12 层注意力                         12×6.55M            ≈  78.6M
第 0 层稠密 FFN                     3×1280×6848         ≈  26.3M
11 层 MoE（64 路由专家）            11×220.2M           ≈ 2422.2M
11 层 MoE（2 共享专家）             11×6.88M            ≈  75.7M
11 个路由器 + RMSNorm 等零头        11×0.082M + 0.03M   ≈   0.9M
─────────────────────────────────────────────────────────────
总参数                                                   ≈ 2769M ≈ 2.8B
```

**每 token 激活参数**（一次前向真正触到的权重）：

```
注意力（12 层全跑）                 12×6.55M            ≈  78.6M
第 0 层稠密 FFN                                         ≈  26.3M
MoE 层 ×11：路由器 0.082M + 6 路由专家 20.6M + 共享 6.9M = 27.6M/层
                                                      ≈ 303.7M
嵌入查表                            只取 1 行 1280 个数  ≈ 0.001M
LM head（末次 matvec，全词表）       129280×1280         ≈ 165.5M
─────────────────────────────────────────────────────────────
每 token 激活参数                                         ≈ 574M
```

**574M ≈ 570M——模型官方名 DeepSeek3B-MoE-A570M 里的 "A570M" 就是这个数**：A = Activated，且激活口径包含末次的 lm_head matvec（tied embeddings 摊上 165.5M）。激活率 574/2769 ≈ 21%：约 1/5 的参数干活——MoE 的"大容量、小计算"由此得证。

**decode 每步的算力/带宽账**（每乘加按 2 FLOPs，权重 BF16 每参数 2 字节）：

```
算力 ≈ 注意力投影  12 层 × 4 × 2×1280×1280 = 157M
     + 第 0 层 FFN  3 × 2×1280×6848         =  53M
     + 11 层 MoE    (6+2 专家) × 3×2×1280×896 = 607M
     + LM head      2×1280×129280            = 331M
     ≈ 1.15 GFLOPs / token

带宽 ≈ 同一批权重的字节数 ≈ 1.15 GB / token（BF16）
```

算力数与字节数几乎相等并非巧合（每 MAC 读 1 个 2 字节参数、做 2 FLOPs）——decode 每步流过 ~1.15GB 权重却只做 ~1.15 GFLOPs，算术强度 ~2 FLOPs/Byte，是典型的**内存带宽瓶颈**（Roofline 分析见 docs/05）。这也解释了 §8 的优化方向为什么全是围绕"少读权重"展开。

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

每个专家约 6.9MB（BF16，gate/up/down 三块，实算见 §7.4），6 个专家约 41MB 需要从内存加载。

### 8.2 ds-ocr 的优化策略

#### 8.2.1 BF16 零拷贝 mmap

```c
/* 权重直接映射磁盘文件，不加载到内存 */
uint16_t *gate_weight_bf16 = safetensors_get_bf16_direct(sf, tensor);
/* 返回的是 mmap 区域内的指针，零拷贝 */
```

专家权重存储在 safetensors 文件中，通过 mmap 映射到虚拟地址空间。只有实际被访问的专家权重才会被操作系统加载到物理内存（页式调度）。

#### 8.2.2 预分配复用缓冲区

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

#### 8.2.3 Prefill 批量路由

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
float top_weights[6]; /* 对应的路由权重 */

/* 真实路径：moe_forward 内联的选择循环（对 softmax 后的概率取 top-6，
 * 权重直接取概率、不再 renorm，见 §3.5）；
 * ds_moe_top_k 是同逻辑的独立工具函数（renorm 语义），供测试使用 */
ds_moe_top_k(top_indices, top_weights, scores, 64, 6);
```

**结果**：例如 `top_indices = [3, 17, 23, 41, 55, 60]`，`top_weights = [0.25, 0.20, 0.18, 0.15, 0.13, 0.09]`（真实路径下六个权重之和略小于 1，剩余概率质量属于未选中的专家）。

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

## 10. MoE 的训练：负载均衡、容量因子与推理时看不见的机制

> **一句话导读**：本仓库推理代码里找不到 aux loss、z-loss、容量因子——它们全是**训练时**的约束与正则；但理解它们，才能理解权重里的专家为什么"各管一摊"、路由为什么长成现在这个形状。

### 10.1 核心矛盾：路由是"选"出来的，损失却要可导

MoE 训练有一个内建矛盾：

- 前向里的 **top-K 是硬选择**（argmax 家族，不可导）——token 要么走专家 3，要么不走，没有"走 0.7 步专家 3"
- 但梯度下降需要损失处处可导

训练期 MoE 的解法：**让梯度穿过路由概率 p，而不穿过选择本身**。对被选中的专家 k，输出 ∝ p_k · Expert_k(x)——Expert_k(x) 的梯度正常回传，p_k 也可导（它是 logits 的 softmax）。"选哪 K 个"这个离散决策不直接接收梯度，它被 softmax 概率间接引导：概率大的专家更可能被选。

这套机制能跑，但立刻引出三个训练期才有的问题——**路由塌缩、logit 膨胀、token 溢出**，分别对应三个训练机制：aux loss、z-loss、capacity factor。

### 10.2 负载均衡辅助损失（aux loss）：f_i 与 P_i 的完整推导

**问题（路由塌缩，router collapse）**：路由是"赢者通吃"的。训练初期某个专家偶然得分略高 → 更多 token 流向它 → 它被更频繁地训练 → 得分更高……正反馈滚雪球，最后 64 个专家只剩三五个在干活，其余因得不到梯度而废掉——MoE 退化成一个小稠密模型，参数全浪费。

**解决**：在任务损失之外加一个**负载均衡辅助损失**（Switch Transformer，Fedus et al. 2021），同时度量"token 实际怎么分"和"路由器想让它们怎么分"。对第 i 个专家定义两个量（T 为 batch 内 token 数，N 为专家数，K 为 top-K）：

```
f_i = (1/(K·T)) Σ_t 1[专家 i ∈ token t 的 top-K]     ← 分配频次：i 实际被选中几次（离散，不可导）
P_i = (1/T) Σ_t p_i(t)                               ← 平均概率：路由器想分给 i 的平均 softmax 概率（连续，可导）
```

- f_i 是"事实"，P_i 是"意愿"；绝对均衡时 f_i = P_i = 1/N（Σf_i = ΣP_i = 1，f_i 除以 K·T 保证这一点）
- **辅助损失**：

```
L_aux = α · N · Σᵢ₌₁ᴺ f_i · P_i          （α 为超参，Switch 论文取 0.001~0.01）
```

**为什么是这个形状？逐项拆**：

- 均匀时每项 f_i·P_i = 1/N²，L_aux = α·N·N·(1/N²) = α——下界；某专家独大时 f_i、P_i 同时大，该项显著超过 1/N²，损失抬高——惩罚
- **乘积的巧思**：f_i 不可导，但梯度可以从 P_i 流进路由 logits。优化器要压低 f_i·P_i，最顺手的路径是把"实际很忙（f_i 大）的专家"的 P_i 压小——即直接教路由器"少把概率分给已经很忙的专家"，这恰是均衡想要的行为
- **两个因子缺一不可**：只有 f_i 不可导（梯度传不进去）；只有 P_i 会退化——路由器可以对全体专家输出均匀概率（P_i 全 1/N），但只要选择仍涌向某几个专家，f_i 依旧失衡。f_i 报告真实流量，P_i 提供可导的调节旋钮
- α 平衡两股力量：太大牺牲任务精度，太小均衡不住

**本仓库的"活化石"**：随模型发布的 Python 参考实现（`models/DeepSeek-OCR-2/modeling_deepseekv2.py`）里就有这段训练机制——`aux_loss = (Pi * fi).sum() * self.alpha`（`fi = ce * n_routed_experts` 即 f_i×N，序列级统计 `seq_aux=True`，`aux_loss_alpha=0.001`）。C 推理引擎（`ds_moe_decoder.c`）不包含它的任何部分：**训练约束在推理时消失**（§10.6）。

> 🗣️ **说人话**：负载均衡 aux loss——防止"所有客人都挤到一家店"。训练时有个病叫**路由塌缩**：没人管的话，路由器会偷懒，把几乎所有 token 都塞给同几个专家，其余几十个专家形同虚设。aux loss 就是经理盯排队长度：`f_i` 是专家 i **实际接了多少客**（真实客流，不可导），`P_i` 是路由器**想派多少客**给它（可导）。罚的是乘积 `f_i·P_i`（§10.2），效果等于对系统喊一句："已经很挤的店，别再打广告了。"均匀时损失下界是 α，谁扎堆谁抬损失。两个因子缺一不可：只有 f_i，梯度传不进去；只有 P_i，又会退化成假均匀。
> *边界*：它只管**长期统计**均衡，单步仍可能扎堆（§11.2）；而且它是**额外损失**，α 调大就伤主任务——所以 DeepSeek-V2 干脆改用 bias 旋钮（§10.5）。

### 10.3 Router z-loss：按住 logits 的"膨胀"

ST-MoE（Zoph et al. 2022）补充了第二个正则，针对路由 logits 的**绝对幅度**：

```
L_z = (1/B) Σ_b ( log Σₑ e^(x_{b,e}) )²        （B 为 batch 内 token 数，x 为路由 logits，系数典型 0.001）
```

即对每个 token 的 log-sum-exp（softmax 分母取对数）求平方再平均。logits 整体平移不改变 softmax 分布，"均值大"本身无害；但**绝对值大了**有两个实害：

1. **数值**：训练常用 BF16/FP16，eˣ 在 x 大时上溢（`ds_softmax` 的注释：expf(89) 即 Inf），路由概率变 NaN
2. **梯度**：logits 幅度大 → softmax 进入饱和区 → P_i 的梯度消失 → §10.2 的 aux loss 也跟着失效

z-loss 只压 log-sum-exp 的幅度、不碰 logits 之间的相对差（softmax 分布不变）——专治膨胀，不改变路由语义。

### 10.4 容量因子（capacity factor）与 token 溢出（drop）

上面是损失层面；还有一层**工程硬约束**。并行训练时每个专家驻留在一个设备分区上，输入张量形状必须预先确定——不能"专家 3 今天收 900 个 token、明天 17 个"。于是给每个专家设**容量**：

```
expert_capacity = (batch内token数 / N) × capacity_factor
```

- 均匀时每专家恰好分到 T/N 个；capacity_factor（典型 1.0~1.25）留余量
- **溢出即丢弃（token drop）**：超出容量的 token 不进专家，直接走残差连接（输出 ≈ 输入原样穿过），该层对它的 FFN 加工等于没做
- **两难**：capacity_factor 调大 → 溢出少但显存/算力浪费在 padding 空槽上；调小 → 省资源但丢 token 多、信息损失
- 后续工作（如 MegaBlocks 的 dropless MoE）用稀疏 kernel 绕开"定形"限制，但"按专家计数"的思想仍是训练部署的基础

> 🗣️ **说人话**：容量因子——每间教室先摆好固定座位，装不下的人只能站门口。并行训练时专家分散在不同设备上，每台机器得**提前知道**自己要收多少 token——不能说"专家 3 今天收 900 个、明天 17 个"。于是给每个专家发固定座位：`容量 = (token 数 / 64) × 因子`（§10.4）。比如一批 1024 个 token、64 个专家，均匀是每家 16 个；因子给 1.25 就摆 20 个座。哪家突然来了 50 个，多出的 30 个进不了门，只能**原样穿过**（走残差，这一层对它们等于白做）。座位摆多了空着浪费，摆少了丢人——这就是 capacity factor 的两难。
> *边界*：本仓库单机逐 token 串行，每 token 的 top-6 天然零溢出，这套约束推理时不存在（§10.6）；MegaBlocks 这类 dropless MoE 也用稀疏 kernel 绕开了"定形"限制。

### 10.5 DeepSeek 的无辅助损失负载均衡：bias 调节法

aux loss 有先天缺陷：它是**额外损失**，与任务目标博弈（α 大了伤主任务），且只能通过损失面间接施压。DeepSeek-V2/V3 换了个思路——**不用辅助损失，直接在线调节路由本身**：

```
选择时：s'_i = s_i + b_i        ← 加 bias 的分数，只用于"选谁进 top-K"
加权时：g_i = s_i               ← 真正乘在专家输出上的权重仍是原始分数（V2 的做法）
```

b_i 是每个专家一个标量，**不是梯度学出来的，是规则驱动的**：

```
每步训练结束后统计各专家负载：
  专家 i 超载（选中次数 > 均值） → b_i ← b_i − γ    （降低它的吸引力）
  专家 i 空闲                    → b_i ← b_i + γ    （提高吸引力）
```

直觉：bias 是一个"手动恒温器"。aux loss 用梯度间接施压、可能伤主任务；bias 法按负载信号直接双向旋钮，任务损失完全不被污染。DeepSeek-V3 在此之上把门控换成 sigmoid、对选中分数重归一化，再补一个极小的序列级辅助损失（α=0.0001）兜底。

**本仓库模型属于哪一派？** 看 checkpoint 的 `config.json` 与其配套的 `configuration_deepseek_v2.py` 默认值：`"topk_method": "greedy"`、`"n_group": 1`、`scoring_func="softmax"`、`norm_topk_prob=False`、`routed_scaling_factor=1.0`——即朴素"softmax 打分 → greedy top-6"，**无 bias、无分组限选、不 renorm**，正与本仓库 C 实现（`ds_moe_router_bf16` 打分 → moe_forward 内联 softmax → 选择法取 top-6，§3.5 手算的那条路径）一致。3B 级模型专家少、批次小，均衡压力远小于 236B/671B 级别，朴素路由即可。

### 10.6 为什么推理引擎不需要这些？

| 机制 | 训练时存在的原因 | 推理时为何消失 |
|------|----------------|--------------|
| aux loss（f_i·P_i） | 防路由塌缩——塌缩只发生在**学习**过程中 | 路由器已冻结，f_i、P_i 不再变化，无需惩罚 |
| z-loss | 防 logits 膨胀——膨胀是训练动态的产物 | logits 由冻结权重一次性算出，量级已定型 |
| capacity factor / drop | 并行设备的输入形状必须固定 | 本仓库单 CPU 逐 token 串行处理，各 token 的 top-6 天然零溢出 |
| bias 在线调节 | 均衡是训练期的持续过程 | 无"在线"，无负载统计可做 |

推理时剩下的只有**语义**：softmax → top-6 → 按概率加权 + 2 个共享专家（§3.5 的五步）。训练期的一切约束都化进了冻结的权重里——这是"推理 = 训练的纯前向切片"（docs/01 §10）在 MoE 上的具体体现。

---

## 11. 专家并行 EP 与 all-to-all 通信

> **一句话导读**：单机跑 MoE（本仓库）时权重就是全部家当；真正的大 MoE（DeepSeek-V3 的 671B）一台机器装不下——专家并行（Expert Parallelism, EP）把专家摊到多卡，每层靠两次 all-to-all 通信"送料"和"收货"。

### 11.1 概念图：专家切分 + 两次 all-to-all

数据并行 DP（每卡放完整模型）放不下 671B；张量并行 TP 切每个矩阵、通信太碎。EP 的切法很"粗"——**按专家维切分**：每张卡完整持有 N/E 个专家的全部参数（E 为卡数），注意力/共享专家/路由器通常每卡一份或另用 TP 切：

```
        4 卡 EP，64 专家 → 每卡 16 个专家
卡0: 专家 0~15      卡1: 专家 16~31      卡2: 专家 32~47      卡3: 专家 48~63

一个 token 在卡0 算完注意力，路由器给出 top-6 = {3, 17, 23, 41, 55, 60}
  → 专家 3 在卡0；17、23 在卡1；41 在卡2；55、60 在卡3
```

于是一个 MoE 层的执行变成三段：

```
① dispatch all-to-all：把 token 的隐状态 x 发给"它的专家所在"的每张卡
      （卡0 的 x 要送 4 份——去向由 top-6 的专家分布决定）
② 各卡本地计算：自己持有的专家处理"收到的所有 token"
      （卡1 收到若干 token × 各自 top-6 中落在本卡的专家 → 批量 SwiGLU）
③ combine all-to-all：把各专家输出 × 路由权重，加回 token 的"母卡"
      （再飞一次，方向相反）
```

### 11.2 推理部署视角的三个要点

- **通信量与 batch 的博弈**：dispatch/combine 每层各飞一次 [token × hidden] 量级的张量。batch 越大，单卡"收到的 token 数"越多，专家矩阵乘越接近计算密集，通信占比越小——**MoE 推理天生喜欢大 batch**；batch=1 的单请求逐 token 场景下 EP 纯亏（本仓库正是 batch=1 的单机形态，EP 毫无收益）
- **负载不均的部署面**：训练期的 aux loss / bias 只保证**长期统计**均衡；单步仍可能大部分 token 涌向同一批专家 → 个别卡排队、其余空转。部署侧对策：通信与共享专家计算重叠（DeepSeek-V2 的 shared expert overlap）、热点专家冗余复制
- **与 TP 的组合**：注意力段照旧可按头做 TP；EP 只管 FFN 段——现代部署常是 TP（注意力）+ EP（专家）的混合并行

### 11.3 回到本仓库：单机的 "EP" 是什么样

单 CPU 上"专家分布在哪"的答案是 safetensors 文件——64 个专家的 gate/up/down 权重按张量散布在 mmap 区域（`ds_moe_decoder.c` 的 `ds_decoder_load` 加载，`ds_ocr.h` 的 `experts[64]` 指针数组），"通信"退化为**内存/磁盘读**：

- decode 每步只读 top-6 路由专家 + 2 共享专家的权重（≈ 8×6.9MB × 11 层 ≈ 0.6GB）；mmap 按页懒加载，没被点名的 58 个专家连物理页都不进
- prefill 才会碰全部 64 个专家（所有 token 的路由并集），`ds_decoder_prefill` 里的 madvise(MADV_WILLNEED) 提前预取下一层专家 mmap——"dispatch 之前先把料备好"的单机版
- §8.1 的内存带宽瓶颈，本质就是单机版的 all-to-all 代价：把 token"送进"专家的带宽，从 NVLink 换成了内存总线

至此 §8（推理优化）、§10（训练约束）、§11（部署并行）三个视角拼在一起，就是 MoE 从训练到推理的完整生命周期。

> 🗣️ **说人话**：专家并行 EP 的 all-to-all——单机是"同楼递外卖"，多机是"跨城快递"。单机跑 MoE（本仓库）时，"专家在哪"的答案是内存：每步从 mmap 里读 top-6 + 2 个共享专家的权重，没被点名的 58 个专家连物理页都不进（§11.3）。但 671B 一台机器装不下，就得把 64 个专家摊到多卡（EP），于是每层凭空多出两趟通信：**dispatch** 把 token 送到"它的专家所在的卡"，算完再 **combine** 把结果加回来（§11.1）。这就像单机是"同一栋楼里递外卖"，多机是"跨城快递"——batch 越大，每张卡收到的 token 越多、专家矩阵乘越接近满负荷，快递费摊得越薄；batch=1 的单请求里跑 EP 纯亏（本仓库正是这种形态，所以根本用不上 EP）。
> *边界*：通信量是每层 [token × hidden] 量级飞两趟，且 aux loss 只保证长期均衡、单步仍可能个别卡排队（§11.2）；EP 只管 FFN 段，注意力部分常另用 TP 切。

---

## 小结

你现在理解了 MoE 架构的每一个细节：

1. **动机**：用稀疏激活实现"大模型容量、小计算量"
2. **路由器**：Gate 线性层 + 全体 softmax + top-K 选择（本仓库解码器不 renorm，直接取 softmax 概率）
3. **专家**：独立的 SwiGLU FFN，结构相同参数不同
4. **共享专家**：永远激活，处理通用知识
5. **精度**：路由器也用 BF16，确保与 Python 一致
6. **优化**：BF16 mmap、缓冲区复用、批量路由
7. **训练视角**：aux loss（f_i·P_i，α 缩放）防路由塌缩、z-loss 防 logit 膨胀、容量因子定形与 token 溢出——推理时全部消失，只剩 softmax + top-6 + 加权合并
8. **无辅助损失均衡**：DeepSeek 的 bias 调节法；本模型 config 为 greedy，即朴素路由
9. **EP**：专家切分 + dispatch/combine 两次 all-to-all；单机场景退化为 mmap 读 top-6 专家权重

**下一步**：在 [03-推理引擎实战篇](./03-推理引擎实战篇.md) 中，我们将深入纯 C 推理引擎的实现细节——从 safetensors 加载到 kernel 优化，看两个项目如何把理论变成高效的代码。
---

## 🔭 进阶延伸

本篇的后续深化见：09-量化与部署篇（专家权重的 INT8 与页缓存）、10-论文导读（Switch/Mixtral/DeepSeek-V2 谱系）、11-从零构建大模型：预训练篇（MoE 训练在完整训练循环中的位置）。
