# ds-ocr 学习文档：从入门到精通大模型推理

> 通过 ds-ocr（DeepSeek-OCR 推理引擎）和 qwen-asr（Qwen3-ASR 推理引擎）两个纯 C 项目，从零理解大模型推理的完整知识体系。

---

## 📚 文档目录

| 序号 | 文档 | 主题 | 阅读时间 |
|------|------|------|---------|
| 01 | [Transformer 基础篇](./01-Transformer基础篇.md) | 注意力、多头、FFN、位置编码、归一化、残差 | ~30 min |
| 02 | [MoE 混合专家篇](./02-MoE混合专家篇.md) | 路由、专家、共享专家、稀疏激活 | ~25 min |
| 03 | [推理引擎实战篇](./03-推理引擎实战篇.md) | Safetensors、BF16、内核优化、KV缓存、采样 | ~35 min |
| 04 | [完整推理流程篇](./04-完整推理流程篇.md) | 图像/音频→编码→解码→文字的全链路 | ~25 min |

**总计约 2 小时**，从零基础到能读懂大模型推理引擎的 C 代码。

---

## 🗺️ 学习路径

```
零基础 ──→ 01 Transformer基础 ──→ 02 MoE混合专家 ──→ 03 推理引擎实战 ──→ 04 完整推理流程
           (理解骨架)             (理解创新)          (理解实现)           (理解全貌)
```

每篇文档都是**理论 + 代码**双线并行：
- 📖 **理论**：深入浅出讲解原理，用类比和图示帮助理解
- 💻 **代码**：直接引用 ds-ocr 和 qwen-asr 的 C 代码，理论与实践一一对应

---

## 📖 各篇概要

### 01 - Transformer 基础篇
理解 Transformer 的每一个核心组件：
- **嵌入层**：token → 向量（查表），BF16 零拷贝，嵌入共享
- **位置编码**：RoPE 旋转编码（解码器）、2D 位置编码（视觉）、正弦编码（编码器）
- **自注意力**：QKV 投影、缩放点积、因果掩码
- **多头注意力**：MHA → GQA → MQA 的演进
- **前馈网络**：标准 GELU FFN → SwiGLU
- **归一化与残差**：RMSNorm、Per-head Q/K Norm、Pre-Norm
- **编码器 vs 解码器**：双向 vs 因果 vs 混合注意力

### 02 - MoE 混合专家篇
深入 MoE 架构的每一个细节：
- **动机**：大模型容量、小计算量
- **路由器**：Gate 线性层 + softmax + top-K 选择
- **专家结构**：独立的 SwiGLU FFN，结构相同参数不同
- **共享专家**：永远激活，处理通用知识
- **精度**：BF16 路由确保与 Python 一致
- **ds-ocr 实例**：64 路由专家(top-6) + 2 共享专家
- **完整代码追踪**：从路由得分到最终输出的每一步

### 03 - 推理引擎实战篇
纯 C 推理引擎的实现细节：
- **Safetensors**：mmap 零拷贝加载、多分片支持
- **BF16 精度**：存储减半、on-the-fly 转换、流式 argmax
- **数学内核**：generic → NEON → AVX → BLAS 三级优化
- **KV 缓存**：数据结构、GQA 对缓存大小的影响、动态扩展
- **Prefill vs Decode**：批量矩阵乘法 vs 矩阵×向量
- **采样**：贪心、温度、重复惩罚、n-gram 阻断
- **线程池**：并行矩阵乘法、多 crop 并行

### 04 - 完整推理流程篇
从输入到输出的全链路解析：
- **ds-ocr**：图像预处理 → SAM ViT → DeepEncoder V2 → Projector → Prefill → Decode → 文字
- **qwen-asr**：Mel 频谱图 → Conv+Transformer → Projector → Prefill → Decode → 文字
- **共性模式**：预处理 → 编码 → 投影 → Prompt → Prefill → Decode → 解码
- **性能分析**：耗时分布、内存带宽瓶颈、优化方向

---

## 🔗 项目链接

- [ds-ocr](https://github.com/vislee/ds-ocr) — DeepSeek-OCR 纯 C 推理引擎
- [qwen-asr](https://github.com/antirez/qwen-asr) — Qwen3-ASR 纯 C 推理引擎（antirez）

---

## 📋 前置知识

- **编程**：基础 C 语言（能读懂变量声明、for 循环、指针）
- **数学**：高中数学（向量点积、矩阵乘法、指数函数）
- **无需**：Python、PyTorch、深度学习基础（文档从零讲起）

---

## 🎯 如何使用这些文档

### 零基础读者
按 01 → 02 → 03 → 04 的顺序阅读，每篇读完后在代码中找到对应的实现。

### 有 Transformer 基础的读者
可以跳过 01，直接从 02 开始（MoE 是现代大模型的关键创新）。

### 想直接看推理流程的读者
直接跳到 04，遇到不理解的组件再回看前面的章节。

### 想修改代码的读者
03 是最重要的——理解了引擎的实现细节，才能安全地修改和优化。

---

## 📝 术语对照表

| 英文术语 | 中文 | 代码中的缩写 |
|---------|------|-------------|
| Attention | 注意力 | `attn` |
| Multi-Head Attention | 多头注意力 | `mha` |
| Grouped Query Attention | 分组查询注意力 | `gqa` |
| Feed-Forward Network | 前馈网络 | `ffn`, `mlp` |
| Rotary Position Embedding | 旋转位置编码 | `rope` |
| Mixture of Experts | 混合专家 | `moe` |
| Key-Value Cache | 键值缓存 | `kv_cache` |
| Brain Float 16 | BF16 浮点 | `bf16` |
| Root Mean Square Normalization | 均方根归一化 | `rms_norm` |
| Layer Normalization | 层归一化 | `layer_norm` |
| Tokenizer | 分词器 | `tokenizer` |
| Embedding | 嵌入 | `embed`, `emb` |
| Logits | 逻辑值/未归一化概率 | `logits` |
| Autoregressive | 自回归 | — |
| Prefill | 预填充 | `prefill` |
| Decode | 解码 | `decode`, `dec_` |
| Gate/Router | 门控/路由器 | `gate` |
| Expert | 专家 | `expert` |
| SwiGLU | SwiGLU 激活函数 | `swiglu` |
