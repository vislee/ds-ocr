# ds-ocr 学习文档：从入门到精通大模型推理

> 通过 ds-ocr（DeepSeek-OCR 推理引擎）和 qwen-asr（Qwen3-ASR 推理引擎）两个纯 C 项目，从零理解大模型推理的完整知识体系。支持 DeepSeek-OCR V1/V2 和 Unlimited-OCR V3 三种模型。

---

## 📚 文档目录

| 序号 | 文档 | 主题 | 阅读时间 |
|------|------|------|---------|
| 00 | [术语表 Glossary](./00-术语表-Glossary.md) | 全部 13 篇核心名词/专业术语，按章节归类速查 | ~20 min |
| 01 | [Transformer 基础篇](./01-Transformer基础篇.md) | 注意力、多头、FFN、位置编码、归一化、残差 | ~30 min |
| 02 | [MoE 混合专家篇](./02-MoE混合专家篇.md) | 路由、专家、共享专家、稀疏激活 | ~25 min |
| 03 | [推理引擎实战篇](./03-推理引擎实战篇.md) | Safetensors、BF16、内核优化、KV缓存、采样 | ~35 min |
| 04 | [完整推理流程篇](./04-完整推理流程篇.md) | V1/V2/V3 三版本全链路：图像→编码→解码→文字 | ~30 min |
| 05 | [性能优化实战篇](./05-性能优化实战篇.md) | v0.5→v1.1 优化之路：Profiler、并行、批量sgemm、R-SWA | ~30 min |
| 06 | [视觉编码器篇](./06-视觉编码器篇.md) | ViT、SAM 窗口注意力、CLIP、位置嵌入插值、动态多裁剪 | ~30 min |
| 07 | [分词与词表篇](./07-分词与词表篇.md) | BPE/byte-level BPE、特殊 token、added_tokens、三个真实 bug | ~20 min |
| 08 | [解码策略与后处理篇](./08-解码策略与后处理篇.md) | 贪心/采样、重复惩罚、n-gram 阻断、det 标签清理、幻觉防御 | ~25 min |
| 09 | [量化与部署篇](./09-量化与部署篇.md) | INT8 per-row、带宽roofline、mmap/页缓存、平台指令分发 | ~25 min |
| 10 | [论文导读与进阶路线篇](./10-论文导读与进阶路线篇.md) | 15 篇必读论文导读、谱系图、动手课题、精通自测清单 | ~30 min |
| 11 | [从零构建大模型：预训练篇](./11-从零构建大模型：预训练篇.md) | 理论+实战：数据配方、算力账、训练循环、失败模式速查、OLMo/TinyLlama 复现路线 | ~45 min |
| 12 | [从零构建大模型：后训练与对齐篇](./12-从零构建大模型：后训练与对齐篇.md) | 理论+实战：微调/RL 决策框架、LoRA 超参表、DPO/GRPO 完整配置、reward 设计、调试速查 | ~45 min |
| 13 | [进阶专题篇](./13-进阶专题篇.md) | 训练数学、FlashAttention 推导、MHA→MLA、长上下文、MoE 训练细节、服务系统 | ~40 min |

**总计约 6.5 小时**：01–05 打地基（推理入门），06–10 建高楼（推理精通），11–12 追本溯源（训练视角），13 攻坚六大前沿专题。阅读前可先花 20 分钟过一遍 [00 术语表](./00-术语表-Glossary.md) 打底，读证随时回查。

---

## 🗺️ 学习路径

```
【入门】零基础 ─→ 01 Transformer基础 ─→ 02 MoE混合专家 ─→ 03 推理引擎实战
                  (理解骨架)           (理解创新)        (理解实现)
【进阶】      ─→ 04 完整推理流程 ─→ 05 性能优化实战 ─→ 06 视觉编码器 ─→ 07 分词与词表
                  (理解全貌)          (理解极限)         (看懂图像侧)      (看懂文本侧)
【精通】      ─→ 08 解码策略与后处理 ─→ 09 量化与部署 ─→ 10 论文导读与进阶路线
                  (控制输出质量)         (压榨硬件)        (论文↔代码地图)
【溯源】      ─→ 11 从零构建：预训练 ─→ 12 从零构建：后训练与对齐
                  (模型怎么炼成)          (行为怎么塑造)
【攻坚】      ─→ 13 进阶专题：训练数学/FlashAttention/MLA/长上下文/MoE/服务
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
