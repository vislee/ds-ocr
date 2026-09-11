# 00 · 术语表 Glossary

> 覆盖 ds-ocr 全部 13 篇的**核心名词与专业术语**，按章节归类。每个词条给出**一句话定义** + **关键机制**，并标注对应的深入学习章节。
> 英文术语的缩写（`attn`、`gqa` 等在代码中的写法）见 README 的"术语对照表"。
> 术语解释中 `<理论依据>` 为权威定义要点，完整论文见 10 篇论文导读。

---

## 目录

- [第一部分 · Transformer 基础](#第一部分--transformer-基础)
- [第二部分 · MoE 混合专家](#第二部分--moe-混合专家)
- [第三部分 · 推理引擎](#第三部分--推理引擎)
- [第四部分 · 视觉编码](#第四部分--视觉编码)
- [第五部分 · 分词与词表](#第五部分--分词与词表)
- [第六部分 · 解码策略与后处理](#第六部分--解码策略与后处理)
- [第七部分 · 量化与部署](#第七部分--量化与部署)
- [第八部分 · 训练与对齐](#第八部分--训练与对齐)

---

## 第一部分 · Transformer 基础

### Token / 子词（Subword）
**文本的最小处理单元**，既不是整句也不是单字，而是按频率从语料中切分出的子词。模型在词表里查 token 的 embedding 向量。初学者最常见的误区是以为"词表存单词"——实际上存的是子词。
- 相关章节：01 · 07

### Embedding
**查表操作**：把一个 token 的下标映射成一个稠密向量（hidden 维）。ds-ocr 中输出层 `lm_head` 与输入 embedding **共享权重**，既省内存又帮助训练稳定。
- 相关章节：01

### Attention（自注意力）
**让每个 token 关注序列中其他 token 的机制**。通过 Query、Key、Value 三个投影，计算 Q·K 的相似度得分，softmax 归一化后对 V 加权求和。缩放因子 `√d_k` 是必要的：<不除以它，点积数值随维度增大而膨胀，会使 softmax 进入饱和区、梯度接近 0。>
- 相关章节：01

### Multi-Head Attention（MHA，多头注意力）
把 hidden 拆成多个头，**每个头在独立子空间里关注不同关系**。<多个头并行计算点积注意力，最后拼接再投影，让模型在不同表示子空间捕捉不同的依赖关系。>多头通过"不同子空间关注不同关系"提升表达能力。
- 相关章节：01 · 13

### Grouped Query Attention（GQA，分组查询注意力）
**多个 Query 头共享一组 Key/Value 头**的折中方案。<每个 Query 头仍单独，但 KV 头分成若干组共享。>相比 MHA，KV Cache 显著缩小；相比 MQA 又保留更多表达能力，避免 MQA 的效果损失。从 **KV Cache 占用**角度看，GQA 直接把缓存大小压缩到 `1/G`（G 为每组 query 头数）。
- 相关章节：01 · 03 · 13

### MQA（多查询注意力）
**所有 Query 头共享同一组 K/V** 的极端方案，KV Cache 最小但对效果影响较大，已被 GQA 取代。

### Feed-Forward Network（FFN，前馈网络）
**对每个 token 逐位做非线性变换**的 MLP。<在 Transformer 中通常先升维到 intermediate（如 4×hidden）再降维回 hidden。>与注意力"让 token 交互"不同，FFN 负责"变换信息"，二者是 attention 和 FFN 两大支柱。
- 相关章节：01 · 02

### SwiGLU（激活函数）
**结合 Swish 门控和 GLU 的门控激活函数**：`SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)`，用一个 Swish 激活的分支做"门"来控制另一分支。<由 Shazeer 2020 年提出，相比 ReLU/GELU FFN，SwiGLU 用额外的权重矩阵来做门控，同参数量下表达力更强，被 LLaMA、DeepSeek 等广泛采用。>ds-ocr 的 FFN 正是 SwiGLU。
- 相关章节：01 · 02

### RoPE（旋转位置编码）
**把位置信息编码进 Q/K 的旋转操作**。<RoFormer 提出，将相对位置依赖通过旋转矩阵集成进 self-attention：把向量的相邻两维看成复平面上的点，按位置旋转 θ 角度。>它用**多个频率**（波长从短到长）对不同维度旋转——类似信号处理中的不同频段；高频通道捕捉相邻位置关系，低频通道捕捉长距离关系。<由此 Q、K 的点积自然携带二者的相对位置差，且具备外推能力。>
- 相关章节：01 · 13

### RMSNorm（均方根归一化）
**用均方根值（RMS）做缩放、不减均值**的归一化。<LayerNorm 需要算均值+方差，涉及减法；RMSNorm 只除以 RMS，计算更省、数值更稳定。>残差连接"让梯度有捷径"而不会消失，配合 Pre-Norm 结构被现代大模型普遍采用。
- 相关章节：01

### Pre-Norm vs Post-Norm
**归一化放在残差分支前**（Pre-Norm，现代主流）还是**残差相加后**（Post-Norm，原始 Transformer）。Pre-Norm 更利于深层训练稳定。

### Encoder-Only / Decoder-Only / Encoder-Decoder
三种架构家族：**编码器**（双向注意力，如 BERT）、**解码器**（因果注意力，如 GPT/LLaMA）、**编码器-解码器**（如 T5）。ds-ocr 的解码核心是 **Decoder-Only** 因果注意力。
- 相关章节：01 · 04

### QK-Norm（Q/K 归一化）
对 Query 和 Key 在计算注意力**之前**额外做一次 RMSNorm，抑制 logit 数值膨胀、稳定训练。<常见于 ViT 和部分 MoE 模型，是近年出现的稳定性技巧。>注意：Unlimited-OCR V3 的 decoder **没有** q_norm/k_norm，权重加载需处理这种差异。
- 相关章节：01 · 04

---

## 第二部分 · MoE 混合专家

### Mixture of Experts（MoE，混合专家）
**用多个专家网络 + 一个路由器，让每个 token 只激活少数专家**的架构。<核心思想：模型总参数量很大（容量大），但每次推理只计算少数专家（计算量小）。>稀疏激活带来"大容量、小计算量"的收益。与之相对的是**稠密 FFN**（无路由，每个 token 过全部参数）。
- 相关章节：02 · 13

### Gate / Router（门控，路由器）
**决定每个 token 分给哪些专家**的线性层。<输入 token 的 hidden，经过 gate 得出一组专家得分，再 top-K 选出专家。>
- 相关章节：02

### top-K 路由
**每个 token 只激活得分最高的 K 个专家**（ds-ocr 为 top-6）。K 越大串行计算越多、K 越小越稀疏高效。

### Expert（专家）
**参数独立、结构相同的 SwiGLU FFN**。结构完全相同，参数彼此不同，各自擅长不同的特征/知识子空间。
- 相关章节：02

### Shared Expert（共享专家）
**永远被激活、不参与路由**的少数专家，负责处理所有 token 都需要的通用知识。<与路由专家互补：路由专家负责专精，共享专家兜底共性。>ds-ocr：**64 路由专家（top-6）+ 2 共享专家**。
- 相关章节：02

### Capacity Factor（容量因子）
MoE 训练中为了让专家批处理平衡而引入的缓冲系数。<当某专家收到的 token 过多时，超出 capacity 的 token 会被丢弃或重路由。>非 aux-loss 的负载均衡手段之一。
- 相关章节：02 · 13

### Aux-loss-free（无辅助损失）
**DeepSeek-V3 采用的不加辅助负载均衡损失的 MoE 路由**。<通过给每个专家加偏置（bias）实现负载均衡，避免辅助损失对主损失的干扰，训练更稳定。>
- 相关章节：13

### FLOPs（浮点运算量）
模型一次前向/反向的总浮点运算次数，衡量**计算量**（区别于参数量"容量"）。Roofline 分析中用它与内存带宽、峰值算力的关系判断是否受算力或带宽限制。
- 相关章节：02 · 05 · 09

---

## 第三部分 · 推理引擎

### Prefill vs Decode
**推理的两个阶段**：Prefill 一次性处理整段输入 prompt（批量矩阵乘法，矩阵规模大），Decode 逐个生成 token（矩阵×向量 matvec，带宽受限）。两阶段计算特征完全不同，是性能优化的分水岭。
- 相关章节：03 · 05

### KV Cache（键值缓存）
**解码阶段复用历史 token 的 K/V 向量，避免重复计算**。<attention 中每个新 token 都要与所有历史 token 算点积，若不缓存，逐 token 解码会随长度产生大量重复计算。KV Cache 大小与序列长度**线性相关**，长上下文下内存压力极大——这正是 GQA/MQA/MLA 优化缓存的核心动机。>
- 相关章节：03 · 13

### Logits
**词表上未归一化的原始得分**。<解码最后用 `lm_head`（与 embedding 共享权重）算出每个 vocab token 的得分，再经 softmax 变成概率。>因只需选最大值，贪心解码用 **argmax** 即可，不必对 12 万词表全量 softmax。
- 相关章节：03 · 08

### BF16（Brain Float 16）
**与 FP32 相同指数位（8 位）、更少尾数位（7 位）的 16 位浮点**。<范围与 FP32 一样大、精度只有 7 位有效数字；牺牲精度换更好的动态范围和训练/推理稳定性。>FP16 范围小但精度高；BF16 是"保范围、弃精度"。ds-ocr 权重以 BF16 用 **mmap 零拷贝**存储，计算时 on-the-fly 转成 FP32。
- 相关章节：03 · 09 · 11

### Safetensors
**安全的张量序列化格式**。<只存储权重数据，不包含可执行代码，规避了基于 pickle 的 `.bin` 格式的任意代码执行风险；同时支持 mmap 零拷贝加载，比 torch .bin 更快。>ds-ocr 用 mmap 直接把磁盘上的 BF16 权重映射进地址空间，避免整读。
- 相关章节：03

### mmap（内存映射）
把文件映射到进程地址空间，**按需/零拷贝访问**，不经过标准 read。<配合操作系统的 page cache，模型权重可边用边加载，无需预读整份文件。>这是纯 C 推理引擎"无依赖加载大权重"的关键。
- 相关章节：03 · 09

### Thread Pool（线程池）
**预创建的固定工作线程集合**，任务队列分发，避免频繁创建/销毁线程的开销。<ds-ocr 用它并行化矩阵乘法、多 crop 处理等。>
- 相关章节：03 · 05

### sgemm / matvec
**单精度（fp32 中间计算）通用矩阵乘法** vs **矩阵×向量**。批量上下文（Batch）用 sgemm，逐 token 解码用 matvec。ds-ocr 从 v0 到 v1.1 的优化主线之一就是把 decode 的 matvec 批量化/向量化。
- 相关章节：05

### Kernel / Kernel Fusion（内核融合）
**把多个应当顺序执行的数学运算合并到一个循环/kernel 里执行**。<减少中间结果的落盘往返（内存读写），典型如"在线 softmax + 融合 BF16 matvec"。>ds-ocr 在三层（generic/NEON/AVX）上做平台优化并融合算子。
- 相关章节：03 · 05 · 09

### Continuous batching / Chunked prefill / Prefix caching
**服务化推理的高效技术**：
- **Continuous batching**（连续批处理）：不等一个整 batch 完成就插入新请求，动态组合长短请求，提高 GPU 利用率。
- **Chunked prefill**：把长 prompt 的 prefill 拆成小块穿插到 decode 中，压低首 token 时延尖峰。
- **Prefix caching**：复用相同 prompt 前缀的 KV Cache。
- 相关章节：03 · 13

### PagedAttention / vLLM
**vLLM 提出的按页管理 KV Cache**，类似操作系统虚拟内存分页。<解决 KV Cache 碎片化和内存预分配浪费，是 vLLM 高吞吐推理的核心。>
- 相关章节：03 · 09 · 13

### Speculative Decoding（投机解码）
**用一个便宜的小模型草拟多个 token，大模型并行验证**，在结果一致时一次前向产出多个 token，加速自回归。

### Profiler（性能分析器）
**统计各阶段耗时的工具**，找出热点（hotspot）再针对性优化。ds-ocr 的 v0.5→v1.1 优化就是在 profiler 定位出 prefill/decode 各自瓶颈后展开的。
- 相关章节：05

---

## 第四部分 · 视觉编码

### Patch（图像块）
**把图像切成的小块**。<ViT 不把整张图直接喂入，而是切成固定大小的 patch（如 16×16 像素），每个 patch 线性投影成一个向量。>ds-ocr 的 768×768 图像 → 48×48=2304 个 patch。相邻 patch 像素相关性极强，天然冗余——这正是后续压缩的起点。
- 相关章节：01 · 04 · 06

### ViT（Vision Transformer）
**把 Transformer 应用到图像**的模型。<Google 首次提出：图像切成 patch 做 patch embedding，加上 [CLS] token 和位置编码后送到标准 Transformer。>ds-ocr 的视觉侧基于 SAM 内部的 ViT 编码器（ViT-B）。
- 相关章节：04 · 06

### SAM（Segment Anything Model）
**通用的图像分割基础模型**。<图像编码器用 MAE 预训练的 ViT（支持 ViT-B/L/H），对高分辨率输入做适配。>ds-ocr 用 **SAM ViT-B** 作为视觉特征提取器，生成每个 patch 的高维特征 token。
- 相关章节：04 · 06 · 10

### CLS token
**附加在序列开头的特殊 token，用来聚合整张图/整序列的信息**，通常取其最终表示为分类/全局特征。

### Conv2D / Conv 压缩（视觉压缩）
**用卷积把大量 patch token 压缩成少量 token**。<相邻 patch 信息高度重叠，卷积通过合并相邻 patch 的公共特征，只输出"差异信号"，从而去冗余。>DeepSeek-OCR 视觉压缩链路：768×768 图 → 2304 patches（896 维）→ SAM → 2304 tokens → Conv 压缩 → **144 tokens**（压缩比 16×）。
- 相关章节：04 · 06

### Projector（投影器）
**把视觉特征对齐到语言 token 空间的线性/MLP 映射**，让视觉 token 能被 decoder 直接当作输入 token 处理。
- 相关章节：04

### Global crop / Local crop / Multi-crop（动态多裁剪）
**把图像分成全局图和若干局部裁剪图分别编码**的技术。<Unlimited-OCR V3 用 `down_up_scale_1024` 从全局 1024×1024 图生成 640×640 的局部 crop，各 crop 用 `view_seperator` token 分隔、逐 crop 融合特征。>DeepSeek-OCR 主打 Doc 文档+纯文本，多 crop 主要用于处理图文混排/局部细节。
- 相关章节：04 · 06

### R-SWA（Reference Sliding Window Attention，参考滑窗注意力）
**融合"全局参考 token 注意力"与"局部滑窗注意力"的机制**。<让每个 token 既能访问固定窗口内的局部上下文，又能访问少数全局参考 token（如首部 summary/图像级 token），兼顾局部细节与全局语义、控制计算量。>ds-ocr 的 V3 解码器用它作为注意力结构。
- 相关章节：04 · 05 · 06

### NaViT（Patch n' Pack）
多尺度 patch 打包训练 ViT 的方法，可处理**可变分辨率**的输入。
- 相关章节：06 · 10

---

## 第五部分 · 分词与词表

### BPE（Byte Pair Encoding，字节对编码）
**基于"反复合并最高频子词对"构建词表的子词分词算法**。<从一个字符/字节的初始词表出发，每次统计语料中相邻子词对的出现频率，把最高频的一对合并成一个新子词，直到词表达到目标大小。BPE 按频率直接合并；WordPiece 则按合并分数挑选。>
- 相关章节：07

### Byte-level BPE（BBPE，字节级 BPE）
**在 UTF-8 字节层面做 BPE**。<把任意文本（含所有语言/Emoji）都编码成字节再 BPE 合并，保证零 OOV、任何输入都能 round-trip 还原。>GPT-2/LLaMA 均采用，ds-ocr 的 detox 流程依赖它保证 token 可逆。
- 相关章节：07

### Unicode / UTF-8
**统一字符集** vs **变长编码**：UTF-8 用 1–4 字节编码所有 Unicode 字符。字节级 BPE 直接在 UTF-8 字节上操作，天然支持多语言与 Emoji。
- 相关章节：07

### Vocabulary（词表）
**模型能识别的全部 token 的集合**。<词表大小决定 embedding/lm_head 的行数；改变词表 = 全部 embedding 重训。>ds-ocr 词表 **129280**，中文场景偏大词表可降低序列长度。
- 相关章节：07 · 11

### OOV（Out-of-Vocabulary，词表外词）
**词表中不存在的词**。Byte-level BPE 因在字节层操作，理论上**消除 OOV**——任何输入字节都可被编码。
- 相关章节：07

### Added / Special tokens（特殊 token）
**词表中的控制符号**，如 `<|begin_of_image|>`、`<|eos|>`、`<|newline|>`、`view_seperator` 等，用于标记边界、结构、结束。cfg 中的 `added_tokens` 需在 BPE merge 之外单独处理。
- 相关章节：04 · 07 · 08

### SentencePiece
Google 的**无监督文本 tokenizer**（支持 BPE/Unigram），常与 byte-level 结合跨语言分词。

---

## 第六部分 · 解码策略与后处理

### argmax / 贪心解码（Greedy）
**每步选 logits 最大的 token**。最快但可能陷入局部最优、输出重复。
- 相关章节：08

### Temperature（温度）
**softmax 前对 logits 缩放的温度参数**。`T>1` 分布更平缓（更随机），`T→0` 逼近 argmax（更确定）。
- 相关章节：08

### Top-K / Top-P 采样
**限制候选 token 集合的采样策略**：Top-K 只保留 logits 最高的 K 个，Top-P（nucleus）保留累计概率达到 p 的最小集合。两者都用于剪掉低概率长尾、提高采样质量。
- 相关章节：08

### Repetition Penalty（重复惩罚）
**对已出现 token 的 logits 打折**，抑制模型输出重复/复读。
- 相关章节：08

### n-gram 阻断（No Repeat Ngram）
**当预测的 token 会与历史组成已出现过的 n-gram 时，直接屏蔽该 token**，从根源上防止重复片段。
- 相关章节：08

### EOS（End-of-Sentence）token
**表示生成结束的特殊 token**。<解码输出 `<|eos|>` 时停止生成。>OCR 中控制何时终止对图片/文本的描述。
- 相关章节：04 · 08

### DET（Document Element Tagging，文档元素标签）
**DeepSeek-OCR 的 doc 输出中的结构化标签系统**，用 `<|det|>标题<|/det|>` 之类的标签标注版面结构。<后处理需清洗/校验这些标签，去掉不完整的标签对，避免输出格式破坏。>
- 相关章节：08

### Hallucination（幻觉）
**模型生成与输入不符的"无中生有"内容**。<OCR 场景常见于模型在图片之外杜撰文本；BF16 截断/数值问题可能放大某些边界情形下 eos 概率波动触发幻觉。>后处理层做防御（如 min_new_tokens、标签校验）。
- 相关章节：08

### CFG（Classifier-Free Guidance）
**无分类器引导**，用条件/无条件 logits 加权的采样技巧，增强输出对条件（prompt/图像）的遵循度。
- 相关章节：04 · 08

---

## 第七部分 · 量化与部署

### Quantization（量化）
**用更低精度（如 INT8/INT4）近似表示权重/激活**，压缩模型体积、减少访存和计算。<量化核心是**缩放因子 scale**：把浮点范围映射到整数范围 `q = round(x / scale) + zero_point`。>可在大模型部署时显著降低带宽占用与内存。
- 相关章节：09

### INT8 / INT4
8 位 / 4 位整数量化。<INT8 较成熟、精度损失小；INT4 内存更省但反量化开销高。>注意：ds-ocr 实测 INT4 量化在当前实现上**比 BF16 更慢**——反量化开销大于省下的带宽，是"量化为省带宽"需要权衡的典型反例。
- 相关章节：09

### Per-row / Per-channel 量化粒度
按行/按通道分别算缩放因子的量化。<粒度越细越能保留 outlier，但也引入更多 scale 存储开销。>Per-row 在"保护异常值 vs 存储开销"间取折中，ds-ocr 采用 per-row。
- 相关章节：09

### Outlier（异常值）
**激活或权重中少数幅度远大于整体分布的数值**。<LLM 通常存在 outlier 通道，它们对精度影响极大；窄尾数精度（如 BF16 尾数只有 7 位）容易在此失真。>量化需要特殊处理 outlier 以避免精度崩塌。
- 相关章节：09 · 03

### AWQ（Activation-aware Weight Quantization，激活感知量化）
**根据激活幅度决定哪些权重通道更重要并加以保留**的量化方法。<结合逐通道缩放，优先保护与高幅值激活对应的权重通道，显著降低量化掉点。>
- 相关章节：09 · 10

### SmoothQuant
**把激活中的 outlier 迁移到权重里再量化**的方法，让"难量化的激活"变成"好量化的权重"。
- 相关章节：09 · 10

### Roofline（屋顶模型）
**用"峰值算力"与"内存带宽"两条线判断一个计算核心受哪种限制**的分析模型。<在交点左侧受带宽限制（memory-bound），右侧受算力限制（compute-bound）。>纯 CPU 推理的 matvec/INT8 通常落在带宽受限区，这是 ds-ocr 性能优化的理论抓手。
- 相关章节：09 · 05

### NEON / AVX / SIMD
**ARM / x86 CPU 的单指令多数据（SIMD）向量指令集**。<一条指令同时对多个数据做运算，是 kernel 提速的核心手段。>ds-ocr 按编译期检测分发到 generic/NEON/AVX 三层 kernel。
- 相关章节：03 · 09

### MLC / TVM
**MLC-LLM 与 Apache TVM**：基于编译优化的 LLM 部署框架，支持把模型编译成跨平台的高效二进制，与手写 C kernel 是"自动 vs 手动"的两种优化路线。
- 相关章节：09 · 10

### vLLM / PagedAttention
见第三部分"PagedAttention / vLLM"。高吞吐 LLM 推理服务器及其页式 KV Cache 管理方案。
- 相关章节：09 · 13

---

## 第八部分 · 训练与对齐

### Scaling Laws（缩放定律）
**描述模型性能随参数量/数据量/算力增长的规律**。<Chinchilla 定律给出"给定算力下最优 token/参数比"，指导资源配置。>是"为什么模型越大越强"背后的量化规律。
- 相关章节：11 · 10

### 预训练（Pretraining）
**在大规模无标注语料上预测下一个 token**，让模型学到语言/世界知识。<训练目标是 next-token prediction 交叉熵损失。>ds-ocr 的解码器在预训练阶段已具备文本生成能力。
- 相关章节：11

### 后训练 / 对齐（Post-training / Alignment）
**把预训练模型与人类偏好对齐**的阶段，包含 SFT（监督微调）、RLHF/DPO/GRPO 等。<解决"能力有了但不听话/不正确"的问题。>
- 相关章节：12

### SFT（Supervised Fine-tuning，监督微调）
**用标注好的 (prompt, 期望输出) 数据微调模型**，教会模型特定任务/指令行为。
- 相关章节：12

### RLHF（Reinforcement Learning from Human Feedback）
**基于人类反馈的强化学习**：训练奖励模型打分，再用 PPO 等强化学习优化策略，让输出符合人类偏好。<需要额外的奖励模型和在线采样，训练成本高。>
- 相关章节：12

### PPO（Proximal Policy Optimization）
RLHF 中最常用的强化学习算法，用**价值网络（critic）**估计优势函数、约束策略更新幅度。<需要同时维护策略、critic、reward 多个模型，工程复杂。>
- 相关章节：12

### DPO（Direct Preference Optimization，直接偏好优化）
**不训练奖励模型、不用强化学习，直接用偏好对优化策略**。<核心思想：用"增加偏好样本、降低非偏好样本"的方式，将 RLHF 目标改写为静态损失，训练更简单稳定。>
- 相关章节：12

### GRPO（Group Relative Policy Optimization，群体相对策略优化）
**DeepSeek 提出的强化学习算法，用组内相对奖励做基线、免去 critic 价值网络**。<传统 PPO 需要额外训练 value function 估计优势；GRPO 通过对同问题采样一组输出、以组内相对得分作 baseline，简化训练。>DeepSeek-R1-V3 的 reasoning 训练核心。
- 相关章节：12

### Reward model / Reward hacking（奖励模型 / 奖励黑客）
**给输出打分的模型** vs **模型钻奖励函数漏洞刷分的现象**。<reward hacking 是 RL 领域典型失败模式，需设计 reward 抵制 (例如复用版 eos/hack 检测)。>
- 相关章节：12

### LoRA / QLoRA（低秩适配）
**冻结原权重，只训练一小部分低秩增量 ΔW = A·B** 的微调方法。<参数量骤减、显存占用低；QLoRA 再对权重做量化进一步省显存。>适合小成本微调大模型。
- 相关章节：12

### RAG（Retrieval-Augmented Generation，检索增强生成）
**生成前先从知识库检索相关内容注入 prompt**，缓解幻觉、引入外部知识。
- 相关章节：08 · 13

### AMP（Automatic Mixed Precision，自动混合精度）
**训练时自动在 FP32/BF16（或 FP16）间切换**。<用半精度算前向/反向、FP32 维护主权重与优化器状态，兼顾速度与稳定。>它正是"BF16 训练"的工程实现。
- 相关章节：11

### FSDP / Megatron-LM（分布式训练）
**数据/张量并行的分布式训练框架**。<FSDP 把模型分片到多卡（ZeRO），Megatron-LM 提供张量/流水并行。>训练超大模型必备。
- 相关章节：11

### AdamW / Warmup（优化器与预热）
**去权重衰减的 Adam 变体** 与 **学习率先升后降的预热策略**，现代 LLM 训练标配。
- 相关章节：11

---

## 参考资料

术语定义的权威依据来自以下来源。标 🔗 的论文与 [10-论文导读与进阶路线篇](./10-论文导读与进阶路线篇.md)§2「必读论文」表格逐行对应，可在该篇看每篇的"带走一个思想 + 对应 C 代码"。

- 🔗 ViT：*An Image is Worth 16x16 Words*（Dosovitskiy et al.）→ 论文表 #6
- 🔗 SAM：*Segment Anything*（Kirillov et al.）→ 论文表 #8
- 🔗 RoPE：*RoFormer*（Su et al.）→ 论文表 #4
- 🔗 GQA：*GQA: Training Generalized Multi-Query Transformer*（Ainslie et al.）→ 论文表 #9
- 🔗 SwiGLU：*GLU Variants Improve Transformer*（Shazeer, 2020）→ 论文表 #5
- 🔗 FlashAttention：*FlashAttention*（Dao et al.）→ 论文表 #10
- 🔗 RMSNorm：*Root Mean Square Layer Normalization*（Zhang & Sennrich）→ 论文表 #2
- 🔗 Attention：*Attention Is All You Need*（Vaswani et al.）→ 论文表 #1
- Safetensors / KV Cache / PagedAttention / vLLM：官方文档，推理系统部分见论文导读 §4
- 🔗 DPO：*Direct Preference Optimization*（Rafailov et al.）→ 论文导读 §5/12 篇
- 🔗 GRPO：DeepSeekMath *Incentivizing Reasoning Capability*（Shao et al.）→ [12-后训练与对齐篇](./12-从零构建大模型：后训练与对齐篇.md)
- AWQ / SmoothQuant：对应量化论文（非表内，见 [09-量化与部署篇](./09-量化与部署篇.md)）
- 🔗 Chinchilla：*Training Compute-Optimal LLMs* → [11-预训练篇](./11-从零构建大模型：预训练篇.md)

> 部分术语（如 MLA、R-SWA、MoE 负载均衡）在本仓库关联模型代码中有更具体的工程形态，
> 可配合 03/04/13 篇的完整代码追踪阅读。

