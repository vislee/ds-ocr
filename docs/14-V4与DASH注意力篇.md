# 14 - V4 与 DASH 注意力篇：多页文档的块稀疏注意力

> **阅读目标**：读懂第 4 个模型 Unlimited-OCR-2（V4）的两个核心机制——**page2d 二维位置编码**与 **DASH 块稀疏注意力**，以及把 C 实现从"输出乱码"修到"与官方逐字一致"过程中踩过的 **5 个真实 bug**。读完你将掌握：多页文档的位置编码设计、块稀疏路由的完整数学、以及"有回归门控的数值实验"这套工程方法。
>
> **前置**：04 章（V1/V2/V3 全链路）、05 章（性能优化方法）、13 章 2.3 节（entmax 与注意力稀疏化）。

---

## 目录

1. [V4 是什么：多页文档与部署形态](#1-v4-是什么)
2. [问题：多页文档为什么需要新的位置编码](#2-问题)
3. [page2d 二维位置编码](#3-page2d-二维位置编码)
4. [DASH 块稀疏注意力：完整数学](#4-dash-块稀疏注意力)
5. [页坐标状态机：解码时如何推进](#5-页坐标状态机)
6. [五个真实 bug 复盘：从乱码到逐字一致](#6-五个真实-bug-复盘)
7. [V3 R-SWA vs V4 DASH 对比表](#7-对比表)
8. [工程方法：有回归门控的数值实验](#8-工程方法)
9. [自测](#9-自测)

---

## 1. V4 是什么

Unlimited-OCR-2（下称 V4）与 V3（Unlimited-OCR）共享同一套视觉编码器（SAM + CLIP 特征拼接 + 线性投影），区别全部在**解码器的注意力方案**上：

| | V3 | V4 |
|---|---|---|
| 目标场景 | 单页长文档 | **多页文档**（prompt = `<PAGE><image>` × N 页 + 指令） |
| 注意力 | R-SWA：参考集 + 滑动窗口 128 | **DASH**：块稀疏路由 + 局部窗 |
| 位置编码 | 1D RoPE（theta=10000） | **page2d 二维 RoPE**（tok 轴 + page 轴） |
| 部署对齐 | HF transformers | vLLM `unlimited_page2d_dash` 插件（C 实现对齐的是 HF 里的 vLLM-parity 路径） |
| 解码采样 | ngram=35 守卫 | 纯贪心（官方 infer 默认无 ngram） |

配置在 `models/Unlimited-OCR-2/config.json` 的 `sparse_attention_config` 节点，C 实现的超参数常量集中在 `ds_dash.h`。

> **模型版本探测**：`detect_model_version()`（ds_ocr.c）在 V3 **之前**检查 `"training_stage"=="dash"` 与 `sparse_attention_config`——因为 V4 的 config 同样含 `sliding_window_size`，检查顺序错了会把 V4 误判成 V3。

---

## 2. 问题：多页文档为什么需要新的位置编码

先想清楚 V3 的 R-SWA 为什么不够用。

V3 的方案是"参考集（全部视觉 token）+ 最近 128 token 永远可见，中间的滑出窗口"。对单页文档，参考集就是全部图像内容，天然正确。但多页文档下两个问题出现：

1. **位置编码混页**：1D RoPE 下，第 2 页的第 10 个视觉 token 与第 1 页的第 10 个视觉 token 的相对距离 = 整页长度。模型要靠"距离"判断"这行字属于哪一页的哪张图"，页数一多距离频谱就被拉爆。
2. **注意力预算爆炸**：N 页 × 273 token/页的 KV 全部参与注意力，而生成的正文 token 其实只需要看**自己那一页**的图像。

V4 的解法分两步：**把"第几页"编码进位置**（page2d RoPE，第 3 节），**把注意力预算按页路由**（DASH，第 4 节）。

---

## 3. page2d 二维位置编码

### 3.1 页桶：每个 token 有两个坐标

对整段序列构建 `(tok_pos, page_pos)` 两个坐标（官方 `_build_page_rope_pos`，C 实现 `ds_dash_build_coords()`）：

```
prompt = [BOS, <PAGE>, img×273, 指令×6]
          │      │       │        │
page_pos: 0      1       1        0     ← 指令在最后一张图之后 → 保留桶 0
tok_pos:  0      0       1..273   1..6   ← 页桶内的 running count
```

三条规则（都来自官方实现，C 代码逐条对齐）：

1. **`<PAGE>` 是 1-based 计数**：遇到 marker，page 加一。所以第一页数据的 page=1（不是 0）。
2. **保留桶 0**：BOS 和"最后一张图之后的指令文本"的 page 回到 0。这样指令与任何真实页的 Δpage ≥ 1，永远不会被误认为"同页"。
3. **页内 running count**：tok 是"本页桶内第几个 token"（跨不连续区间累计）。官方注释解释了为什么不能用"段内偏移"：页 k 拥有多段不连续区域（prompt 图像段 + 响应正文段），段内重启会让两个不同 token 撞坐标。

### 3.2 角度：tok 轴 63 对 + page 轴 1 对

`DeepseekV2IndexerRotary2D`（C 实现 `rope2d_cos_sin_row()`）把 128 维的旋转角拆成：

- **tok 轴 63 对频率**：`inv_tok[i] = 10000^(-i/63)`（resample 模式——在 63 对里重新采样整个频谱，两端波长都保留）
- **page 轴 1 对频率**：`inv_page = 256^(-0/1) = 1` → **角度 = (page % 2) × 1.0 弧度**

> ⚠️ **最常见的实现错误**：把 page 轴角度写成 `p / 256`。`256` 是 base，指数是 `j/page_pairs = 0/1 = 0`，所以 `inv_page = 256⁰ = 1`——角度就是 `(page%2)` 弧度本身。这不是"除以基频"，而是"0 号频率对，基频无所谓"。角度差 256 倍后同页/跨页的相位差被抹成 ~0.004 rad，页判别能力全部丢失（这就是第 6 节 bug ③）。

`page_mod=2` 把 page 折叠成 0/1 交替——**只表达"同页/不同页"**，不表达第几页。配合 tok 轴的页内坐标，"第 3 页第 40 个 token"和"第 7 页第 40 个 token"的 tok 角完全相同、page 角相同（都是奇数页），但它们本来就不需要互相区分——路由（第 4 节）负责页选择。

主分支（Q/K）与路由器（idx_q/idx_k）用**同一套 cos/sin 表**（`ds_dash_rope_row()`；解码侧 Q/K 的旋转入口在 `ds_moe_decoder.c` 的 `ds_rswa_attention_aligned_bf16` 之外的 V4 分支——即 `decoder_layer_forward()` 的 V4 分支），因为 `sparse_index_dim == head_dim == 128`；若两者不等，官方会退回 1D（`_dash_indexer_rope` 的告警分支）。

### 3.3 公式推导：page2d 与一维 RoPE 的对比

先回忆一维 RoPE（01 章 §4.3）：把 hidden 的每相邻两维看成一个二维平面，位置 m 的向量旋转 m·θ_i 角。旋转是正交变换，两个旋转后向量的内积只依赖**转角差**：

```
1D RoPE:  <R(mθ)q, R(nθ)k> = Σ_i Re[ q_i · k_i* · e^{i(m−n)θ_i} ]  ← 只依赖 Δm = m−n
```

page2d 做的事是让**不同维度的平面各转各的坐标轴**：128 维 = 64 个平面，前 63 个平面按 tok 坐标转（`rope2d_cos_sin_row()` 的 `ang[i] = tok·10000^(−i/63)`），第 64 个平面按 page 坐标转（`ang[63] = (page%2)·1`）。因为每组平面只被自己的坐标轴驱动，内积按平面**分解**——不同平面之间的项不存在（各自是独立二维子空间的正交旋转，交叉项为零）：

```
page2d:   <R(tok_m,pg_m)q, R(tok_n,pg_n)k>
        = Σ_{i=0..62} f_i(tok_m − tok_n)  +  g(pg_m − pg_n)     ← 两个轴各管各的
其中      f_i(Δtok)  = q_i·k_i·cos(Δtok·θ_i) ± …（63 对，θ_i = 10000^(−i/63)）
          g(Δpg)     = q_63·k_63·cos(Δpg·1 rad)（1 对，Δpg ∈ {0, 1}，因 page%2）
```

两个直接推论：

1. **页内相对距离与文档总长解耦**：tok 坐标是页内 running count，所以 f_i 只看"同一页里相隔几个 token"——第 3 页第 40 个 token 与第 7 页第 40 个 token 的 f 项完全相同。1D RoPE 下它们的 Δm 隔着整页长度，频谱被拉爆（第 2 节的病）；page2d 把病根从"绝对位置"上切掉了。
2. **page 轴只判"同页/跨页"**：page%2 把 Δpg 折叠成 0 或 1。同页 → g 项 = q·k·cos(0) = q·k（完整保留）；跨页 → g 项 = q·k·cos(1 rad) ≈ 0.54·q·k（被压掉 ~46%）。一个弧度就把"同页相似度系统性高于跨页"烧进了打分——**不需要**编码"第几页"，页序判断交给 DASH 路由器（第 4 节的页候选）。这也是为什么 1 rad（`inv_page = 256⁰ = 1`）不是设计缺陷而是设计本身：base=256 只给未来 `page_pairs>1` 的多对 page 频率预留。

与一维路径的实现对照：

| | 1D RoPE（V1/V2/V3） | page2d（V4） |
|---|---|---|
| 坐标 | 绝对位置 m（全序列计数） | (tok, page) 双坐标（`ds_dash_build_coords()`） |
| 频率对 | 64 对共用一根轴 | 63 对 tok + 1 对 page（`rope2d_cos_sin_row()`） |
| 内积只依赖 | Δm | (Δtok, Δpage%2) |
| 跨页行为 | 距离随页数线性膨胀 | 页内距离不变；跨页只有一个固定相位差 |
| 布局 | NeoX 拆半（`ds_apply_rope_neox`，ds_kernels.c） | 同为 NeoX 拆半，cos/sin 行前后半复制（`apply_rope2d`，ds_dash.c） |

---

## 4. DASH 块稀疏注意力

### 4.1 总览：一个 query 看哪些 KV

对每个 query token，可见 KV 集合 = **局部窗 ∪ 被路由选中的远程块**：

```
KV 序列按 32 个 token 一块分块（nblk = ⌈len/32⌉）
├─ 局部窗：位置 (q-128, q]，无偏置，永远可见（sparse_local_block=4 × 块 32）
└─ 远程块：路由器从全部块里挑 K=12 个，各带一个 log-prior 偏置
```

注意力本体：`softmax(QK^T/√128 + m) @ V`，其中掩码 `m`：局部窗=0、选中块=bias、其余=−∞（`biased_attention()` / `_forward_dense`）。BF16 KV 时点积走 `ds_dot_bf16xf32`（bf16 行切片 × f32 Q，展开精确）。

### 4.2 块分数：head_cls 摘要（stage0）

路由器先用两个小投影把每个 token 映射成 128 维索引向量（**输入是层输入残差流，不是 RMSNorm 后的 x_norm**——官方 `_router_qk` 收 `hidden_states`）：

```
idx_q = page2drope( GemmaRMS( q_proj(x) ) )     # GemmaRMS: scale = (1+w)·x/rms(x)
idx_k = page2drope( GemmaRMS( k_proj(x) ) )
```

然后每个块算一个**摘要向量** k_bar：块内 token 对可学习向量 `head_cls` 做 softmax 加权平均（`_block_scores_infer` / C 实现 `block_scores_headcls()`）。k_bar 与 query 无关——**每块算一次，所有 query 复用**（C 实现曾对每个 query 重复计算，实测白烧 ~2.6 GFLOP/层，已修）。

块分数 `z[q,b] = clamp(idx_q · k_bar[b] / √128, ±1e4)`，再套 remote 掩码：`b ≤ q_blk − 4` 的块才是远程候选，否则 z = −1e9（局部窗已被 token 级掩码覆盖，路由只管远处）。

### 4.3 路由选择：top-M → entmax15 → 前 K（vLLM 部署语义）

`_infer_route_select`（C 实现 `route_select()`）四步：

```
1. cand = remote ∧ 页候选        # 页候选 = 含本页 image token 的块（页感知）
   无候选 → 放开全部 remote       # per-query fallback
2. x = cand ? z·idx_scale : −1e4 # idx_scale = dash_scale/√128 = 128/√128 = 11.31
   # 注意：cand 已含 remote —— 未过 remote 的块即使进了 top-M 窗口也是 invalid
3. top-M=16 窗口 → entmax15（α=1.5 稀疏 softmax，支持集平方和=1）
4. 前 K=min(12, M) 个槽位；bias = (log(clamp(prior,1e-4)) − μ)/σ
   # μ 只对 valid 槽求均值；invalid 槽 bias=0 且注意力侧排除
```

三个容易做错的点：

- **K 是读预算不是"远程预算"**。训练路径的 `_remote_topk = topk16 − local4 = 12` 与部署路径的 `dash_topk_blocks=12` 数值巧合相等，但语义不同——部署路径直接选 12 个槽位，不再额外叠加局部块。
- **entmax 只在 top-16 窗口内做**，不是全块。窗口内归一化保证 prior 可与 floor=1e-4 直接 clamp 比较。
- **sentinel（−1e4）会占槽**。候选不足 M/K 时，哨兵补进窗口 → entmax 后其 prior≈0 → clamp 到 floor → logp≈−9.2，但 `valid=False` → **不进 μ、注意力侧排除**。C 实现曾把"页候选但未过 remote"的哨兵槽误标 valid 并参与注意力（bug ⑤），小前缀场景下直接导致输出循环。

### 4.4 解码：同样的路由，逐 query

解码（`_dash_decode_forward` / `ds_dash_attention_decode()`）与 prefill 共用同一套路由数学，差别有三：

1. **idx_k 必须按层缓存全量历史**（官方 `past_kv._dash_idx_k[layer]` 追加）——详见 bug ①。
2. 候选域用 `_uocr_dash_decode_allow` 的页匹配：query 的候选页 `qp = mcum − 1 − pp`（响应正文页 k ↔ 图像页 k），`qp < 0` 或无同页 image 块 → 放开全部 remote。
3. Q/K 的 page2d 角用**页坐标状态机**（第 5 节）递推出来的 (tok, page)。

### 4.5 手算算例：16 个 token 走完 DASH 全流程

真实常数（block=32、local=4、M=16、K=12）太大，没法纸上算。本节把常数等比缩小——**block_size=4、local_blocks=1（局部窗 4 token）、top-M=4、K=2**，路由数学的每一步与 `ds_dash.c` 完全同构，只是量级变了（`idx_scale` 在 toy 里取 1，真实 11.31，只改量级不改排序）。

**输入**：单页 prompt，seq_len=16：

```
pos:   0    1      2   3   4   5   6   7   8   9   10  11  12  13  14   15
内容:  BOS  <PAGE> i   i   i   i   i   i   i   i   i   i   i   i   ins  ins
块:    └─────块 0────┘   └─────块 1────┘   └─────块 2────┘   └──── 块 3 ────┘
```

**第 1 步：页坐标（`ds_dash_build_coords()`，规则见 §3.1）**。last_img=13，pp=1：

```
page:  0    1      1 1 1 1 1 1 1 1 1 1 1 1   0    0     ← 指令回保留桶 0
tok:   0    0      1 2 3 4 5 6 7 8 9 10 11 12   1    2     ← 页内 running count
cand:  -1   0      0 0 0 0 0 0 0 0 0 0  0  0    -1   -1    ← 0-based 图像页
pf_cnt[0]=1（BOS）  pf_cnt[1]=13（<PAGE>+12 img）        ← 状态机种子
块页范围 blk_bmin/bmax: 块 0..3 都是 [0,0]（只统计 <image> token）
```

**第 2 步：prefill 后 decode 两步（`ds_dash_advance_coords()`）**。pos16 生成 `<PAGE>`：mcum=2 → page=min(0+1, n_pg−1=1)=1，tok=pf_cnt[1]=13。pos17 生成正文 token：tok=14。**查询 q_abs=17**，候选页 qp = mcum−1−pp = 0 ✓（响应页 0 ↔ 图像页 0）。

**第 3 步：块分数（`block_scores_headcls()`）**。q_blk = 17/4 = 4，KV 共 18 个 token → nblk=5。remote 掩码：b ≤ q_blk−local_blocks = 3 才是远程 → 块 4（自己所在块）被排除，块 0..3 是候选。设 idx_q 与各块摘要 k_bar 的相容度算出（128 维点积/√128，toy 里直接给出结果）：

```
z = [1.2,  0.3,  −0.8,  2.1,  −1e9]   ← 块 4 被 remote 掩码置 −1e9
     块0   块1   块2    块3    块4
```

**第 4 步：路由（`route_select()` 四步，§4.3）**。页候选：blk_bmin/bmax 覆盖 qp=0 的块 = {0,1,2,3}，与 remote 取合取后 cand 仍是 {0,1,2,3}，无哨兵需要补位。x = z·idx_scale = [1.2, 0.3, −0.8, 2.1]；top-M=4 窗口 = 全部候选；进 `entmax15()`（α=1.5）：

```
xs = (x−max)/2 = [−0.45, −0.9, −1.45, 0]     ← max 取自块 3 的 2.1
降序 [0, −0.45, −0.9, −1.45]，逐前缀求 τ_r = mean_r − sqrt(max(0,(1−ss_r))/r)：
  r=1: τ=−1.000  ≤ xs[0]=0      ✓ support=1
  r=2: τ=−0.895  ≤ xs[1]=−0.45  ✓ support=2   ← τ* 取这里
  r=3: τ=−0.895  ≤ xs[2]=−0.9？ −0.895 > −0.9 ✗（前缀 3 的假设不自洽）
  r=4: τ=−0.700  ≤ xs[3]=−1.45 ✗
τ* = −0.895 → prior = clamp(xs−τ*, 0)²:
  块3: (0+0.895)²  = 0.802      块0: (−0.45+0.895)² = 0.198
  块1: −0.9+0.895 < 0 → 0      块2: → 0        （平方和 = 1.000 ✓）
```

entmax 的稀疏性在这里现形：**块 1、块 2 被精确置 0**——softmax 永远做不到这一点（13 章 §2.3）。取前 K=2 槽位：槽 0 = 块 3（prior 0.802）、槽 1 = 块 0（prior 0.198）。偏置 = (log(prior)−μ)/σ：

```
logp = [−0.221, −1.618]   μ = −0.919（只对 valid 槽求均值）
bias = [+0.70, −0.70]     ← μ 恰是两槽中点，偏置对称
```

**第 5 步：块内注意力（`attn_head_range()`）**。查询 17 的可见集 = 局部窗 (17−4, 17] = pos14..17 ∪ 选中块 {3, 0}：

```
t:     0    1    2    3 │ 4 … 11        │ 12   13   │ 14   15   16   17
bias: −0.70 −0.70 −0.70 −0.70 │ ── 未选中 ── │ +0.70 +0.70 │ 0    0    0    0
归属:  └── 块 0（选中，b=−0.70）┘ │ 块1/2 跳过   │ └─ 块 3 ─┘  │ └─ 局部窗 ─┘
      （t=14,15 虽在块 3 内，但 is_local 先判 → bias=0 而非 +0.70）
```

两个值得停半秒的细节：① **t=4..11 直接 `continue`**——连 QK 点积都不算（块稀疏省算力的物理来源），本例 18 个 KV 只读 10 个；② **局部窗优先于块偏置**（`attn_head_range()` 里 `is_local` 先判），所以同属块 3 的 pos12/13 拿 +0.70、pos14/15 拿 0。之后就是标准的在线 softmax（running max/sum，13 章 §2）：每个可见 token 的分数 = QK/√128 + 上表的 bias，逐头加权求和 V。

**对回真实常数**：本例的 18 KV/10 可见在真实场景（多页文档，total_kv 数千）放大成"每 query ≤ 128+12×32=512 个可见 KV、与文档总长无关"（`attn_head_range()` 注释里的稀疏度算账）；toy 里被 entmax 精确清零的块 1/块 2，对应真实运行中"路由器认为不相关、连一次点积都不值得花"的绝大多数远程块。

---

## 5. 页坐标状态机

prefill 结束时，`ds_dash_build_coords()` 留下三份状态（官方 `_page_rope_meta` + vLLM `page_state`）：

| 状态 | 含义 | 官方来源 |
|---|---|---|
| `pf_cnt[页]` | 每页 **in_prefill**（≤ last_img）token 数——注意指令文本不算 | `(onehot·in_prefill).sum` |
| `pp` / `mcum` | prompt 页数 / 累计 `<PAGE>` 数（含生成） | `page_state.py` |
| `cur_page, cur_tok` | 最后一个 prompt token 的坐标 | `st['page']/st['tok']` |

解码每收到一个 token（`ds_dash_advance_coords()`）：

```
if token == <PAGE>:
    mcum += 1
    page = min(page+1, n_pg−1)
    tok  = pf_cnt[page]          # 响应正文接在该页图像的 running count 之后
else:
    tok += 1
候选页 qp = mcum − 1 − pp        # 响应页 k ↔ 图像页 k；<0 → 候选域放开
```

单页为例：prompt `[BOS, <PAGE>, img×273, 指令×6]`，pf_cnt[1]=274。第一个生成的 token 就是 `<PAGE>`（文档解析的输出格式）→ mcum=2 → page=min(1, 1)=1, tok=274——响应正文与图像共用同一个页桶，tok 无缝续接。

---

## 6. 五个真实 bug 复盘

以下是 C 实现从"输出重复空块"到"与官方逐字一致"过程中真实踩过、逐个对照官方 Python 定位修复的 bug。每条按「症状 → 根因 → 教训」组织。

### bug ①：decode 路由读到跨层脏数据（最致命）

- **症状**：多页大图上，开头若干块内容正确，生成约 128 个 token（局部窗耗尽图像区）后，退化成重复的空坐标块。
- **根因**：官方按层缓存路由 K（`past_kv._dash_idx_k[layer]`，prefill 种子 + decode 追加）；C 实现让 12 层共用一份 scratch，且 decode 只写第 0 行——层 L 的块分数读到的是**其他层的历史 + 垃圾内存**。图像 token 离开局部窗后，唯一的内容来源（路由选块）全部失准。
- **修复**：`idx_k_hist[n_layers][max_seq×128]`，prefill 拷入、decode 追加，读取按层。
- **教训**：多模块共用的 scratch 必须问一句"谁在什么时候覆盖它"；跨层状态绝不能共用。

### bug ②：读预算 K=8（应为 12）+ entmax 作用域错误

- **症状**：与 ① 类似的退化，但更早出现。
- **根因**：把训练路径的 `_remote_topk = topk16 − local4 = 12` 误解成"部署 K=12 里还含 4 个局部"，于是只选 8 个远程；且 entmax 在全部 nblk 上做，而部署语义是 **top-M=16 窗口内 entmax**。两者都让 prior/bias 系统性偏离。
- **教训**：训练路径与部署路径的**同名参数语义可能不同**，以部署路径（vLLM parity）为准。

### bug ③：page 轴角度多除 256

- **症状**：所有多页判别失效，长文档跨页内容张冠李戴。
- **根因**：`inv_page = 256^(0/1) = 1`，角度应为 `(page%2)·1.0 rad`；实现写成了 `p/256`（见 3.2 节 ⚠️）。
- **教训**：公式里每个数字都要回到定义——`base^(j/pairs)` 在 j=0 时恒为 1，base 只是给未来 page_pairs>1 预留的。

### bug ④：entmax 输出按"排序后位置"写回

- **症状**：偏置值与块错配，选中块的先验概率张冠李戴。
- **根因**：官方 `_Entmax15.forward` 对**未排序** X 逐位输出 `clamp(X−τ*,0)²`；C 实现把排序副本的值写回了原数组——prior 被按大小重排。
- **教训**：Python 的 `sort` + 逐位运算移植时，输出索引必须跟随原索引。

### bug ⑤：valid 掩码漏掉 remote 约束（小图循环的直接元凶）

- **症状**：小前缀（候选块 < 12）时，输出陷入重复循环。
- **根因**：官方 `cand = remote ∧ page候选`——top-M 窗口里的 sentinel（−1e4）槽 `valid=False`，不进 μ、注意力排除。C 实现的 valid 只查了页候选，**没查 remote**——被 sentinel 占据的槽拿到 −3.3 的偏置参与注意力。
- **修法**：记录 `eff = remote ∧ cand`，valid 查 eff。
- **教训**：大图（候选 ≥ 12）永远触发不了、小图必现的 bug，用**候选数 < 预算**的测试用例才能抓到。扩充黄金图集的小图就是这么来的。

### 6.6 一个方法论收尾

这五个 bug 全部是**对照官方 Python 逐行定位**的：先把官方 `_INFER` 路径在 CPU 上跑通（torch + 一组兼容 shim），再让 C 与它在同一输入上对拍。中间建立了 `tools/regress.sh`（30 组合冻结标签）作为安全网——**之后每一个数值实验（BF16 KV、专家去拷贝）都先过这道门**。r2 的教训补充：编码/解码侧 1.5% 的 exp 近似就会让 CER 冲破阈值（V3 小图 95.4%），数值精度在这个引擎里不可妥协。

### 6.7 从 bug 学到的工程检查清单

五个 bug 各自沉淀成一条可执行的检查项——下次移植/优化注意力路径时按此自查：

1. **共享 scratch 的"覆盖时间线"**（bug ①）：任何被多个调用方复用的 buffer，画一张"谁在哪个阶段写、谁在哪个阶段读"的时间线；跨层状态（按层缓存）绝不允许共用一块 scratch。修复时配一张写入侧（`router_qk` 拷入历史）+ 读取侧（`block_scores_headcls` 按层读）的双向断言。
2. **同名参数的训练/部署语义核对**（bug ②）：从 config 读出的每个超参，回到**部署路径**的官方代码确认语义（K 是"槽位数"还是"远程预算"），不从训练代码抄。
3. **常数回到定义 + 退化输入用例**（bug ③）：公式里每个数字问一次"它从哪来"（`base^(j/pairs)` 在 j=0 恒为 1）；为退化形态写单元测试（page_pairs=1、单页、单候选）钉住行为——`rope2d_cos_sin_row` 里"bug ③ 防回归点"注释就是这道钉子。
4. **排序+逐位运算的移植**（bug ④）：Python 的 sort-then-transform 移植成 C 时，输出索引必须跟随原索引；对拍用"值=下标"的测试向量，重排一眼现形（`entmax15` 的"排序只用于求 τ*"注释）。
5. **预算不足的边界用例**（bug ⑤）：候选数 < 预算时 sentinel 会占槽——测试集必须包含"小图/短前缀"这类**触发不了满预算**的用例，不只测大图（大图永远测不出这类 bug）；valid 标志必须查完整的合取条件（remote ∧ 页候选），不是其中一个。
6. **通用兜底**：每条修复至少配一个防回归断言或黄金图集用例；"大图必过、小图必挂"的 bug 说明图集的尺寸覆盖有洞——补图集与修 bug 同样重要。

---

## 7. 对比表：V3 R-SWA vs V4 DASH

| 维度 | V3 R-SWA | V4 DASH |
|---|---|---|
| 可见集 | 参考集 [0, prefill_len) ∪ 窗口 (q−128, q] | 局部窗 (q−128, q] ∪ 路由 K=12 块 |
| 位置编码 | 1D RoPE（theta=10000） | page2d：tok 63 对（resample）+ page 1 对（mod 2） |
| 路由 | 无 | head_cls 块摘要 → top16 → entmax15 → 12 槽 log-prior 偏置 |
| 候选域 | — | 同页 image 块（`dash_image_only`），per-query fallback |
| 解码状态 | 无（窗口滑动即可） | 页坐标状态机（pf_cnt/mcum/qp） |
| KV 存储 | F32 或 **BF16**（r2 实验：V3 全绿 + decode +8%） | F32（BF16 实验打红：DASH 对 KV 精度敏感，已回退） |
| 采样守卫 | ngram=35（main.c 默认） | 无（官方纯贪心） |
| C 代码 | ds_moe_decoder.c `ds_rswa_*` | ds_dash.c 全文件 + ds_moe_decoder.c 分支 |

### 7.1 取舍账：计算量 / 显存 / 实现复杂度

上面的表是"机制差异"，选型时还要算三笔账（数字按 V4 真实常数，KV 8192 槽、每页 281 token）：

| 维度 | V3 R-SWA | V4 DASH | 谁赢 |
|---|---|---|---|
| 每 query 注意力读取 | 参考集 [0, prefill_len) ∪ 窗口 128 → **随文档总长线性增长**（单页 ≈400，8 页 ≈2380） | 局部窗 128 ∪ K×32=384 → **≤512，与总长无关**（`attn_head_range()` 的 continue 跳过） | 单页（<384 token）V3 反而更省；**页数 >2 起 V4 反超**，8192 满载省 ~16× |
| 路由自身开销 | 0 | 每 query 对 nblk 个块各一次 128 维点积 + 每块一次 k_bar（每层一次性）| V3；但这是低阶项，注意力本体省的 FLOP 远大于它 |
| KV 显存 | F32 或 BF16（r2 实验：V3 全绿且 decode +8%） | 必须 F32（BF16 打红，§7 表）→ 同长度下是 V3-BF16 的 **2×** | V3；V4 另需 idx_k_hist ≈ 12 层×8192×128×4B ≈ 50 MB |
| 实现复杂度 | 两段**连续区间**扫描，一个函数（ds_kernels.c `ds_rswa_attention_aligned_bf16`），解码零状态 | 离散块并集无法用区间表达 → 逐 token 判定 + 路由五步 + 页坐标状态机 + 按层历史（ds_dash.c 全文件）；五个真实 bug 全出于此 | V3 压倒性；V4 的复杂度是"多页文档"这个需求的对价 |

一句话：**短文档别用 DASH**（机制开销收不回来）；它的收益曲线从"页数×281 超过局部窗+读预算"开始抬头，目标场景本来就是多页长文档——与 §1 的"V4 为多页而生"互为印证。

---

## 8. 工程方法：有回归门控的数值实验

DASH 类优化的每一次改动都可能扰动浮点结果，而 V3/V4 的输出对扰动极其敏感（一个近并列 argmax 翻转就会级联）。本轮沉淀的方法论：

1. **先建门**：`tools/regress.sh` 30 组合冻结标签（人工核对），CER 阈值冻结；`test_unit` 覆盖率套件守护内核数学。
2. **实验有界**：每个数值实验（BF16 KV、专家去拷贝、fast_expf）只碰一个变量，跑门、红即回退。本轮实测：fast_expf 进编码器 → V3 小图 CER 95%；BF16 KV 在 V3 全绿、在 V4 打红 → 按模型粒度启用。
3. **对照官方**：实现差异不靠猜，把官方路径在 CPU 跑通后同输入对拍（V4 五个 bug 全部这样定位）。
4. **失败也记录**：BLOCKED.md 记下每次"为什么不行"，下一轮不再踩。

---

## 9. 自测

1. prompt `[BOS, <PAGE>, img×N, 指令]` 中，BOS 和指令的 page 桶是多少？为什么故意与图像页错开？
2. `inv_page` 为什么等于 1？如果 `page_pairs=2`，两个 page 轴频率分别是多少？
3. 一个 query 最多能看多少个 KV token？分别由哪两部分组成、各带什么偏置？
4. entmax15 在 top-16 窗口上做，窗口里的 sentinel（−1e4）槽最终参与注意力吗？为什么？
5. decode 状态机收到生成的 `<PAGE>` 时，`tok` 为什么重置为 `pf_cnt[page]` 而不是 0？
6. bug ①（idx_k 跨层脏数据）为什么恰好在大图"生成 128 个 token 之后"才暴露？
7. 为什么 V3 的 BF16 KV 实验全绿，V4 却打红？这说明了 DASH 的什么性质？
8. 手算例（§4.5）里 pos12/13 与 pos14/15 同属选中块 3，为什么前者偏置 +0.70、后者是 0？如果块 1/2 也被选中但 prior 被 entmax 清零，它们会参与注意力吗？
9. 单页 281 token 的文档，V3 R-SWA 与 V4 DASH 哪个每 query 读的 KV 更少？页数到多少后结论反转？（提示：§7.1 的账）

---

> **相关**：[04 完整推理流程篇](./04-完整推理流程篇.md)（V1/V2/V3 对照）· [05 性能优化实战篇](./05-性能优化实战篇.md)（回归门控方法论）· [13 进阶专题篇](./13-进阶专题篇.md)（entmax 推导）
