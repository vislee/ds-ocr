/*
 * ds_moe_decoder.c - DeepSeek-V2 MoE Decoder for DeepSeek-OCR
 * ds_moe_decoder.c — DeepSeek-V2 MoE 解码器实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【解码器单层前向传播详解】(详见 01-Transformer基础篇.md + 02-MoE混合专家篇.md)
 * ─────────────────────────────────────────────────────────────────────
 *
 * Dense层 (Layer 0):
 *   ┌──────────────────────────────────────────────────────┐
 *   │ x_norm = RMSNorm(x, input_norm)                      │  ← Pre-Norm
 *   │ q,k,v = Wq*x, Wk*x, Wv*x  (BF16→F32 on-the-fly)    │  ← QKV投影
 *   │ q = rms_norm_per_head(q, q_norm)  [V1/V2 only]      │  ← Per-head Q norm
 *   │ k = rms_norm_per_head(k, k_norm)  [V1/V2 only]      │  ← Per-head K norm
 *   │ apply_rope(q, k, pos)                                │  ← RoPE旋转位置编码
 *   │ cache_k[pos] = k, cache_v[pos] = v                   │  ← KV缓存写入
 *   │ attn_out = causal_attention(q, cache_k, cache_v)      │  ← 因果注意力
 *   │ proj_out = Wo @ attn_out                              │  ← 输出投影
 *   │ x = x + proj_out                                     │  ← 残差连接
 *   │ x_norm = RMSNorm(x, post_attn_norm)                  │
 *   │ gate = W_gate @ x_norm, up = W_up @ x_norm           │  ← Dense SwiGLU
 *   │ ffn_out = W_down @ (SiLU(gate) ⊙ up)                 │
 *   │ x = x + ffn_out                                      │  ← 残差连接
 *   └──────────────────────────────────────────────────────┘
 *
 * MoE层 (Layer 1~11):
 *   ┌──────────────────────────────────────────────────────┐
 *   │ (注意力部分同上，省略)                                │
 *   │ x_norm = RMSNorm(x, post_attn_norm)                  │
 *   │ ── MoE 前向 ──                                       │
 *   │ scores = gate_weight @ x_norm → softmax → top-6      │  ← 路由器
 *   │ ── 路由专家 (top-6) ──                               │
 *   │ for k=0..5:                                          │
 *   │   expert_id = top_indices[k]                         │
 *   │   gate_up = gate_up_fused[expert_id] @ x_norm        │  ← 融合gate+up
 *   │   gate, up = split(gate_up)                          │
 *   │   hidden = SiLU(gate) ⊙ up                           │  ← SwiGLU激活
 *   │   expert_out = down_weight[expert_id] @ hidden       │  ← Down投影
 *   │   output += top_weights[k] * expert_out              │  ← 加权组合
 *   │ ── 共享专家 (永远激活) ──                            │
 *   │ s_gate_up = shared_gate_up_fused @ x_norm            │  ← 2个共享专家一起算
 *   │ s_gate, s_up = split(s_gate_up)                      │
 *   │ s_hidden = SiLU(s_gate) ⊙ s_up                       │
 *   │ shared_out = shared_down @ s_hidden                   │
 *   │ output += shared_out                                 │  ← 共享专家贡献
 *   │ x = x + output                                       │  ← 残差连接
 *   └──────────────────────────────────────────────────────┘
 *
 * 【Decode vs Prefill 的实现差异】
 *   Decode (单token): matvec(BF16×F32) + 串行6个expert
 *   Prefill (多token): sgemm(F32×F32) + Gather-Compute-Scatter批量expert
 *
 * 【v0.5→v0.9 优化历程】
 *   - Argmax LM Head: 流式argmax，不分配129280个logits → 8×加速
 *   - 连续expert块: 所有expert在同一块内存 → 2.4× MoE加速
 *   - Fused gate+up: 1次matvec代替2次 → 减少x的内存读取
 *   - madvise预取: 异步预读下一个expert权重 → 减少page fault
 *   - Selective Repetition Penalty: 只重算历史中出现过的token → 减少惩罚开销
 *
 * Architecture:
 * - 12 transformer layers
 * - Layer 0: dense FFN (SwiGLU, intermediate=6848)
 * - Layers 1-11: MoE MLP (64 routed experts top-6 + 2 shared experts, moe_inter=896)
 * - Self-Attention: MHA (10 Q heads, 10 KV heads, head_dim=128)
 * - MoE: 64 routed experts (top-6) + 2 shared experts (always active)
 * - Expert SwiGLU: gate + up → SiLU → down
 * - Tied embeddings: tok_embeddings == lm_head
 */

#include "ds_moe_decoder.h"
#include "ds_dash.h"              /* V4 DASH 块稀疏注意力 */
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include "ds_metal.h"
#include "ds_quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

/* ========================================================================
 * Forward Declaration
 * ======================================================================== */
/* moe_forward — MoE 前向传播（单 token decode 路径）
 *
 * 处理顺序:
 *   1. Router: x_norm @ W_gate → softmax → top-6 选择
 *   2. Routed experts: 对 top-6 专家逐专家做 gate+up+SwiGLU+down
 *   3. Combine: 按路由权重加权求和 top-6 个 expert 输出
 *   4. Shared experts: 2个永久激活的专家，gate+up+SwiGLU+down
 *   5. Final: routed_output + shared_output
 *
 * ┌────────────────────────────────────────────────────────────────┐
 * │ 输入: x [hidden]                                               │
 * │  Router ─→ softmax over 64 ─→ top-6 indices + weights          │
 * │    │                                                            │
 * │    ├→ Expert[k=0]: gate_up_fused @ x → SwiGLU → down → w0 * out │
 * │    ├→ Expert[k=1]: ...                                         │
 * │    ├→ ...                                                      │
 * │    ├→ Expert[k=5]: ...                                         │
 * │    └→ Shared: gate_up_fused @ x → SiLU → down                  │
 * │    ┌─ 加权求和 ─────────────────────────────────────┐           │
 * │    │ routed_output = Σ(w_k * expert_out_k) + shared_out │       │
 * │    └───────────────────────────────────────────────────┘         │
 * │ 输出: routed_output + shared_output  [hidden]                    │
 * └────────────────────────────────────────────────────────────────┘
 *
 * 【CPU 与 GPU 双路径】
 * CPU: 串行逐 expert matvec，madvise 预取下一 expert 权重
 * GPU (Metal): 批量将 top-6 expert 提交 GPU 并行计算
 *
 * 【权重布局优化】（v0.9 设计；r2 任务3 已移除连续块拷贝，见下）
 * 所有 64 个 expert 的 gate_up_fused 权重放在一块连续内存
 * (expert_block_bf16)，共享 expert 放在尾部。这样一次 madvise
 * 预取整个块，减少 page fault 导致的随机读延迟。
 * 现状：拷贝因 mmap 源页双份驻留（峰值 RSS ~11-12GB）被移除，
 * expert_block_bf16/gate_up_fused_bf16 恒为 NULL，decode 改走
 * split 布局 + 分组行并行（见 moe_forward 实现体注释）。*/

/* moe_forward — MoE层的前向传播（单token decode路径）
 * 处理64个路由专家(top-6) + 2个共享专家的完整计算流程。
 * 所有buffer由调用者提供（复用ctx中的预分配内存），避免反复malloc/free。*/
static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_ctx_t *ctx,
                         float *expert_gate_buf, float *expert_up_buf,
                         float *expert_gate_up_buf, float *expert_hidden_buf,
                         float *shared_gate_buf, float *shared_up_buf,
                         float *shared_gate_up_buf, float *shared_swiglu_buf,
                         float *shared_out_buf,
                         float *expert_outputs_buf);

/* ========================================================================
 * BF16 → F32 Conversion Helpers
 * ======================================================================== */
/* 注意: 所有权重以 BF16 格式存储（通过 mmap 零拷贝加载），
 * 在 matmul/matvec 运算时逐元素转换为 F32（on-the-fly 转换）。
 * 这避免了预转换全部权重的 F32 内存开销（~1.6GB 额外）。
 * 解码器中的 F32 运算精度高于 Python 的 BF16 累积精度。
 * 当 ds_bf16_simulate_python=1 时，中间结果会被截断回 BF16 精度
 * 以匹配 Python 的 autocast("cuda", dtype=bfloat16) 行为。*/

/* ========================================================================
 * KV Cache Access Helpers — F32 direct storage
 * ======================================================================== */
/* KV 缓存结构: 一个紧凑的三维数组
 *   维度: [num_layers][kv_cache_max][kv_row_stride] (F32)
 *   kv_row_stride ≥ kv_dim，对齐到 64 字节边界以优化 SIMD 读取
 *   
 * 【设计选择: F32 缓存 → 无需解压缩】
 * 与 qwen-asr 不同（缓存 BF16，读取时批量 F32→BF16→F32 转换），
 * ds-ocr 将 K/V 直接以 F32 存储在缓存中。代价是缓存容量减半，
 * 但好处是每次解码步骤不需要逐位置转换 BF16→F32 —— 注意力计算
 * 时 KV 可直接以 F32 读取，省掉 O(seq_len) 的格式转换开销。
 * 对于解码器~300个KV位置，每步节省约2-3μs。
 *
 * 【V3 R-SWA 缓存策略】
 * V3 (Unlimited-OCR) 使用 R-SWA 注意力，KV 缓存包含两类:
 *   1. Reference tokens: [0..prefill_token_count-1] — 视觉token，永久保留
 *   2. Sliding window tokens: 最近 W=128 个文本 token
 * 缓存大小理论上限 = prefill_token_count + W，但当前实现仍用
 * append-only 模式（文本token持续追加），尚未实现环形缓冲区。*/

/* ========================================================================
 * KV Cache Access Helpers — F32 direct storage (no BF16 conversion)
 * 实现移至 ds_dash.h (ds_kv_k_row/ds_kv_v_row static inline)，供 V4 DASH 模块共用
 * ======================================================================== */

/* ds_kv_k_layer / ds_kv_v_layer - 取本层 K/V 缓存的基址（第 0 行）
 * 【算法】纯指针算术，不做任何数据搬运：
 *   addr = base + layer * kv_cache_max * stride * elem_size
 * 【参数】ctx  — 全局上下文（含 kv_cache_k/v、kv_cache_max、_kv_row_stride、kv_bf16）
 *         layer — 解码器层号 [0, dec_layers)
 * 【为什么】KV 缓存按 [layers][max_seq][kv_row_stride] 三维平铺，每层是一段
 *   连续 slab；单 token decode 每层都要从头扫 K/V，先拿到基址可避免在注意力
 *   内层循环里重复计算层间偏移。
 * 【为什么返回 void*】缓存元素宽度由 ctx->kv_bf16 决定（V1/V2=F32 4B，
 *   V3/V4=BF16 2B），调用方按需转型；行内定位用 ds_kv_k_row（ds_dash.h）。
 * 【参考】ds_dash.h 同名布局注释；docs/03-推理引擎实战篇 KV 缓存设计。 */
/* Get pointer to start of K/V cache for a layer (row 0) */
static inline void *ds_kv_k_layer(ds_ctx_t *ctx, int layer) {
    int stride = ctx->_kv_row_stride;
    size_t elem = ctx->kv_bf16 ? sizeof(uint16_t) : sizeof(float);
    return (char *)ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * stride * elem;
}

static inline void *ds_kv_v_layer(ds_ctx_t *ctx, int layer) {
    int stride = ctx->_kv_row_stride;
    size_t elem = ctx->kv_bf16 ? sizeof(uint16_t) : sizeof(float);
    return (char *)ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * stride * elem;
}

/* ds_kv_store_row - 把一个位置（一层）的 K 或 V 行写入缓存
 * 【算法】kv_bf16=0: 直接 memcpy F32（V1/V2 主路径，零转换）；
 *         kv_bf16=1: F32→BF16 批量截断（V3/V4，写入即量化省一半内存）。
 * 【参数】dst — 目标行地址（ds_kv_k_row/v_row 算出，已含层与位置偏移）
 *         src — 新算出的 K 或 V 向量 [kv_dim]，kv_dim = n_kv_heads*head_dim = 1280
 *         n   — 写入元素数（= kv_dim）
 * 【为什么】写入发生在 RoPE 之后：缓存里存的是"已带位置信息"的 K，
 *   decode 时注意力可直接点积，无需对缓存行再做旋转。
 * 【参考】docs/04-完整推理流程篇 §2（prefill/decode 写缓存时机）。 */
/* Store F32 K/V directly into the aligned F32 KV cache */
static inline void ds_kv_store_row(ds_ctx_t *ctx, void *dst, const float *src, int n) {
    if (ctx->kv_bf16) {
        ds_f32_to_bf16_batch((uint16_t *)dst, src, n);
    } else {
        memcpy(dst, src, (size_t)n * sizeof(float));
    }
}

/* ========================================================================
 * Dense FFN Forward: SwiGLU (Layer 0 only)
 * ======================================================================== */
/* dense_ffn_forward - Layer 0 稠密 FFN（SwiGLU）前向传播（decode 单 token / prefill 复用）
 * 【算法/数学】SwiGLU 三明治结构:
 *   gate = x @ W_gate^T            [1,1280] → [1,6848]
 *   up   = x @ W_up^T              [1,1280] → [1,6848]
 *   h    = SiLU(gate) ⊙ up         SiLU(x)=x·σ(x)，σ 为 sigmoid（逐元素）
 *   output = h @ W_down^T          [1,6848] → [1,1280]
 * 【参数】output      — 输出 [hidden=1280]
 *         x           — 输入（post-attn RMSNorm 之后激活）[hidden]
 *         layer       — 第 0 层权重（dense_gate/up/down_weight_bf16，BF16 mmap）
 *         gate/up/swiglu_buf — 调用者传入的中间缓冲 [intermediate=6848]；
 *                      传 NULL 时内部 malloc（prefill 批量路径偶发使用）
 * 【为什么 gate/up 不像 MoE 路径那样预融合】只有这一层是 dense，融合权重
 *   需要额外一份拷贝（gate+up 拼接），收益抵不过 3.4MB×2 的内存驻留；
 *   MoE 层 64 个专家逐一融合则收益显著（见 ds_ocr.h gate_up_fused_bf16 注释）。
 * 【为什么 Layer 0 用 Dense 而不是 MoE？】(dec_first_k_dense=1)
 *   第一个 transformer 层处理"原始"输入特征，需要全参数来提取通用特征；
 *   浅层语义尚未分化，路由器缺乏可靠信号，MoE 分工无从谈起。
 * 【参考】docs/01-Transformer基础篇 §7.3 SwiGLU；docs/02 §6.2 层分工；
 *   PyTorch 等价: DeepseekV2MLP（LlamaMLP 同构）。
 *
 * 实现要点:
 *   1. gate 和 up 分别做 BF16 matvec（非融合版本，因为 dense 只有一层）
 *   2. SwiGLU 使用 ds_swiglu_direct 直接计算（不经过 gate_up 交错排列）
 *   3. buffer 由调用者提供或内部临时分配（用于 prefill 批量模式）
 *
 * 为什么 Layer 0 用 Dense 而不是 MoE？
 * 第一个 transformer 层处理"原始"输入特征，需要全参数来提取通用特征。
 * 后续层(MoE)在通用特征基础上做专业化分工（路由到不同专家）。*/

static void dense_ffn_forward(float *output, const float *x,
                               ds_dec_layer_t *layer, ds_config_t *cfg,
                               float *gate_buf, float *up_buf, float *swiglu_buf) {
    int hidden = cfg->dec_hidden;
    int intermediate = cfg->dec_intermediate; /* 6848 */

    /* Allocate buffers if not provided (for prefill) */
    int own_gate = (gate_buf == NULL);
    int own_up = (up_buf == NULL);
    int own_swiglu = (swiglu_buf == NULL);
    if (own_gate) gate_buf = (float *)malloc(intermediate * sizeof(float));
    if (own_up) up_buf = (float *)malloc(intermediate * sizeof(float));
    if (own_swiglu) swiglu_buf = (float *)malloc(intermediate * sizeof(float));

    /* Gate + Up projections */
    ds_linear_nobias_bf16(gate_buf, x, layer->dense_gate_weight_bf16, 1, hidden, intermediate);
    ds_linear_nobias_bf16(up_buf, x, layer->dense_up_weight_bf16, 1, hidden, intermediate);

    /* BF16 truncation — matches the prefill dense path (and Python autocast).
     * The decode path previously skipped these, so layer-0 outputs drifted
     * from the prefill-truncated distribution that produced the first token. */
    {
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) {
            ds_bf16_truncate_array(gate_buf, intermediate);
            ds_bf16_truncate_array(up_buf, intermediate);
        }
    }

    /* SwiGLU: SiLU(gate) * up — direct computation without interleaving.
     * Previously: gate_up[2*i]=gate[i], gate_up[2*i+1]=up[i], then ds_swiglu_multiply.
     * That interleaving caused non-sequential writes (cache-unfriendly).
     * ds_swiglu_direct computes SiLU(gate[i]) * up[i] directly. */
    ds_swiglu_direct(swiglu_buf, gate_buf, up_buf, 1, intermediate);

    {
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) ds_bf16_truncate_array(swiglu_buf, intermediate);
    }

    /* Down projection */
    ds_linear_nobias_bf16(output, swiglu_buf, layer->dense_down_weight_bf16, 1, intermediate, hidden);

    {
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) ds_bf16_truncate_array(output, hidden);
    }

    if (own_gate) free(gate_buf);
    if (own_up) free(up_buf);
    if (own_swiglu) free(swiglu_buf);
}

/* mlp_forward - FFN 分派层（Dense 或 MoE）
 *
 * 【算法】按 layer_idx 与 dec_first_k_dense（=1）的比较分派:
 *   layer_idx < 1 → dense_ffn_forward （Layer 0，intermediate=6848 的稠密 SwiGLU）
 *   layer_idx >= 1 → moe_forward     （Layer 1~11，64 专家 top-6 + 2 共享）
 * 【参数】output — FFN 输出 [hidden=1280]；x — 输入 [hidden]（post-attn norm 后）
 *         layer_idx — 当前层号，其余 buffer 均为 ctx 预分配的复用缓冲
 * 【为什么】浅层 token 表示尚未语义分化，路由器无法给出有意义的专家分工，
 *   故第 0 层用全参数 Dense；从第 1 层起才切换 MoE（docs/02 §6.2）。
 * 【参考】PyTorch: DeepseekV2MoE 中 n_routed_experts/first_k_dense_replace 的对应逻辑。
 *
 * 所有 buffer 由调用者传入（decode 时复用 ctx 中的预分配内存），
 * 避免每步 malloc/free 的开销。*/
static void mlp_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_ctx_t *ctx, int layer_idx,
                         float *dense_gate_buf, float *dense_up_buf, float *dense_swiglu_buf,
                         float *expert_gate_buf, float *expert_up_buf,
                         float *expert_gate_up_buf, float *expert_hidden_buf,
                         float *shared_gate_buf, float *shared_up_buf,
                         float *shared_gate_up_buf, float *shared_swiglu_buf,
                         float *shared_out_buf,
                         float *expert_outputs_buf) {
    if (layer_idx < cfg->dec_first_k_dense) {
        /* Dense FFN (SwiGLU) */
        dense_ffn_forward(output, x, layer, cfg, dense_gate_buf, dense_up_buf, dense_swiglu_buf);
    } else {
        /* MoE */
        moe_forward(output, x, layer, cfg, ctx,
                    expert_gate_buf, expert_up_buf,
                    expert_gate_up_buf, expert_hidden_buf,
                    shared_gate_buf, shared_up_buf,
                    shared_gate_up_buf, shared_swiglu_buf,
                    shared_out_buf,
                    expert_outputs_buf);
    }
}

/* moe_forward - MoE 层前向传播，单 token decode 路径（实现体）
 * 完整算法/数据流图见文件顶部前置声明处的文档块，此处只列执行序:
 *   Step 1  路由: BF16 gate matvec → 全局 softmax(64) → 无放回选 top-6
 *   Step 2  路由专家（按可用性自动分派）:
 *           Metal GPU 批量 → 分组行并行(快路径) → 旧连续块 madvise + 逐专家
 *           → 逐专家 split/INT4（当前构建的常态: split mmap 布局）
 *   Step 3  ds_expert_combine: output = Σ w_k · expert_out_k
 *   Step 4  共享专家: INT4 → Metal → fused BF16 → split BF16，输出累加进 output
 * 【维度】x/output: [1280]；每专家 gate/up: [896,1280]、down: [1280,896]；
 *         共享专家两份拼一起算: [1792,1280]/[1280,1792]
 * 【参考】docs/02-MoE混合专家篇 §9 完整追踪；docs/05-性能优化实战篇（优化历程）。
 */
static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_ctx_t *ctx,
                         float *expert_gate_buf, float *expert_up_buf,
                         float *expert_gate_up_buf, float *expert_hidden_buf,
                         float *shared_gate_buf, float *shared_up_buf,
                         float *shared_gate_up_buf, float *shared_swiglu_buf,
                         float *shared_out_buf,
                         float *expert_outputs_buf) {
    int hidden = cfg->dec_hidden;
    int moe_inter = cfg->dec_moe_inter;
    int n_experts = cfg->dec_n_routed_experts;
    int top_k = cfg->dec_top_k;
    int n_shared = cfg->dec_n_shared_experts;

    /* Step 1: Router gate scores
     * 路由器是一个 [64,1280] 的小线性层：scores[e] = x·W_gate[e]。
     * 理论上每 token 只有 top-6/64 专家被激活 → 激活参数 ~570M / 总参 3B，
     * 这就是 MoE "稀疏激活"带来的容量-算力解耦（见 docs/02-MoE混合专家篇）。 */
    float scores_buf[DS_MAX_EXPERTS];          /* 栈上 64×4B=256B，免 malloc */
    float *scores = (n_experts <= DS_MAX_EXPERTS) ? scores_buf :
                    (float *)malloc(n_experts * sizeof(float));
    /* Use BF16 gate weight to match Python's BF16 precision.
     * F32 gate weight can select different experts than Python BF16,
     * causing accuracy degradation especially for long sequences. */
    ds_moe_router_bf16(scores, x, layer->gate_weight_bf16, hidden, n_experts);
    /* 关键：路由用 BF16 权重。F32 路由可能与 Python 选出不同专家——
     * 排名相近的两个专家在 F32/BF16 下次序可交换，一旦换路输出就系统性偏离 */

    /* Step 2: Softmax over all experts first, then select top-K (matching Python).
     * Previously ds_moe_top_k did softmax over only top-K, which was wrong —
     * the softmax values should be computed over all 64 experts before selecting.
     * 【为什么先全局 softmax 再 top-6】softmax 的分母含全部 64 个专家，
     *   先选后归一（ds_moe_top_k 的旧做法）权重会系统性偏大，
     *   与 Python 参考实现的输出不可对齐。
     * 【注意：此处权重不再 renormalize】top_weights 直接取全局 softmax
     *   概率，6 个权重之和 < 1（剩余概率质量留给未选中专家）——
     *   与 docs/02 §3.2 伪代码的 top_k_probs/sum(top_k_probs) renorm 步骤
     *   不同，以代码为准（DeepSeek-V2 风格）。softmax 内嵌 max-subtract
     *   防 e^s 上溢，与 docs/13 §1.1 的数值稳定技巧一致。 */
    {
        float max_s = -1e30f;                       /* 数值稳定：减去最大值防 exp 上溢 */
        for (int e = 0; e < n_experts; e++)
            if (scores[e] > max_s) max_s = scores[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < n_experts; e++) {
            scores[e] = ds_fast_expf(scores[e] - max_s);  /* softmax 分子 e^(s-max) */
            sum_exp += scores[e];
        }
        float inv = 1.0f / sum_exp;                 /* 归一化 Σp_e = 1 */
        for (int e = 0; e < n_experts; e++)
            scores[e] *= inv;
    }
    int top_indices[DS_MAX_EXPERTS];
    float top_weights[DS_MAX_EXPERTS];
    for (int k = 0; k < top_k; k++) {               /* 选择法取 top-6：64×6 次比较，~微秒级 */
        int best = -1; float best_score = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            int already = 0;
            for (int j = 0; j < k; j++)             /* 跳过已入选的专家（无放回 top-k） */
                if (top_indices[j] == e) { already = 1; break; }
            if (already) continue;
            if (scores[e] > best_score) { best_score = scores[e]; best = e; }
        }
        top_indices[k] = best;      /* 第 k 名专家的编号 */
        top_weights[k] = best_score;/* 其 softmax 概率——加权组合系数。
                                     * 不做 renormalize（不再除以 6 项之和），
                                     * selection 天然按概率降序产出 top_indices */
    }
    if (scores != scores_buf) free(scores);

    /* Step 3: Forward through each selected routed expert.
     * Metal GPU path: batch all experts on GPU for massive parallelism.
     * CPU path: sequential per-expert matvec with prefetch optimization.
     *
     * 【Metal 触发条件（四者同时满足）】
     *   1. ctx->metal_ctx 非空 —— Metal 设备初始化成功；
     *   2. ctx->metal_moe_env  —— 启动时检测到 DS_METAL_MOE 环境变量
     *      （getenv 提升到加载期，避免每 token 反复查询）；
     *   3. ds_metal_is_available —— GPU 通道可用；
     *   4. 第一个选中专家的 gate_up_fused_bf16 非空 —— GPU 内核按
     *      "gate_up 拼接布局" 编写，split 布局无法提交。
     * 任一不满足 → 回退 CPU 路径（无任何降级提示，属正常分派而非错误）。
     * r2 任务3 移除连续块拷贝后 gate_up_fused_bf16 恒为 NULL，
     * 此 GPU 路径当前处于休眠状态，保留以支持未来重建 fused 布局。 */
    float *expert_outputs = expert_outputs_buf ? expert_outputs_buf :
                            (float *)calloc(top_k * hidden, sizeof(float));

    if (ctx->metal_ctx && ctx->metal_moe_env && ds_metal_is_available(ctx->metal_ctx) &&
        layer->experts[top_indices[0]].gate_up_fused_bf16) {   /* 无连续块（r2 任务3）→ CPU split 路径 */
        /* ── Metal GPU path: MoE expert forward ── */
        const uint16_t *gate_up_ptrs[DS_MAX_EXPERTS];
        const uint16_t *down_ptrs[DS_MAX_EXPERTS];
        for (int k = 0; k < top_k; k++) {
            gate_up_ptrs[k] = layer->experts[top_indices[k]].gate_up_fused_bf16;
            down_ptrs[k] = layer->experts[top_indices[k]].down_weight_bf16;
        }
        /* GPU computes combined expert output directly into 'output' */
        ds_metal_moe_experts(ctx->metal_ctx, x, gate_up_ptrs, down_ptrs,
                              top_weights, top_k, hidden, moe_inter, output);
        /* Skip CPU combine since GPU already combines */
        if (!expert_outputs_buf) free(expert_outputs);
        goto shared_experts;
    }

    /* ── CPU path: 分组行并行（r4 优化）──
     * 旧实现: 每个选中 expert 独立调用 2-3 次 matvec（split 布局 gate/up/down
     * 共 3 次），每层 ~19 个微小型并行区——barrier 同步开销大于矩阵本身，
     * 实测 t1 与 t14 解码耗时相同（MoE 7.7s vs 8.4s，多线程反而 +10% 纯开销）。
     * 现在: 同类投影的行合并为 1 个并行区——
     *   split 布局 (V4): gate / up / down 各 1 个并行区，共 3 个；
     *   fused 布局:      gate_up 1 个 + down 1 个。
     * 行对（2r, 2r+1）永不跨线程拆分，每行仍由同一 NEON 内核按成对路径累加
     * → 输出与逐专家串行版本逐位一致（md5 已验证）。 */
    /* 布局探测：top_k>1 且未量化才可走分组快路径（6 个专家的行合并成一个
     * 线程池并行区）。INT4 权重是分块量化格式，无法被 BF16 分组内核消费，
     * 故 int4_enabled 时直接落逐专家路径。 */
    int all_fused = (top_k > 1) && !layer->int4_enabled;
    int all_split = (top_k > 1) && !layer->int4_enabled;
    for (int k = 0; (all_fused || all_split) && k < top_k; k++) {
        const uint16_t *fused = layer->experts[top_indices[k]].gate_up_fused_bf16;
        const uint16_t *gw = layer->experts[top_indices[k]].gate_weight_bf16;
        const uint16_t *uw = layer->experts[top_indices[k]].up_weight_bf16;
        if (!layer->experts[top_indices[k]].down_weight_bf16) {
            all_fused = all_split = 0;
        } else if (fused) {
            all_split = 0;      /* 该专家 fused → 混合布局走旧路径 */
        } else if (gw && uw) {
            all_fused = 0;      /* 该专家 split */
        } else {
            all_fused = all_split = 0;
        }
    }

    if (all_fused || all_split) {
        const uint16_t *w_a[DS_MAX_EXPERTS];   /* fused: gate_up / split: gate */
        const uint16_t *w_b[DS_MAX_EXPERTS];   /* split: up */
        const uint16_t *w_down[DS_MAX_EXPERTS];
        const float *x_shared[DS_MAX_EXPERTS]; /* gate/up: 各单元共享同一 x */
        const float *x_hidden[DS_MAX_EXPERTS]; /* down: 各专家的 SwiGLU 输出 */
        for (int k = 0; k < top_k; k++) {
            int expert_id = top_indices[k];
            if (all_fused) {
                w_a[k] = layer->experts[expert_id].gate_up_fused_bf16;
            } else {
                w_a[k] = layer->experts[expert_id].gate_weight_bf16;
                w_b[k] = layer->experts[expert_id].up_weight_bf16;
            }
            w_down[k] = layer->experts[expert_id].down_weight_bf16;
            x_shared[k] = x;
            /* 预取本专家全部权重（原实现为循环内预取下一个专家） */
            size_t inter_bytes = (size_t)hidden * moe_inter * 2;
            madvise((void *)w_down[k], inter_bytes, MADV_WILLNEED);
            if (all_fused) {
                madvise((void *)w_a[k], 2 * inter_bytes, MADV_WILLNEED);
            } else {
                madvise((void *)w_a[k], inter_bytes, MADV_WILLNEED);
                madvise((void *)w_b[k], inter_bytes, MADV_WILLNEED);
            }
        }

        /* gate_up_all 复用为 split 布局的 gate_all + up_all 两段 */
        float *gate_up_all = ctx->moe_grouped_gate_up_buf;   /* [top_k × 2*inter] */
        float *hidden_all = ctx->moe_grouped_hidden_buf;     /* [top_k × inter] */
        if (gate_up_all && hidden_all) {
            extern int ds_bf16_simulate_python;
            float *up_all = gate_up_all + (size_t)top_k * moe_inter;
            for (int k = 0; k < top_k; k++)
                x_hidden[k] = hidden_all + (size_t)k * moe_inter;

            if (all_fused) {
                /* Phase 1: 全部专家的 gate_up 投影（一个并行区） */
                ds_grouped_matvec_bf16(gate_up_all, x_shared, w_a, top_k,
                                        2 * moe_inter, hidden);
            } else {
                ds_grouped_matvec_bf16(gate_up_all, x_shared, w_a, top_k,
                                        moe_inter, hidden);
                ds_grouped_matvec_bf16(up_all, x_shared, w_b, top_k,
                                        moe_inter, hidden);
            }

            /* 逐专家: 截断 gate/up → SwiGLU → 截断（与旧路径相同的函数与顺序，
             * 元素级操作，串行 ~20μs） */
            for (int k = 0; k < top_k; k++) {
                float *gu, *uu;
                float *hbuf = hidden_all + (size_t)k * moe_inter;
                if (all_fused) {
                    gu = gate_up_all + (size_t)k * 2 * moe_inter;
                    uu = gu + moe_inter;
                } else {
                    gu = gate_up_all + (size_t)k * moe_inter;
                    uu = up_all + (size_t)k * moe_inter;
                }
                if (ds_bf16_simulate_python) {
                    ds_bf16_truncate_array(gu, moe_inter);
                    ds_bf16_truncate_array(uu, moe_inter);
                }
                ds_swiglu_direct(hbuf, gu, uu, 1, moe_inter);
                if (ds_bf16_simulate_python)
                    ds_bf16_truncate_array(hbuf, moe_inter);
            }

            /* Phase 2: 全部专家的 down 投影（一个并行区）→ 截断 → 加权合并 */
            ds_grouped_matvec_bf16(expert_outputs, x_hidden, w_down, top_k,
                                    hidden, moe_inter);
            if (ds_bf16_simulate_python)
                ds_bf16_truncate_array(expert_outputs, top_k * hidden);
            goto moe_combine;
        }
        /* scratch 分配失败 → 落回逐专家路径 */
    }

    if (layer->expert_block_bf16 && top_k > 1) {
        /* 【旧连续块预取路径（当前休眠）】r2 任务3 前所有专家的 gate_up_fused
         * 被拷贝进一块连续内存（expert_block_bf16），此处只 madvise 选中
         * 专家覆盖的 [min_off, max_end) 区间（而非整块 ~400MB），把 6 个
         * 专家的页一次性预读进 page cache。块拷贝已随 r2 任务3 移除，
         * expert_block_bf16 恒为 NULL，分支保留以兼容旧布局。 */
        /* Find min/max offset of selected experts within the block */
        size_t fused_per_expert = (size_t)2 * moe_inter * hidden * sizeof(uint16_t);
        uintptr_t block_base = (uintptr_t)layer->expert_block_bf16;
        uintptr_t min_off = (uintptr_t)layer->experts[top_indices[0]].gate_up_fused_bf16 - block_base;
        uintptr_t max_end = min_off + fused_per_expert;
        for (int k = 1; k < top_k; k++) {
            uintptr_t off = (uintptr_t)layer->experts[top_indices[k]].gate_up_fused_bf16 - block_base;
            if (off < min_off) min_off = off;
            uintptr_t end = off + fused_per_expert;
            if (end > max_end) max_end = end;
        }
        madvise((void *)(block_base + min_off), max_end - min_off, MADV_WILLNEED);
    }
    for (int k = 0; k < top_k; k++) {
        int expert_id = top_indices[k];   /* 依路由概率降序访问：概率高的专家权重更可能已驻留缓存 */
        /* Prefetch next expert's weights (mmap'd region; r2 任务3 后 gate/up 也走 mmap) */
        if (k + 1 < top_k) {
            int next_id = top_indices[k + 1];
            size_t inter_bytes = (size_t)hidden * moe_inter * 2;
            madvise((void *)layer->experts[next_id].gate_weight_bf16, inter_bytes, MADV_WILLNEED);
            madvise((void *)layer->experts[next_id].up_weight_bf16, inter_bytes, MADV_WILLNEED);
            madvise((void *)layer->experts[next_id].down_weight_bf16, inter_bytes, MADV_WILLNEED);
        }
        /* Use INT4 path if quantized weights available, otherwise BF16.
         * 【INT8("INT4"是历史旗标名) scale 处理】ds_int4_block_t 按"行"量化：
         *   每行存 int8 索引 + 独立 scale[r]/offset[r]，matvec 时逐行现场
         *   反量化 W[r,j] = scale[r]·q + offset[r] 后与 x 累加——
         *   精度敏感的缩放留在 F32，只有权重本体降为 1 字节/元素
         *   （~2× 小于 BF16，见 ds_quantize.h 头注释）。
         * 【激活条件】gate_up_fused 的 INT8 块必须存在；r2 任务3 移除 fused
         *   源后该块不再构建（qweight=NULL），此路径随之休眠——同层的
         *   down_weight INT8 块虽仍会被 ds_quantize_moe_int4 生成，但无入口。 */
        if (layer->int4_enabled && layer->experts_int4[expert_id].gate_up_fused.qweight) {
            ds_expert_forward_fused_int4(expert_outputs + k * hidden, x,
                                          &layer->experts_int4[expert_id].gate_up_fused,
                                          &layer->experts_int4[expert_id].down_weight,
                                          hidden, moe_inter,
                                          expert_gate_up_buf, expert_gate_buf,
                                          expert_up_buf, expert_hidden_buf);
        } else if (layer->experts[expert_id].gate_up_fused_bf16) {
            ds_expert_forward_fused(expert_outputs + k * hidden, x,
                                     layer->experts[expert_id].gate_up_fused_bf16,
                                     layer->experts[expert_id].down_weight_bf16,
                                     hidden, moe_inter,
                                     expert_gate_up_buf, expert_gate_buf,
                                     expert_up_buf, expert_hidden_buf);
        } else {
            ds_expert_forward(expert_outputs + k * hidden, x,
                              layer->experts[expert_id].gate_weight_bf16,
                              layer->experts[expert_id].up_weight_bf16,
                              layer->experts[expert_id].down_weight_bf16,
                              hidden, moe_inter,
                              expert_gate_buf, expert_up_buf,
                              expert_gate_up_buf, expert_hidden_buf);
        }
    }

    /* Step 4: 加权合并路由专家的输出（分组快路径经 goto moe_combine 汇合至此）
     * GPU 路径已在 Metal 内核中合并，直接 goto shared_experts 跳过这里。 */
moe_combine:
    /* ds_expert_combine 先 memset 再累加:
     *   output[h] = Σ_{k=0..5} top_weights[k] * expert_outputs[k*hidden + h]
     * 加权系数即 Step 2 的全局 softmax 概率（未 renormalize）。 */
    ds_expert_combine(output, expert_outputs, top_indices, top_weights, top_k, hidden);
    if (!expert_outputs_buf) free(expert_outputs);
    else memset(expert_outputs_buf, 0, top_k * hidden * sizeof(float));
    /* caller 的 moe_expert_outputs 是普通 malloc（ds_ocr.c:1219），这里清零
     * 恢复"缓冲区进入 moe_forward 时恒为零"的 calloc 不变量——Metal 路径
     * 跳过 combine 时不写该缓冲，清零属于防御性维护（当前 CPU 各路径在
     * combine 前都会整块覆写，读位置不依赖此清零）。 */

shared_experts:
    /* Step 5: 共享专家前向传播
     * 共享专家永远激活（不受路由选择影响），处理通用知识。
     * DeepSeek-V2 每层有 2 个共享专家（共享 gate + 共享 up/down），
     * 隐含维度 = 2 * moe_inter = 2 * 896 = 1792。
     *
     * 有 4 种实现路径，按优先级排列:
     *   INT4 → Metal GPU → Fused BF16 (gate+up 合并) → 分离 BF16 (gate, up 分开)
     *
     * 【为什么共享专家不需要路由？】
     * 路由专家的设计意图是"专业化分工"——每个专家处理不同类型的输入。
     * 共享专家的设计意图是"通用知识"——所有输入都需要的基础能力（如语法、格式）。
     * 实验表明，共享专家显著提高了训练效率和模型质量。*/
    if (layer->int4_enabled && layer->shared_gate_up_int4.qweight) {
        /* ── INT4 path for shared experts ──
         * 逐行 scale/offset 反量化（同上方路由专家 INT8 路径）；
         * 源是 shared_gate_up_fused_bf16，r2 任务3 后源缺失 → 块不构建，
         * 此分支当前休眠，激活时 shared_inter = 2*896 = 1792 一次算完。 */
        int shared_inter = n_shared * moe_inter;
        ds_expert_forward_fused_int4(shared_out_buf, x,
                                      &layer->shared_gate_up_int4,
                                      &layer->shared_down_int4,
                                      hidden, shared_inter,
                                      shared_gate_up_buf, shared_gate_buf,
                                      shared_up_buf, shared_swiglu_buf);
        ds_vec_add(output, output, shared_out_buf, hidden);  /* 共享专家输出直接累加进残差 */
    } else if (ctx->metal_ctx && ctx->metal_moe_env && ds_metal_is_available(ctx->metal_ctx) &&
        layer->shared_gate_up_fused_bf16 && layer->shared_down_weight_bf16) {
        /* ── Metal GPU path: shared experts ──
         * 同样要求 fused 布局（GPU 内核按 gate_up 拼接编写）；
         * fused 指针恒为 NULL → 实际回落到下方 split BF16 路径。 */
        int shared_inter = n_shared * moe_inter;
        ds_metal_shared_experts(ctx->metal_ctx, x,
                                 layer->shared_gate_up_fused_bf16,
                                 layer->shared_down_weight_bf16,
                                 hidden, shared_inter, output);
    } else if (layer->shared_gate_up_fused_bf16) {
        /* Fused gate+up path: single matvec [hidden → 2*shared_inter]
         * 融合优势：一次 matvec 读一遍 x（1280 float 复用两次投影），权重组件也连续 */
        int shared_inter = n_shared * moe_inter;
        ds_linear_nobias_bf16(shared_gate_up_buf, x, layer->shared_gate_up_fused_bf16,
                               1, hidden, 2 * shared_inter);
        /* Split into gate and up —— fused 布局约定：前半 gate，后半 up */
        memcpy(shared_gate_buf, shared_gate_up_buf, shared_inter * sizeof(float));
        memcpy(shared_up_buf, shared_gate_up_buf + shared_inter, shared_inter * sizeof(float));

        /* BF16 truncate (match Python precision) */
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) {
            ds_bf16_truncate_array(shared_gate_buf, shared_inter);
            ds_bf16_truncate_array(shared_up_buf, shared_inter);
        }

        /* Direct SwiGLU */
        ds_swiglu_direct(shared_swiglu_buf, shared_gate_buf, shared_up_buf, 1, shared_inter);

        if (ds_bf16_simulate_python) {
            ds_bf16_truncate_array(shared_swiglu_buf, shared_inter);
        }

        ds_linear_nobias_bf16(shared_out_buf, shared_swiglu_buf, layer->shared_down_weight_bf16,
                               1, shared_inter, hidden);

        if (ds_bf16_simulate_python) {
            ds_bf16_truncate_array(shared_out_buf, hidden);
        }

        ds_vec_add(output, output, shared_out_buf, hidden);
    } else if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
        /* Fallback: separate gate + up (for prefill or if fused not available)
         * 【当前构建的实际路径】权重直接取 mmap 的 split 布局原生指针，
         * 与 decode 路由专家的数学逐元素一致；2 个共享专家的 gate/up/down
         * 各是一个 [1792,1280]/[1280,1792] 大矩阵，一次 matvec/sgemm 覆盖两者。 */
        int shared_inter = n_shared * moe_inter;

        ds_linear_nobias_bf16(shared_gate_buf, x, layer->shared_gate_weight_bf16, 1, hidden, shared_inter);
        ds_linear_nobias_bf16(shared_up_buf, x, layer->shared_up_weight_bf16, 1, hidden, shared_inter);

        /* Direct SwiGLU — no gate_up interleaving needed */
        ds_swiglu_direct(shared_swiglu_buf, shared_gate_buf, shared_up_buf, 1, shared_inter);

        ds_linear_nobias_bf16(shared_out_buf, shared_swiglu_buf, layer->shared_down_weight_bf16,
                               1, shared_inter, hidden);

        ds_vec_add(output, output, shared_out_buf, hidden);
    }
}

/* ========================================================================
 * Decoder Layer Forward Pass (single token)
 * ======================================================================== */

/* decoder_layer_forward - 解码器单层前向传播（单 token decode 路径）
 * 【算法/数学】Pre-Norm transformer 层，维度以 hidden=1280/head_dim=128 标注:
 *   x_norm = RMSNorm(x, input_norm)                    [1280] → [1280]
 *   q,k,v  = Wq/Wk/Wv @ x_norm                          → q[1280] k,v[1280]
 *            （[1280] 按头切开视作 [10,128]）
 *   q,k    = per-head RMSNorm(q,k) · RoPE(pos)          ← 逐头归一 + 旋转
 *   cache_k/v[l][pos] = k,v                             ← KV cache 写入
 *   attn   = softmax(q·Kᵀ/√128 + mask)·V  (在线softmax)  → [1280]
 *   out    = x + Wo @ attn                              ← 注意力残差
 *   x_norm = RMSNorm(out, post_attn_norm)
 *   out    = out + MLP(x_norm)   （L0 Dense / L1-11 MoE）← FFN 残差
 * 【参数】x/out — 输入与输出 [hidden=1280]（调用方用双缓冲交替）；layer_idx — 层号；
 *         pos — 本 token 的全局位置（= kv_cache_len，用于 RoPE 表与 KV 写入行）
 * 【为什么 Pre-Norm】归一化放在子层之前，残差主干上没有 norm，梯度可以
 *   无衰减直达浅层，训练/深网络更稳定（Post-Norm 每层都过 norm 会放大
 *   波动）；现代 LLM（Llama/Qwen/DeepSeek）一律 Pre-Norm。
 * 【为什么 decode 用 matvec 而 prefill 用 sgemm】seq_len=1 时算术强度极低，
 *   瓶颈在权重内存带宽，matvec(BF16 on-the-fly) 免去整块 F32 权重转换；
 *   prefill 多行摊薄权重读取，cblas_sgemm 吞吐远高于逐行 matvec。
 * 【参考】docs/01 §5-§8；docs/04 §2.6-2.7；PyTorch: DeepseekV2DecoderLayer。
 */
static void decoder_layer_forward(ds_ctx_t *ctx, const float *x, float *out,
                                   int layer_idx, int pos) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    ds_dec_layer_t *layer = &dec->layers[layer_idx];

    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    /* softmax 前的 1/√head_dim 缩放：q·k 是 128 项乘积求和，分量独立零均值时
     * Var(q·k) = d·Var(q)·Var(k)，logit 波动随 √d 增长会把 softmax 推进饱和区
     * （梯度趋零）；除以 √128 把方差拉回 O(1)（推导见 docs/13 §1.5）。
     * 本模型 MHA 无 GQA，n_kv_heads == n_heads == 10。 */
    float scale = 1.0f / sqrtf((float)head_dim);

    double t_layer_start = 0, t0 = 0;
    if (ctx->profile_enabled) t_layer_start = ds_time_sec();

    /* 输入 RMSNorm: Pre-Norm 模式
     * Transformer 的前置归一化（Pre-LayerNorm），在注意力/FFN 之前做归一化。
     * Pre-Norm 相比 Post-Norm 更稳定，训练时可以使用更大的学习率。
     * 公式: x_norm = x / sqrt(mean(x²) + eps) * weight
     *   eps=1e-6（dec_rms_norm_eps）防止零向量除零；mean(x²) 是平方均值
     *   （不做均值中心化——这是与 LayerNorm 的唯一数学差异，省一趟求和）；
     *   weight 是 RMSNorm 的缩放参数（一个可学习的向量，F32 存储）。
     * 注意残差取的是原始 x，norm 只喂给 QKV 投影——Pre-Norm 的标志。*/
    float *x_norm = ctx->dec_x_norm;
    ds_rms_norm(x_norm, x, layer->input_norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* QKV projections (fused single-token matvec)
     * 【为什么三合一】Q/K/V 共享同一个输入 x_norm，融合入口让一次并行区
     *   调度同时覆盖 Wq[1280,1280]+Wk[1280,1280]+Wv[1280,1280] 三块 BF16
     *   权重的行切分，省两次线程池 barrier；BF16→F32 在 matvec 内逐元素
     *   完成（on-the-fly），权重永不整体转换。 */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;

    ds_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                layer->wq_weight_bf16,
                                layer->wk_weight_bf16,
                                layer->wv_weight_bf16,
                                hidden, q_dim, kv_dim);

    /* Per-head Q/K RMSNorm (DeepSeek-V2 V1 style; V2/OCR-2 does not use these)
     * 【QK-Norm 动机】训练后期个别通道的 q/k 幅值会持续膨胀，点积 logits
     *   随之爆炸、注意力过早饱和。对每个 head 的 128 维段独立做 RMSNorm
     *   （q_norm/k_norm 各 [head_dim=128]，10 个头共享同一份缩放向量），
     *   把每个头的 Q/K 钉回单位 RMS——与 1/√head_dim 缩放构成双保险
     *   （docs/13 §1.5）。QK-Norm 的做法最早见于 ViT-22B，Qwen3 等沿用。
     * 【顺序】norm 必须在 RoPE 之前：RoPE 是正交旋转，若先旋转再逐通道
     *   乘 g，则 diag(g)² 与旋转矩阵不可交换，Q·K 的点积不再只依赖
     *   相对位置 m-n，RoPE 的相对位置性质被破坏。 */
    if (layer->q_norm_weight)   /* q_norm: [128] 每 head 一份共享缩放，10 个 head 各自归一 */
        ds_rms_norm_per_head(q, layer->q_norm_weight, 1, n_heads, head_dim, cfg->dec_rms_norm_eps);
    if (layer->k_norm_weight)   /* k_norm 同理作用于 10 个 KV head；K 归一化使注意力分数量级稳定 */
        ds_rms_norm_per_head(k, layer->k_norm_weight, 1, n_kv_heads, head_dim, cfg->dec_rms_norm_eps);

    /* RoPE: 对 Q 和 K 应用旋转位置编码（NeoX/Llama 风格 split-half 旋转）
     * 【角度公式】第 i 对频率 ω_i = theta^(-2i/head_dim)（theta=10000），
     *   位置 pos 的旋转角 θ_pos,i = pos·ω_i；低维 ω 大转得快（分辨相邻
     *   位置），高维 ω 小转得慢（编码远距离）。
     * 【哪两维配对】NeoX 风格是"前后半配对"：维度 d 与 d+64（half=64）
     *   组成一个旋转对，out[d] = x[d]·cos[d] - x[d+64]·sin[d]，
     *   out[d+64] = x[d]·sin[d] + x[d+64]·cos[d]，cos/sin 表两半重复。
     *   注意这与 docs/01 §4.3"每对相邻维度"（GPT-J 交错配对）的描述不同，
     *   以代码（ds_apply_rope_neox 的 rotate_half 实现）为准。
     *   旋转后 Q_m·K_n 只依赖相对位置 m-n——注意力天然感知相对距离。
     *
     * 预计算的 cos/sin 值存储在 ctx->rope_cache_cos/sin 中，
     * 预先计算好所有位置（直到 max_seq_len）的三角函数值，
     * 避免每步重复计算 sin/cos。
     *
     * V4 (DASH): 换用 page2d 二维 RoPE 表 (ds_dash 维护)——
     * tok 轴页内计数 + page 轴页号 mod 2，是官方 sparse_main_rope_mode="page2d"。*/
    float *cos_vals, *sin_vals;
    if (cfg->model_version == 4) {
        /* ds_dash_rope_row 原型在 ds_dash.h（const float *）；此处只读，无需强转 */
        int pos_v4 = ctx->kv_cache_len;
        cos_vals = (float *)ds_dash_rope_row(ctx, pos_v4, 0);
        sin_vals = (float *)ds_dash_rope_row(ctx, pos_v4, 1);
    } else {
        cos_vals = ctx->rope_cache_cos + pos * head_dim;  /* 位置 pos 的 cos 表项 [128]（两半相同） */
        sin_vals = ctx->rope_cache_sin + pos * head_dim;  /* sin 表项 [128]；两表启动时一次算好 */
    }
    ds_apply_rope_neox(q, cos_vals, sin_vals, 1, n_heads, head_dim);   /* Q: 10 个 head 一起旋转 */
    ds_apply_rope_neox(k, cos_vals, sin_vals, 1, n_kv_heads, head_dim);/* K: 同一位置旋转，保证 Q·K 相对相位正确 */

    if (ctx->profile_enabled) ctx->perf_layer_qkv_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;

    /* Store K, V directly into F32 KV cache (no BF16 conversion needed).
     * This eliminates the per-step batch BF16→F32 conversion that previously
     * dominated attention time (O(seq_len) conversion per decode step).
     * 【写入位置】行地址 = 层基址 + pos×stride；行内按头平铺：
     *   head h 占 [h*128, (h+1)*128)。本模型 MHA（10 Q 头对 10 KV 头），
     *   Q 头 h 读 KV 头 h（GQA 映射为恒等；若为 GQA 则 h/g 取商）。 */
    int cache_offset = ctx->kv_cache_len;                        /* 当前 token 写入的缓存槽位 */
    void *cache_k_row = ds_kv_k_row(ctx, layer_idx, cache_offset);  /* K 行地址 = base + (layer*max_seq+pos)*stride */
    void *cache_v_row = ds_kv_v_row(ctx, layer_idx, cache_offset);
    ds_kv_store_row(ctx, cache_k_row, k, kv_dim);   /* F32 直存（V1/V2）；BF16 存储（V3/V4，r2 任务2） */
    ds_kv_store_row(ctx, cache_v_row, v, kv_dim);

    /* Causal attention — read directly from F32 KV cache (zero conversion).
     * The cache is aligned and contiguous for optimal sequential read.
     * K/V rows are at stride _kv_row_stride (≥ kv_dim, aligned to 64 bytes).
     *
     * 【因果掩码的实现】decode（seq_q=1）时无需显式掩码矩阵：query 的全局
     *   位置是 q_offset=cache_offset，内核只扫 key 行 j∈[0, q_offset]，
     *   "看不到未来"由扫描上界 k_end=q_offset+1 隐式保证。
     * 【online softmax】内核不物化 [1,kv_seq_len] 分数行，而是一边点积一边
     *   维护运行最大值 max_s 与指数和 sum_e，新最大值出现时按
     *   corr=e^(m_old-m') 修正已累加的 O——与 FlashAttention 同一数学、
     *   块大小退化为 1（推导与算例见 docs/13 §2.2-§2.4）。
     *
     * R-SWA (Reference Sliding Window Attention, Unlimited-OCR):
     * During decode (cache_offset >= prefill_token_count), only attend to:
     *   1. Reference tokens: positions [0..prefill_token_count-1] (visual + prompt tokens)
     *   2. Recent text tokens: positions [window_start..kv_cache_len]
     * where window_start = max(prefill_token_count, kv_seq_len - sliding_window_size)
     * During prefill (cache_offset < prefill_token_count), use full causal attention. */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *attn_out = ctx->dec_attn_out;            /* 注意力输出 [1280] */
    float *k_base = ds_kv_k_layer(ctx, layer_idx);  /* 本层 K 缓存基址（读全部历史位置） */
    float *v_base = ds_kv_v_layer(ctx, layer_idx);  /* 本层 V 缓存基址 */
    int kv_seq_len = cache_offset + 1;              /* 可见长度 = 历史缓存 + 当前 token（因果掩码） */
    int kv_stride = ctx->_kv_row_stride;            /* 行跨度 ≥ kv_dim，64B 对齐利 SIMD 读取 */

    if (cfg->model_version == 4) {
        /* V4: DASH 块稀疏注意力（官方 _dash_decode_forward）。
         * DS_V4_FULLATTN=1: 诊断开关——decode 走全量因果注意力（无路由/偏置），
         * 用于区分“路由选块问题”与“坐标/数值问题”。 */
        static int _full_attn = -1;
        if (_full_attn < 0) _full_attn = getenv("DS_V4_FULLATTN") ? 1 : 0;
        if (_full_attn) {
            ds_causal_attention_aligned(attn_out, q, k_base, v_base,
                                        1, kv_seq_len, n_heads, n_kv_heads, head_dim,
                                        scale, cache_offset, kv_stride);
        } else {
            ds_dash_attention_decode(ctx, layer_idx, x, attn_out);
        }
    } else if (cfg->sliding_window_size > 0 && cache_offset >= ctx->prefill_token_count) {
        /* R-SWA decode path: attend only to reference + sliding window.
         * 【为什么】OCR 的图像证据必须全程可见，但生成文本可无限长——
         * 若只保留滑窗，图会被"滑掉"；若全保留，KV 无限增长。折中：
         * 参考集（视觉+prompt）永驻 + 最近 W 个生成 token 滑动，
         * KV 占用封顶在 prefill_len + W（训练时即按此可见性约束训练）。
         * 与 StreamingLLM 的 attention sinks 同属"固定前缀+滑窗"家族，
         * 区别在于前缀是任务必需的证据而非数值锚点（docs/13 §4.6）。 */
        int prefill_len = ctx->prefill_token_count;   /* 参考集长度：视觉+prompt token（如 908） */
        int W = cfg->sliding_window_size;             /* 滑窗 W=128：只看最近 128 个已生成 token */
        int window_start = prefill_len > (kv_seq_len - W) ? prefill_len : (kv_seq_len - W);

        /* Two non-contiguous ranges: [0..prefill_len-1] and [window_start..kv_seq_len-1]
         * For efficiency, we compute attention manually with these two ranges.
         * If window_start <= prefill_len, the ranges overlap and we just use
         * [0..kv_seq_len-1] (which happens during warmup when kv_seq_len < prefill_len + W). */
        /* 统一走 RSWA worker: warmup（window_start==prefill_len）时两段连续
         * 覆盖 [0,kv_seq_len)，j 顺序与全量因果一致 → 逐位一致；
         * BF16 时 worker 内用 bf16×f32 点积/axpy（r2 任务2）。 */
        ds_rswa_attention_aligned_bf16(attn_out, q, k_base, v_base,
                                       1, n_heads, n_kv_heads, head_dim,
                                       scale, kv_stride,
                                       prefill_len, window_start, kv_seq_len,
                                       ctx->kv_bf16);
    } else {
        /* Standard causal attention (prefill or no R-SWA) */
        ds_causal_attention_aligned(attn_out, q, k_base, v_base,
                                    1, kv_seq_len, n_heads, n_kv_heads, head_dim,
                                    scale, cache_offset, kv_stride);
    }
    if (ctx->profile_enabled) ctx->perf_layer_attn_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;

    /* Output projection + residual */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *proj_out = ctx->dec_proj_out;
    ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, 1, q_dim, hidden);
    /* wo: [1280,1280] BF16 matvec，把 10 个 head 的拼接输出混合回残差流 */
    if (ctx->profile_enabled) ctx->perf_layer_proj_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;

    /* Post-attention: fused residual add + RMS norm.
     * This combines out = x + proj_out and norm_out = rms_norm(out, weight)
     * into a single operation, eliminating one pass over hidden=1280 elements. */
    ds_residual_rms_norm(out, x_norm, x, proj_out, layer->post_attn_norm, hidden, cfg->dec_rms_norm_eps);

    /* MLP (dense FFN for layer 0, MoE for layers 1-11) */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *mlp_out = ctx->dec_expert_out;  /* reuse preallocated buffer */
    mlp_forward(mlp_out, x_norm, layer, cfg, ctx, layer_idx,
                ctx->dec_dense_gate, ctx->dec_dense_up, ctx->dec_dense_swiglu,
                ctx->moe_expert_gate_buf, ctx->moe_expert_up_buf,
                ctx->moe_expert_gate_up_buf, ctx->moe_expert_hidden_buf,
                ctx->moe_shared_gate_buf, ctx->moe_shared_up_buf,
                ctx->moe_shared_gate_up_buf, ctx->moe_shared_swiglu_buf,
                ctx->moe_shared_out_buf,
                ctx->moe_expert_outputs);

    /* Add MLP output + residual (SIMD vector add)
     * 此刻 out 已含注意力残差（x + Wo@attn），再叠加 MLP/MoE 输出
     * 得到本层完整输出；写入 out（交换缓冲）供下一层当输入，x 保持
     * 只读使残差来源干净。 */
    ds_vec_add(out, out, mlp_out, hidden);

    if (ctx->profile_enabled) {
        ctx->perf_layer_mlp_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;
        ctx->perf_layer_total_ms[layer_idx] += (ds_time_sec() - t_layer_start) * 1000.0;
    }
}

/* ========================================================================
 * Decoder Prefill (multiple tokens)
 * ======================================================================== */

/* ds_decoder_prefill - 解码器预填充：seq_len 个 token 并行过 12 层并填满 KV cache
 * （API 语义与 Prefill vs Decode 的对比见 ds_moe_decoder.h 的完整文档块）
 * 【算法】对每层 l（外层循环层、内层并行 token——权重只读一遍摊给所有 token）:
 *   批量 RMSNorm → 批量 QKV(sgemm) → per-head Q/K norm → 批量 RoPE
 *   → K/V 整块写入 KV cache → 因果批量注意力 → Wo 投影 + 残差
 *   → post-attn norm → FFN（L0 批量 sgemm / L1-11 路由+Gather-Compute-Scatter）
 *   最后对最后一个位置算 logits（首个生成 token 的来源）。
 * 【维度】input_embeds [seq_len, 1280]；Q [seq_len, 1280]；K/V [seq_len, 1280]；
 *   gate_scores [seq_len, 64]；x_norm [seq_len, 1280]
 * 【为什么 prefill 用 sgemm】seq_len≈280-330 时权重读取被摊薄为 1/seq_len，
 *   瓶颈从内存带宽转为计算吞吐，BLAS sgemm 的计算利用率远高于 matvec；
 *   decode 单 token 时反过来（见 ds_decoder_forward 注释）。
 * 【为什么层间 madvise 预取】prefill 每层触碰全部 64 个专家 mmap（~440MB），
 *   冷页缺页中断会卡住层切换，提前 WILLNEED 让内核在算本层时异步读下一层。
 * 【返回】0 成功；-1 = KV 缓存越界或 scratch 分配失败
 * 【参考】docs/04 §2.6-2.7；docs/05-性能优化实战篇（v0.5→v0.9 30s→0.93s）。
 */
int ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* KV cache bounds: a prefix larger than the cache would write past the
     * end of this layer's slab into the next layer's (silent corruption). */
    if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        fprintf(stderr, "ds_decoder_prefill: prefix %d + cached %d exceeds KV cache %d\n",
                seq_len, ctx->kv_cache_len, ctx->kv_cache_max);
        return -1;
    }

    /* Allocate prefill buffers
     * prefill 是一次性路径（每张图调用一次），直接 malloc 即可，
     * 不像 decode 每步都要跑——decode 的临时缓冲才值得预分配复用。 */
    float *x = (float *)malloc(seq_len * hidden * sizeof(float));
    float *x_norm = (float *)malloc(seq_len * hidden * sizeof(float));
    float *Q = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *K = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *V = (float *)malloc(seq_len * kv_dim * sizeof(float));

    memcpy(x, input_embeds, seq_len * hidden * sizeof(float));

    for (int l = 0; l < cfg->dec_layers; l++) {
        ds_dec_layer_t *layer = &dec->layers[l];

        /* Prefetch next layer's weights so the kernel pages them in while
         * this layer computes. Prefill touches ALL routed experts per layer
         * (~440MB), so cold-cache page-in would otherwise stall every layer
         * transition. Prefill routes through the raw gate/up/down mmaps
         * (not the decode-only fused block), so prefetch those. */
        if (l + 1 < cfg->dec_layers) {
            ds_dec_layer_t *nl = &dec->layers[l + 1];
            if (nl->wq_weight_bf16)
                madvise((void *)nl->wq_weight_bf16,
                        (size_t)(q_dim + kv_dim * 2) * hidden * 2, MADV_WILLNEED);
            if (nl->wo_weight_bf16)
                madvise((void *)nl->wo_weight_bf16,
                        (size_t)q_dim * hidden * 2, MADV_WILLNEED);
            for (int e = 0; e < cfg->dec_n_routed_experts; e++) {
                if (nl->experts[e].gate_weight_bf16)
                    madvise((void *)nl->experts[e].gate_weight_bf16,
                            (size_t)cfg->dec_moe_inter * hidden * 2, MADV_WILLNEED);
                if (nl->experts[e].up_weight_bf16)
                    madvise((void *)nl->experts[e].up_weight_bf16,
                            (size_t)cfg->dec_moe_inter * hidden * 2, MADV_WILLNEED);
                if (nl->experts[e].down_weight_bf16)
                    madvise((void *)nl->experts[e].down_weight_bf16,
                            (size_t)hidden * cfg->dec_moe_inter * 2, MADV_WILLNEED);
            }
            if (nl->shared_gate_weight_bf16)
                madvise((void *)nl->shared_gate_weight_bf16,
                        (size_t)cfg->dec_n_shared_experts * cfg->dec_moe_inter * hidden * 2,
                        MADV_WILLNEED);
            if (nl->shared_up_weight_bf16)
                madvise((void *)nl->shared_up_weight_bf16,
                        (size_t)cfg->dec_n_shared_experts * cfg->dec_moe_inter * hidden * 2,
                        MADV_WILLNEED);
            if (nl->shared_down_weight_bf16)
                madvise((void *)nl->shared_down_weight_bf16,
                        (size_t)hidden * cfg->dec_n_shared_experts * cfg->dec_moe_inter * 2,
                        MADV_WILLNEED);
        }

        /* 输入 RMSNorm (prefill 批量模式)
         * 与 decode 路径不同, prefill 一次性处理 seq_len 个 token,
         * 所以 ds_rms_norm 的第二个参数 seq_len > 1, 走 sgemm 批量化路径。*/
        ds_rms_norm(x_norm, x, layer->input_norm, seq_len, hidden, cfg->dec_rms_norm_eps);

        /* QKV 投影: 批量矩阵乘法
         * Prefill 路径用 sgemm (F32×F32) 而不是 matvec (BF16×F32)。
         * BF16 权重在 ds_linear_nobias_bf16 内部转换: BF16 → F32 on-the-fly,
         * 然后用 cblas_sgemm 做批量矩阵乘。
         *
         * 【为什么 prefill 用 sgemm 更快？】
         * matvec 适合单 token（内存带宽瓶颈），
         * sgemm 适合多 token（计算吞吐瓶颈）。
         * Prefill 的 seq_len ≈ 270-330 tokens → sgemm 远快于 matvec。*/
        ds_linear_nobias_bf16(Q, x_norm, layer->wq_weight_bf16, seq_len, hidden, q_dim);
        ds_linear_nobias_bf16(K, x_norm, layer->wk_weight_bf16, seq_len, hidden, kv_dim);
        ds_linear_nobias_bf16(V, x_norm, layer->wv_weight_bf16, seq_len, hidden, kv_dim);

        /* BF16 truncation after QKV to match Python's autocast BF16 matmul output */
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) {
            ds_bf16_truncate_array(Q, seq_len * q_dim);
            ds_bf16_truncate_array(K, seq_len * kv_dim);
            ds_bf16_truncate_array(V, seq_len * kv_dim);
        }

        /* Per-head RMSNorm (optional: V2/OCR-2 does not use these) */
        if (layer->q_norm_weight)
            ds_rms_norm_per_head(Q, layer->q_norm_weight, seq_len, n_heads, head_dim, cfg->dec_rms_norm_eps);
        if (layer->k_norm_weight)
            ds_rms_norm_per_head(K, layer->k_norm_weight, seq_len, n_kv_heads, head_dim, cfg->dec_rms_norm_eps);

        /* Debug: dump layer 0 Q/K/V for comparison with Python */
        if (l == 0 && getenv("DS_DUMP_DECODER")) {
            FILE *df;
            df = fopen("dump/c_layer0_Q_pre_rope.bin", "wb");
            if (df) { fwrite(Q, sizeof(float), seq_len * q_dim, df); fclose(df); }
            df = fopen("dump/c_layer0_K_pre_rope.bin", "wb");
            if (df) { fwrite(K, sizeof(float), seq_len * kv_dim, df); fclose(df); }
            df = fopen("dump/c_layer0_V.bin", "wb");
            if (df) { fwrite(V, sizeof(float), seq_len * kv_dim, df); fclose(df); }
            df = fopen("dump/c_layer0_x_norm.bin", "wb");
            if (df) { fwrite(x_norm, sizeof(float), seq_len * hidden, df); fclose(df); }
            fprintf(stderr, "Dumped layer 0 Q/K/V/x_norm for Python comparison\n");
        }

        /* Apply RoPE using precomputed cache (V4: page2d 表)
         * base = kv_cache_len：支持增量 prefill（缓存里已有前几次的 token 时，
         * 本批的位置从 base 连续编号），单次 prefill 时 base=0。 */
        {
            int base = ctx->kv_cache_len;
            float *cos_p, *sin_p;
            if (cfg->model_version == 4) {
                cos_p = (float *)ds_dash_rope_row(ctx, base, 0); /* 原型在 ds_dash.h */
                sin_p = (float *)ds_dash_rope_row(ctx, base, 1);
            } else {
                cos_p = ctx->rope_cache_cos + base * head_dim;
                sin_p = ctx->rope_cache_sin + base * head_dim;
            }
            ds_apply_rope_neox(Q, cos_p, sin_p, seq_len, n_heads, head_dim);
            ds_apply_rope_neox(K, cos_p, sin_p, seq_len, n_kv_heads, head_dim);
        }

        /* Store K, V directly into F32 KV cache (no BF16 conversion)
         * 逐行写入 [0..seq_len) 个位置；与 decode 相同的行布局，
         * 保证 decode 注意力可以无缝续读。 */
        for (int s = 0; s < seq_len; s++) {
            int pos = ctx->kv_cache_len + s;
            void *cache_k_row = ds_kv_k_row(ctx, l, pos);
            void *cache_v_row = ds_kv_v_row(ctx, l, pos);
            ds_kv_store_row(ctx, cache_k_row, K + s * kv_dim, kv_dim);
            ds_kv_store_row(ctx, cache_v_row, V + s * kv_dim, kv_dim);
        }

        /* Causal attention (prefill all tokens) — read directly from F32 KV cache.
         * V4: DASH 路由注意力（官方 _infer_prefill_attend + _forward_dense）*/
        {
            int total_kv_len = ctx->kv_cache_len + seq_len;
            /* q_offset = kv_cache_len：query s 的全局位置 = q_offset+s，
             * 内核据此保证第 s 个 query 只能看到 key 行 [0, q_offset+s]
             * ——批量因果掩码由位置编号隐式实现，无需 [seq,seq] 掩码矩阵。 */
            float *k_base = ds_kv_k_layer(ctx, l);
            float *v_base = ds_kv_v_layer(ctx, l);
            int kv_stride = ctx->_kv_row_stride;

            float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
            if (cfg->model_version == 4) {
                ds_dash_attention_prefill(ctx, l, x, Q, seq_len, attn_out);
            } else if (ctx->kv_bf16) {
                /* BF16 KV: 整层 K/V 展开到 F32 scratch（每层每 prefill 一次），
                 * 元素值 = 存储的 bf16 精确展开，批量注意力数学不变 */
                if (!ctx->kv_pf_scratch_k) {
                    size_t region = (size_t)ctx->kv_cache_max * kv_stride;
                    ctx->kv_pf_scratch_k = malloc(region * sizeof(float));
                    ctx->kv_pf_scratch_v = malloc(region * sizeof(float));
                    if (!ctx->kv_pf_scratch_k || !ctx->kv_pf_scratch_v) return -1;
                }
                ds_bf16_to_f32_batch(ctx->kv_pf_scratch_k, (const uint16_t *)k_base,
                                     (size_t)total_kv_len * kv_stride);
                ds_bf16_to_f32_batch(ctx->kv_pf_scratch_v, (const uint16_t *)v_base,
                                     (size_t)total_kv_len * kv_stride);
                ds_causal_attention_aligned(attn_out, Q, ctx->kv_pf_scratch_k, ctx->kv_pf_scratch_v,
                                    seq_len, total_kv_len,
                                    n_heads, n_kv_heads, head_dim, scale, ctx->kv_cache_len, kv_stride);
            } else {
                ds_causal_attention_aligned(attn_out, Q, k_base, v_base,
                                    seq_len, total_kv_len,
                                    n_heads, n_kv_heads, head_dim, scale, ctx->kv_cache_len, kv_stride);
            }

            /* Output projection + residual */
            float *proj_out = (float *)malloc(seq_len * hidden * sizeof(float));
            ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, seq_len, q_dim, hidden);
            /* BF16 truncation after output projection */
            extern int ds_bf16_simulate_python;
            if (ds_bf16_simulate_python) ds_bf16_truncate_array(proj_out, seq_len * hidden);
            free(attn_out);

            ds_vec_add(x, x, proj_out, seq_len * hidden);
            free(proj_out);
        }

        /* Debug: dump layer 0 after attention for comparison */
        if (l == 0 && getenv("DS_DUMP_DECODER")) {
            FILE *df = fopen("dump/c_layer0_after_attn.bin", "wb");
            if (df) { fwrite(x, sizeof(float), seq_len * hidden, df); fclose(df); }
            fprintf(stderr, "Dumped C layer 0 after attn (last token mean=%.6f)\n",
                    x[(seq_len-1)*hidden] == x[(seq_len-1)*hidden] ? 
                    0.0f : 0.0f); /* placeholder, actual check below */
            /* Check last token stats */
            float last_mean = 0;
            for (int i = 0; i < hidden; i++) last_mean += x[(seq_len-1)*hidden + i];
            last_mean /= hidden;
            fprintf(stderr, "  C layer0 after_attn last token: mean=%.6f\n", last_mean);
        }

        /* Post-attention RMSNorm */
        ds_rms_norm(x_norm, x, layer->post_attn_norm, seq_len, hidden, cfg->dec_rms_norm_eps);

        /* MLP: dense FFN for layer 0 (batched sgemm), MoE for layers 1-11 (per-token) */
        if (l < cfg->dec_first_k_dense) {
            /* Dense FFN (layer 0): batched via BLAS sgemm — much faster than per-token */
            float *gate_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *up_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *swiglu_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *mlp_out = (float *)malloc(seq_len * hidden * sizeof(float));

            ds_linear_nobias_bf16(gate_buf, x_norm, layer->dense_gate_weight_bf16,
                                   seq_len, hidden, cfg->dec_intermediate);
            ds_linear_nobias_bf16(up_buf, x_norm, layer->dense_up_weight_bf16,
                                   seq_len, hidden, cfg->dec_intermediate);
            if (ds_bf16_simulate_python) {
                ds_bf16_truncate_array(gate_buf, seq_len * cfg->dec_intermediate);
                ds_bf16_truncate_array(up_buf, seq_len * cfg->dec_intermediate);
            }

            /* Direct SwiGLU — no gate_up interleaving needed */
            ds_swiglu_direct(swiglu_buf, gate_buf, up_buf, seq_len, cfg->dec_intermediate);
            if (ds_bf16_simulate_python) ds_bf16_truncate_array(swiglu_buf, seq_len * cfg->dec_intermediate);
            ds_linear_nobias_bf16(mlp_out, swiglu_buf, layer->dense_down_weight_bf16,
                                   seq_len, cfg->dec_intermediate, hidden);

            /* BF16 truncation after dense FFN down projection */
            extern int ds_bf16_simulate_python;
            if (ds_bf16_simulate_python) ds_bf16_truncate_array(mlp_out, seq_len * hidden);

            ds_vec_add(x, x, mlp_out, seq_len * hidden);
            free(gate_buf); free(up_buf); free(swiglu_buf); free(mlp_out);
        } else {
            /* MoE MLP (layers 1-11): batched gate scoring + batched shared experts
             * + per-token routed expert forward. */
            int n_experts = cfg->dec_n_routed_experts;
            int top_k = cfg->dec_top_k;
            int shared_inter = cfg->dec_n_shared_experts * cfg->dec_moe_inter;

            /* --- Batched gate scoring via sgemm ---
             * All seq_len tokens × 64 experts in one matrix multiply.
             * This is much faster than per-token BF16 matvec for large seq_len.
             * ds_linear_nobias_bf16 内部经 ds_bf16_get_f32_view 把 BF16 权重
             * 转为 F32 视图（首次转换后按指针缓存，非 safetensors 内缓存）
             * 再走 cblas_sgemm——累积精度高于 decode 的逐 token BF16 matvec，
             * 但 top-6 选择对小数值差鲁棒，实践中与 decode 选出同一批专家
             * （decode 侧特意用 BF16 gate 权重匹配 Python，见 moe_forward）。 */
            float *gate_scores = (float *)malloc(seq_len * n_experts * sizeof(float));
            ds_linear_nobias_bf16(gate_scores, x_norm, layer->gate_weight_bf16,
                                   seq_len, hidden, n_experts);

            /* --- Batched softmax + top-K for all tokens ---
             * Store top-K indices and weights for all tokens.
             * 与 decode 相同的"先全局 softmax 再无放回 top-6、不 renorm"
             * 语义（见 moe_forward Step 2），逐 token 独立计算。 */
            int *all_top_indices = (int *)malloc(seq_len * top_k * sizeof(int));
            float *all_top_weights = (float *)malloc(seq_len * top_k * sizeof(float));

            for (int s = 0; s < seq_len; s++) {
                float *s_scores = gate_scores + s * n_experts;
                /* Softmax over all n_experts */
                float max_s = -1e30f;
                for (int e = 0; e < n_experts; e++)
                    if (s_scores[e] > max_s) max_s = s_scores[e];
                float sum_exp = 0.0f;
                for (int e = 0; e < n_experts; e++) {
                    s_scores[e] = ds_fast_expf(s_scores[e] - max_s);
                    sum_exp += s_scores[e];
                }
                float inv = 1.0f / sum_exp;
                for (int e = 0; e < n_experts; e++)
                    s_scores[e] *= inv;

                /* Select top-K from softmax'd scores */
                int *tk_idx = all_top_indices + s * top_k;
                float *tk_wt = all_top_weights + s * top_k;
                for (int k = 0; k < top_k; k++) {
                    int best = -1;
                    float best_score = -1e30f;
                    for (int e = 0; e < n_experts; e++) {
                        int already = 0;
                        for (int j = 0; j < k; j++)
                            if (tk_idx[j] == e) { already = 1; break; }
                        if (already) continue;
                        if (s_scores[e] > best_score) {
                            best_score = s_scores[e];
                            best = e;
                        }
                    }
                    tk_idx[k] = best;
                    tk_wt[k] = best_score;
                }
            }

            /* --- Batched shared experts via sgemm ---
             * All seq_len tokens processed at once:
             *   gate_buf[seq_len, shared_inter] = x_norm[seq_len, hidden] @ gate_W[hidden, shared_inter]
             *   up_buf[seq_len, shared_inter] = x_norm[seq_len, hidden] @ up_W[hidden, shared_inter]
             *   swiglu → down_buf[seq_len, hidden] = swiglu_buf[seq_len, shared_inter] @ down_W[shared_inter, hidden]
             */
            float *shared_mlp_out = NULL;
            if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
                float *s_gate = (float *)malloc(seq_len * shared_inter * sizeof(float));
                float *s_up = (float *)malloc(seq_len * shared_inter * sizeof(float));
                float *s_swiglu = (float *)malloc(seq_len * shared_inter * sizeof(float));
                shared_mlp_out = (float *)malloc(seq_len * hidden * sizeof(float));

                ds_linear_nobias_bf16(s_gate, x_norm, layer->shared_gate_weight_bf16,
                                       seq_len, hidden, shared_inter);
                ds_linear_nobias_bf16(s_up, x_norm, layer->shared_up_weight_bf16,
                                       seq_len, hidden, shared_inter);
                ds_swiglu_direct(s_swiglu, s_gate, s_up, seq_len, shared_inter);
                ds_linear_nobias_bf16(shared_mlp_out, s_swiglu, layer->shared_down_weight_bf16,
                                       seq_len, shared_inter, hidden);

                free(s_gate); free(s_up); free(s_swiglu);
            }

            /* --- Routed experts: group by expert ID for batched sgemm ---
             * Instead of per-token expert forward (860 tokens × 6 experts × 3 matvecs = ~15k matvecs),
             * we group tokens by expert ID and process each group with a single sgemm call.
             * With top-6 from 64 experts, each expert handles ~seq_len*6/64 ≈ 80 tokens on average.
             * A single sgemm of [80, 896] @ [896, 1280] is much more efficient than 80 separate
             * [1, 896] @ [896, 1280] BF16 matvecs. */
            /* Count tokens per expert and build index lists */
            int *expert_count = (int *)calloc(n_experts, sizeof(int));
            for (int s = 0; s < seq_len; s++) {
                int *tk_idx = all_top_indices + s * top_k;
                for (int k = 0; k < top_k; k++)
                    expert_count[tk_idx[k]]++;
            }

            /* Build token lists per expert (expert_tokens[e] = list of token indices) */
            int **expert_tokens = (int **)malloc(n_experts * sizeof(int *));
            int *expert_tokens_cap = (int *)malloc(n_experts * sizeof(int));
            for (int e = 0; e < n_experts; e++) {
                expert_tokens[e] = (int *)malloc(expert_count[e] * sizeof(int));
                expert_tokens_cap[e] = expert_count[e];
                expert_count[e] = 0;  /* reset for filling */
            }
            for (int s = 0; s < seq_len; s++) {
                int *tk_idx = all_top_indices + s * top_k;
                for (int k = 0; k < top_k; k++) {
                    int e = tk_idx[k];
                    expert_tokens[e][expert_count[e]++] = s;
                }
            }

            /* Process each expert that has tokens assigned */
            /* calloc：下面 Scatter 阶段是 += 累加（同一 token 的 6 个专家
             * 依次散写），必须从全零起步。 */
            float *routed_mlp_out = (float *)calloc(seq_len * hidden, sizeof(float));

            for (int e = 0; e < n_experts; e++) {
                int n_tok = expert_count[e];
                if (n_tok == 0) continue;

                /* Gather input tokens for this expert: x_expert[n_tok, hidden] */
                float *x_expert = (float *)malloc(n_tok * hidden * sizeof(float));
                for (int i = 0; i < n_tok; i++) {
                    memcpy(x_expert + i * hidden, x_norm + expert_tokens[e][i] * hidden,
                           hidden * sizeof(float));
                }

                /* Batched expert forward: gate/up/down via sgemm */
                float *e_gate = (float *)malloc(n_tok * cfg->dec_moe_inter * sizeof(float));
                float *e_up = (float *)malloc(n_tok * cfg->dec_moe_inter * sizeof(float));
                float *e_swiglu = (float *)malloc(n_tok * cfg->dec_moe_inter * sizeof(float));
                float *e_out = (float *)malloc(n_tok * hidden * sizeof(float));

                ds_linear_nobias_bf16(e_gate, x_expert,
                                       layer->experts[e].gate_weight_bf16,
                                       n_tok, hidden, cfg->dec_moe_inter);
                ds_linear_nobias_bf16(e_up, x_expert,
                                       layer->experts[e].up_weight_bf16,
                                       n_tok, hidden, cfg->dec_moe_inter);
                ds_swiglu_direct(e_swiglu, e_gate, e_up, n_tok, cfg->dec_moe_inter);
                ds_linear_nobias_bf16(e_out, e_swiglu,
                                       layer->experts[e].down_weight_bf16,
                                       n_tok, cfg->dec_moe_inter, hidden);

                /* Scatter-add expert outputs to mlp_out with routing weights
                 * Scatter 阶段：把该专家的每行输出乘路由权重后散回其来源
                 * token 的行。O(top_k) 线性扫找权重是因为 expert_tokens[e]
                 * 只记了 token 号没记权重槽位——top_k=6 时扫一遍即可，
                 * 无需额外的 (token,slot) 索引结构。 */
                for (int i = 0; i < n_tok; i++) {
                    int s = expert_tokens[e][i];
                    /* Find this expert's weight in the top-K for token s */
                    float w = 0.0f;
                    int *tk_idx = all_top_indices + s * top_k;
                    float *tk_wt = all_top_weights + s * top_k;
                    for (int k = 0; k < top_k; k++) {
                        if (tk_idx[k] == e) { w = tk_wt[k]; break; }
                    }
                    float *dst = routed_mlp_out + s * hidden;
                    const float *src = e_out + i * hidden;
                    for (int j = 0; j < hidden; j++)
                        dst[j] += w * src[j];
                }

                free(x_expert); free(e_gate); free(e_up);
                free(e_swiglu); free(e_out);
            }

            /* Combine routed + shared into x (residual add) */
            if (shared_mlp_out) {
                for (int i = 0; i < seq_len * hidden; i++)
                    x[i] += routed_mlp_out[i] + shared_mlp_out[i];
                free(shared_mlp_out);
            } else {
                for (int i = 0; i < seq_len * hidden; i++)
                    x[i] += routed_mlp_out[i];
            }

            free(routed_mlp_out);
            free(gate_scores);
            free(all_top_indices);
            free(all_top_weights);
            for (int e = 0; e < n_experts; e++) free(expert_tokens[e]);
            free(expert_tokens); free(expert_tokens_cap); free(expert_count);
        }

        /* Debug: dump per-layer output for comparison */
        if (getenv("DS_DUMP_LAYERS")) {
            char path[256];
            snprintf(path, sizeof(path), "dump/c_layer%d_out.bin", l);
            FILE *df = fopen(path, "wb");
            if (df) { fwrite(x, sizeof(float), seq_len * hidden, df); fclose(df); }
            /* Stats for last token */
            float last_mean = 0, last_max = 0;
            for (int i = 0; i < hidden; i++) {
                float v = x[(seq_len-1)*hidden + i];
                last_mean += v;
                if (fabsf(v) > last_max) last_max = fabsf(v);
            }
            last_mean /= hidden;
            fprintf(stderr, "  Layer %d done, last token: mean=%.4f max_abs=%.4f\n",
                    l, last_mean, last_max);
        }

        /* Debug: dump last token hidden after each prefill layer for comparison with Python */
        if (getenv("DS_DUMP_LAYERS")) {
            char path[256];
            snprintf(path, sizeof(path), "dump/c_prefill_layer%d_last.bin", l);
            FILE *df = fopen(path, "wb");
            if (df) { fwrite(x + (seq_len-1)*hidden, sizeof(float), hidden, df); fclose(df); }
        }

        if (ds_verbose >= 2)
            fprintf(stderr, "Decoder prefill layer %d/%d done\n", l + 1, cfg->dec_layers);
    }

    /* Dump last token hidden state for debugging (before final norm) */
    if (getenv("DS_DUMP_DECODER")) {
        float *last_x = x + (seq_len - 1) * hidden;
        FILE *f = fopen("dump/dec_prefill_last_hidden.bin", "wb");
        if (f) { fwrite(last_x, sizeof(float), hidden, f); fclose(f); }
        /* Compute logits for last token */
        float *norm_x = (float *)malloc(hidden * sizeof(float));
        ds_rms_norm(norm_x, last_x, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);
        float *logits = (float *)malloc(cfg->vocab_size * sizeof(float));
        if (dec->lm_head_bf16)
            ds_bf16_matvec_pub(logits, norm_x, dec->lm_head_bf16, NULL, hidden, cfg->vocab_size);
        else
            ds_bf16_matvec_pub(logits, norm_x, dec->tok_embeddings_bf16, NULL, hidden, cfg->vocab_size);
        f = fopen("dump/dec_prefill_first_logits.bin", "wb");
        if (f) { fwrite(logits, sizeof(float), cfg->vocab_size, f); fclose(f); }
        /* Print top-10 */
        int top_ids[10]; float top_vals[10];
        for (int i = 0; i < 10; i++) { top_ids[i] = -1; top_vals[i] = -1e30f; }
        for (int i = 0; i < cfg->vocab_size; i++) {
            for (int j = 0; j < 10; j++) {
                if (logits[i] > top_vals[j]) {
                    for (int k = 9; k > j; k--) { top_vals[k] = top_vals[k-1]; top_ids[k] = top_ids[k-1]; }
                    top_vals[j] = logits[i]; top_ids[j] = i; break;
                }
            }
        }
        fprintf(stderr, "Prefill logits top-10: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "[%d:%.3f] ", top_ids[i], top_vals[i]);
        fprintf(stderr, "\n");
        free(norm_x); free(logits);
    }

    /* 所有层的 K/V 都写在同一组缓存位置 [kv_cache_len, kv_cache_len+seq_len)，
     * 此处统一推进全局位置指针；后续 decode 的 token 从 seq_len 之后继续追加。 */
    ctx->kv_cache_len += seq_len;

    /* Compute logits from last token position (needed for first generated token)
     * 自回归的第一个生成 token 由 prefill 序列最后一个位置（通常是 prompt
     * 尾部）的隐状态预测；中间位置的 logits 无需计算。 */
    if (ctx->dec_logits) {
        float *last_x = x + (seq_len - 1) * hidden;
        float *norm_x = (float *)malloc(hidden * sizeof(float));
        ds_rms_norm(norm_x, last_x, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);
        /* BF16 truncation of final hidden state before LM head.
         * Python's autocast("cuda", dtype=bfloat16) computes the entire
         * decoder in BF16, so norm_x is BF16 before the LM head matmul.
         * Truncating here makes logits closer to Python's distribution. */
        extern int ds_bf16_simulate_python;
        if (ds_bf16_simulate_python) ds_bf16_truncate_array(norm_x, hidden);
        if (dec->lm_head_bf16)
            ds_bf16_matvec_pub(ctx->dec_logits, norm_x, dec->lm_head_bf16, NULL, hidden, cfg->vocab_size);
        else
            ds_bf16_matvec_pub(ctx->dec_logits, norm_x, dec->tok_embeddings_bf16, NULL, hidden, cfg->vocab_size);
        free(norm_x);
    }

    /* Debug dump */
    if (getenv("DS_DUMP_DECODER")) {
        float *last_x = x + (seq_len - 1) * hidden;
        FILE *f = fopen("dump/dec_prefill_last_hidden.bin", "wb");
        if (f) { fwrite(last_x, sizeof(float), hidden, f); fclose(f); }
        if (ctx->dec_logits) {
            f = fopen("dump/dec_prefill_first_logits.bin", "wb");
            if (f) { fwrite(ctx->dec_logits, sizeof(float), cfg->vocab_size, f); fclose(f); }
        }
    }

    free(x); free(x_norm); free(Q); free(K); free(V);
    /* BF16 KV 的整层展开 scratch 用完即释（−84MB 驻留，r2 任务2/3） */
    free(ctx->kv_pf_scratch_k);
    free(ctx->kv_pf_scratch_v);
    ctx->kv_pf_scratch_k = NULL;
    ctx->kv_pf_scratch_v = NULL;
    return 0;
}

/* ========================================================================
 * Decoder Forward (single token)
 * ======================================================================== */

/* ds_decoder_forward - 解码器单步前向：吃一个 token 嵌入，吐下一个 token ID
 * （12 层内部逐步数据流图见 ds_moe_decoder.h 的完整文档块）
 * 【算法】拷贝嵌入 → madvise 预取尾部层权重 → 12× decoder_layer_forward
 *   （双缓冲指针交换免 memcpy）→ kv_cache_len++ → final RMSNorm
 *   → LM head 选词。选词分两条路：
 *   fast path（temperature==0，默认）：流式 argmax 直接在 lm_head 的
 *     129280 行 BF16 权重上边乘边记冠军，不物化 logits 向量；
 *   full path（temperature>0 或需要全量 logits）：sgemm 算全部 logits，
 *     再施加重复惩罚/n-gram 阻断/温度采样。
 * 【参数】input_embed — 当前 token 的嵌入 [1280]（tok_embeddings 查表得到，
 *   查表在 ds_ocr.c 的解码循环里完成）；返回值 — 预测的下一个 token ID
 * 【为什么 decode 是内存带宽瓶颈】每步要流过 ~943MB BF16 权重却只有
 *   ~2.4M FLOPs 的计算，理论极限受内存带宽约束（docs/05 的 Roofline 分析）。
 * 【参考】docs/04 §2.7 decode 循环；docs/13 §8（argmax 不需要 softmax——
 *   argmax 对 softmax 单调不变，贪心解码可直接省掉归一化）。
 */
int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Copy input to working buffer */
    float *x = ctx->dec_x;
    memcpy(x, input_embed, hidden * sizeof(float));

    /* Prefetch distant layer weights via madvise.
     * In safetensors alphabetical layout, layers 10/11 sit at file offsets
     * ~1140-2031 MB — between layers 1 and 2! During sequential decode
     * (layer 0→1→2→...→9→10→11), the OS must seek backward to page-in
     * layers 10/11 after layer 9. By issuing WILLNEED at the start of
     * each decode step, the kernel begins async page-in while we compute
     * layers 0-9, reducing cold-cache stall time.
     * Only prefetch layers 10/11 + their guaranteed-used weights (attention
     * + shared experts). Routed experts are prefetched per-expert in moe_forward. */
    for (int _pl = 10; _pl < cfg->dec_layers; _pl++) {
        ds_dec_layer_t *_layer = &dec->layers[_pl];
        /* Attention weights: wq, wk, wv, wo — contiguous in safetensors,
         * so one madvise per contiguous region is sufficient. */
        if (_layer->wq_weight_bf16)
            madvise((void *)_layer->wq_weight_bf16,
                    (size_t)(q_dim + kv_dim * 2) * hidden * 2, MADV_WILLNEED);
        if (_layer->wo_weight_bf16)
            madvise((void *)_layer->wo_weight_bf16,
                    (size_t)q_dim * hidden * 2, MADV_WILLNEED);
        /* Shared expert weights — always used, not dependent on routing.
         * If expert_block is available, shared gate_up_fused is in the
         * contiguous block (prefetch as part of the block). Otherwise
         * prefetch separate gate+up weights from mmap. */
        {
            int shared_inter = cfg->dec_n_shared_experts * cfg->dec_moe_inter;
            if (_layer->shared_down_weight_bf16)
                madvise((void *)_layer->shared_down_weight_bf16,
                        (size_t)hidden * shared_inter * 2, MADV_WILLNEED);
            if (_layer->expert_block_bf16 && _layer->shared_gate_up_fused_bf16) {
                /* Prefetch from the contiguous block: shared gate_up_fused
                 * is at the tail of the expert block */
                size_t shared_fused_bytes = (size_t)2 * shared_inter * hidden * 2;
                madvise((void *)_layer->shared_gate_up_fused_bf16, shared_fused_bytes, MADV_WILLNEED);
            } else if (_layer->shared_gate_weight_bf16) {
                madvise((void *)_layer->shared_gate_weight_bf16,
                        (size_t)shared_inter * hidden * 2, MADV_WILLNEED);
                madvise((void *)_layer->shared_up_weight_bf16,
                        (size_t)shared_inter * hidden * 2, MADV_WILLNEED);
            }
        }
    }

    /* Process through all layers */
    float *out = ctx->dec_layer_out;
    for (int l = 0; l < cfg->dec_layers; l++) {
        decoder_layer_forward(ctx, x, out, l, ctx->kv_cache_len);  /* 第 l 层：读 x 写 out */
        float *tmp = x; x = out; out = tmp;  /* swap pointers, avoid memcpy */
        
        /* Debug: dump hidden state after each layer for comparison */
        if (getenv("DS_DUMP_DECODER_LAYERS")) {
            char path[256];
            snprintf(path, sizeof(path), "dump/decoder_compare/c_hidden_layer%d.bin", l);
            FILE *df = fopen(path, "wb");
            if (df) { fwrite(x, sizeof(float), hidden, df); fclose(df); }
        }

        /* Debug: dump decode step layer outputs for first few decode steps */
        if (getenv("DS_DUMP_DECODE_STEPS") && ctx->kv_cache_len < 865) {
            char path[256];
            snprintf(path, sizeof(path), "dump/c_dec_step%d_layer%d_last.bin", ctx->kv_cache_len, l);
            FILE *df = fopen(path, "wb");
            if (df) { fwrite(x, sizeof(float), hidden, df); fclose(df); }
        }
    }

    /* After all layers, x points to the final hidden state.
     * Due to pointer swapping, x may be either dec_x or dec_layer_out. */

    /* Debug: dump hidden state before norm for first decode steps */
    int _step_before = ctx->kv_cache_len;

    /* Advance KV cache position */
    ctx->kv_cache_len++;    /* 本 token 的 K/V 已在层循环内写入，现在对所有层"可见" */

    /* Final RMSNorm */
    ds_rms_norm(x, x, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);  /* 最后一层 norm，输出进入 LM head */

    /* Debug: dump norm output for first decode steps */
    if (getenv("DS_DUMP_DECODE_STEPS") && _step_before < 865) {
        char path[256];
        snprintf(path, sizeof(path), "dump/c_dec_step%d_norm_last.bin", _step_before);
        FILE *df = fopen(path, "wb");
        if (df) { fwrite(x, sizeof(float), hidden, df); fclose(df); }
    }

    /* BF16 intermediate truncation: match Python's autocast("cuda", dtype=bfloat16).
     * Applied for ALL versions and unconditionally, aligning the decode path
     * with ds_decoder_prefill (which always truncates the final hidden state
     * before the LM head). Previously only V3 truncated here, so V1/V2 decode
     * steps ran at a different precision than the prefill that produced the
     * first token. */
    extern int ds_bf16_simulate_python;
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(x, hidden);
    }

    /* LM head: compute logits then sample with optional repetition penalty */
    int vocab = cfg->vocab_size;
    float *logits = ctx->dec_logits;  /* [vocab_size] buffer */
    const uint16_t *lm_w = dec->lm_head_bf16 ? dec->lm_head_bf16 : dec->tok_embeddings_bf16;
    float rp = ctx->repeat_penalty;  /* Repetition penalty (used in both fast and full paths) */

    /* ──── Fast argmax 路径（无 temperature 采样） ────
     * 进入条件：temperature <= 0（贪心解码，CLI 默认）且 LM head 权重存在。
     * temperature>0 的采样需要完整概率分布，必须走底部 full_logits_path。
     * 【为什么 argmax 比全量 logits 快 8 倍？】
     * 标准做法: x [1280] @ lm_head [vocab=129280, 1280] → logits [129280] → argmax
     *   需要读取 631MB BF16 权重、计算 331M 次点积、分配 0.5MB logits 数组。
     *   每步 ~55ms，纯内存带宽瓶颈（M2 Pro 200GB/s，理论 ~3.2ms，实际 ~55ms 因效率低）。
     *
     * ds_argmax_matvec_bf16 的做法:
     *   直接流式读取 lm_head 的每一行 BF16 权重，和 x 做点积，
     *   只跟踪当前最佳值，不存储全量 logits。
     *   这既节省了分配 logits 数组的 F32 内存（~0.5MB），
     *   又避免了 F32 转换后的回写开销。
     *
     * 【Selective Repetition Penalty】
     * 传统做法: 计算全量 logits → 对所有历史 token 除以 penalty → argmax
     *   代价: 每一项都要做一次遍历史 token 的查找 (O(vocab * history_len))
     * ds-ocr 的做法: 
     *   1. 先 argmax 找到最佳 token
     *   2. 如果最佳 token 在历史中，对其施加 penalty
     *   3. 对历史中的其他 token，逐个计算其 logit 并施加 penalty
     *   只需要重新计算 ~O(history_len) 个而非 ~O(vocab) 个 logits
     *   历史通常 < 300 tokens → 从 129280 降到 ~300 次计算，近 430× 加速 */
    if (ctx->temperature <= 0.0f && lm_w) {
        double t_lm = ctx->profile_enabled ? ds_time_sec() : 0;

        /* Step 0: collect n-gram banned tokens up front (usually 0, at most
         * a couple per step). This lets the argmax skip them directly —
         * one exclusion-argmax pass (~8ms) replaces the old fallback that
         * computed full 129280-way logits via sgemm (60ms+ plus a one-time
         * 631MB BF16→F32 LM head conversion). */
        int banned[48];              /* 本步被禁的 token：warmup ban + n-gram 规则
                                      * 容量 = 8（dec_argmax_ban 上限，见 ds_ocr.h）
                                      * + 40（n-gram 去重后上限，超出即 goto 全量路径） */
        int n_banned = 0;
        /* Warmup ban (min_new_tokens): EOS（V3 另加 Ġ）直接进入排除表，
         * 让流式 argmax 从"当前步"的分布里选出真正的次优 token。
         * 旧实现是在调用方用"陈旧的 prefill logits"做事后抑制，
         * 会选到与当前步无关甚至连续重复的 token。 */
        for (int b = 0; b < ctx->dec_argmax_n_ban && b < 8; b++)
            banned[n_banned++] = ctx->dec_argmax_ban[b];
        int ngram_n = ctx->no_repeat_ngram_size;
        if (ngram_n > 0 && ctx->token_history_len >= ngram_n - 1) {
            /* n-gram 阻断理论（HuggingFace no_repeat_ngram_size）：
             * 若历史末尾的 (n-1) gram 在历史中出现过，则所有曾接续它的 token
             * 都被禁掉——从机制上杜绝重复（V3 默认 n=35，见 docs/05-性能优化实战篇） */
            int prefix_len = ngram_n - 1;
            int *hist = ctx->token_history;
            int hist_len = ctx->token_history_len;
            for (int i = 0; i <= hist_len - prefix_len - 1; i++) {  /* 遍历所有历史 n-gram 起点 */
                int match = 1;
                for (int j = 0; j < prefix_len; j++) {              /* 逐位比较 (n-1) gram */
                    if (hist[i + j] != hist[hist_len - prefix_len + j]) {
                        match = 0; break;
                    }
                }
                if (match) {                                        /* 命中：其后续 token 加入禁用表 */
                    int banned_tok = hist[i + prefix_len];
                    int dup = 0;
                    for (int b = 0; b < n_banned; b++) {
                        if (banned[b] == banned_tok) { dup = 1; break; }
                    }
                    if (!dup) {
                        if (n_banned >= 40) goto full_logits_path;  /* pathological */
                        banned[n_banned++] = banned_tok;
                    }
                }
            }
        }

        /* Step 1: Argmax over all rows — GPU path if DS_METAL_LM env var set.
         * 【Metal 触发】metal_ctx 可用 + 启动时检测到 DS_METAL_LM +
         * 本步无禁用 token（排除式 argmax 未实现 GPU 版）；
         * 任一不满足回退 CPU 流式扫描，两者数学等价。 */
        int best_token;
        if (ctx->metal_ctx && ctx->metal_lm_env && ds_metal_is_available(ctx->metal_ctx) && n_banned == 0) {
            best_token = ds_metal_lm_head_argmax(ctx->metal_ctx, x, lm_w, hidden, vocab);
        } else if (n_banned > 0) {
            /* 有禁用 token：单次排除式 argmax（~8ms），跳过 sgemm 全量回退 */
            best_token = ds_argmax_matvec_bf16_excluding(x, lm_w, hidden, vocab, banned, n_banned);
        } else {
            best_token = ds_argmax_matvec_bf16(x, lm_w, hidden, vocab);   /* 常规路径：一次流式扫描 ~8ms */
        }
        float best_val = ds_bf16_dot_row(x, lm_w, hidden, best_token);   /* 补算冠军的 logit 值（惩罚要用） */
        if (rp > 1.0f && ctx->token_history && ctx->token_history_len > 0) {
            /* First: if the argmax winner is in history, apply penalty
             * 重复惩罚 HF 语义：logit>0 时 ÷rp，<0 时 ×rp（都朝抑制方向推） */
            for (int i = 0; i < ctx->token_history_len; i++) {
                if (ctx->token_history[i] == best_token) {
                    best_val = (best_val > 0) ? best_val / rp : best_val * rp;
                    break;
                }
            }
            /* Then: compute logits for other history tokens that might
             * surpass the penalized best. Only check unique tokens.
             * Banned tokens never win (excluded above). */
            for (int i = 0; i < ctx->token_history_len; i++) {
                int tid = ctx->token_history[i];
                if (tid >= 0 && tid < vocab && tid != best_token) {
                    int is_banned = 0;
                    for (int b = 0; b < n_banned; b++) {
                        if (banned[b] == tid) { is_banned = 1; break; }
                    }
                    if (is_banned) continue;
                    float val = ds_bf16_dot_row(x, lm_w, hidden, tid);
                    val = (val > 0) ? val / rp : val * rp;
                    if (val > best_val) {
                        best_val = val;
                        best_token = tid;
                    }
                }
            }
        }

        if (ctx->profile_enabled) {
            ctx->perf_lm_head_ms += (ds_time_sec() - t_lm) * 1000.0;
        }
        return best_token;
    }

full_logits_path:
    /* ──── Full logits path (temperature sampling or ngram fallback) ──── */
    if (!logits) {
        /* Fallback: direct argmax (no penalty support) */
        int token = ds_argmax_matvec_bf16(x, lm_w, hidden, vocab);
        return token;
    }

    /* Compute full logits vector using sgemm with pre-converted F32 weights.
     * For LM head (1280→129280), cblas_sgemm is much faster than per-row
     * NEON BF16 dot because Accelerate's sgemm is heavily optimized. */
    if (ctx->profile_enabled) {
        double t_lm = ds_time_sec();
        if (dec->lm_head_bf16) {
            if (!ctx->lm_head_f32_ready) {
                size_t n = (size_t)vocab * hidden;
                ctx->lm_head_f32 = ds_bf16_to_f32_alloc(dec->lm_head_bf16, n);
                ctx->lm_head_f32_ready = 1;
                if (ds_verbose >= 1)
                    fprintf(stderr, "LM head: converted %zu BF16→F32 weights for sgemm (%.1f MB)\n",
                            n, n * sizeof(float) / 1048576.0);
                /* F32 副本就绪后丢弃 bf16 源页（−0.66GB；回退路径触碰时重入） */
                {
                    extern void ds_drop_source_pages(const void *p, size_t bytes);
                    ds_drop_source_pages(dec->lm_head_bf16, n * sizeof(uint16_t));
                }
            }
            if (ctx->lm_head_f32) {
                ds_f32_matvec_sgemm(logits, x, ctx->lm_head_f32, hidden, vocab);
            } else {
                ds_bf16_matvec_pub(logits, x, dec->lm_head_bf16, NULL, hidden, vocab);
            }
        } else {
            /* Tied embeddings — use tok_embeddings */
            if (!ctx->tok_emb_f32_ready) {
                size_t n = (size_t)vocab * hidden;
                ctx->tok_emb_f32 = ds_bf16_to_f32_alloc(dec->tok_embeddings_bf16, n);
                ctx->tok_emb_f32_ready = 1;
            }
            if (ctx->tok_emb_f32) {
                ds_f32_matvec_sgemm(logits, x, ctx->tok_emb_f32, hidden, vocab);
            } else {
                ds_bf16_matvec_pub(logits, x, dec->tok_embeddings_bf16, NULL, hidden, vocab);
            }
        }
        ctx->perf_lm_head_ms += (ds_time_sec() - t_lm) * 1000.0;
    } else {
        /* Non-profile path: same sgemm optimization */
        if (dec->lm_head_bf16) {
            if (!ctx->lm_head_f32_ready) {
                size_t n = (size_t)vocab * hidden;
                ctx->lm_head_f32 = ds_bf16_to_f32_alloc(dec->lm_head_bf16, n);
                ctx->lm_head_f32_ready = 1;
                {
                    extern void ds_drop_source_pages(const void *p, size_t bytes);
                    ds_drop_source_pages(dec->lm_head_bf16, n * sizeof(uint16_t));
                }
            }
            if (ctx->lm_head_f32) {
                ds_f32_matvec_sgemm(logits, x, ctx->lm_head_f32, hidden, vocab);
            } else {
                ds_bf16_matvec_pub(logits, x, dec->lm_head_bf16, NULL, hidden, vocab);
            }
        } else {
            if (!ctx->tok_emb_f32_ready) {
                size_t n = (size_t)vocab * hidden;
                ctx->tok_emb_f32 = ds_bf16_to_f32_alloc(dec->tok_embeddings_bf16, n);
                ctx->tok_emb_f32_ready = 1;
            }
            if (ctx->tok_emb_f32) {
                ds_f32_matvec_sgemm(logits, x, ctx->tok_emb_f32, hidden, vocab);
            } else {
                ds_bf16_matvec_pub(logits, x, dec->tok_embeddings_bf16, NULL, hidden, vocab);
            }
        }
    }

    /* Debug: print top-5 logits for first few decode steps */
    if (getenv("DS_DUMP_DECODER") && ctx->kv_cache_len < 870) {
        int _top5_ids[5]; float _top5_vals[5];
        for (int i = 0; i < 5; i++) { _top5_ids[i] = -1; _top5_vals[i] = -1e30f; }
        for (int i = 0; i < vocab; i++) {
            for (int j = 0; j < 5; j++) {
                if (logits[i] > _top5_vals[j]) {
                    for (int k = 4; k > j; k--) { _top5_vals[k] = _top5_vals[k-1]; _top5_ids[k] = _top5_ids[k-1]; }
                    _top5_vals[j] = logits[i]; _top5_ids[j] = i; break;
                }
            }
        }
        fprintf(stderr, "Decode step kv=%d logits top-5: ", ctx->kv_cache_len);
        for (int i = 0; i < 5; i++) fprintf(stderr, "[%d:%.2f] ", _top5_ids[i], _top5_vals[i]);
        fprintf(stderr, "\n");
    }

    /* Apply repetition penalty (rp already defined above) */
    if (rp > 1.0f && ctx->token_history && ctx->token_history_len > 0) {
        /* For each previously generated token, divide its logit by penalty
         * (standard HuggingFace repetition penalty implementation) */
        for (int i = 0; i < ctx->token_history_len; i++) {
            int tid = ctx->token_history[i];
            if (tid >= 0 && tid < vocab) {
                if (logits[tid] > 0) {
                    logits[tid] /= rp;
                } else {
                    logits[tid] *= rp;
                }
            }
        }
    }

    /* N-gram 重复惩罚 (no_repeat_ngram_size)
     * 防止模型生成重复的 n-gram 序列（HuggingFace 标准实现）。
     *
     * 原理:
     *   检查历史中最后 (n-1) 个 token 是否曾在更早位置出现过相同的序列。
     *   如果找到匹配，就禁止（设为 -inf）后来跟随的那个 token。
     *
     * 举例 (ngram_n=3):
     *   历史: [...A, B, C, D, E, B, C, ?]
     *   最后 2 个 = [B, C]
     *   之前的 [B, C] 出现在位置 1-2, 其后续 token = D
     *   所以禁止 D，防止出现 B,C,D 的重复
     *
     * 为什么不用 full logits 路径也能做 ngram blocking？
     * 在 argmax 路径（fast path）中，如果 argmax 选择的 token 被禁止，
     * 就跳转到 full_logits_path 重新选下一个最佳 token。
     * 由于 ngram 通常只禁止 0-1 个 token，大部分步骤走 fast path。*/
    int ngram_n = ctx->no_repeat_ngram_size;
    if (ngram_n > 0 && ctx->token_history_len >= ngram_n - 1) {
        int prefix_len = ngram_n - 1;
        int *hist = ctx->token_history;
        int hist_len = ctx->token_history_len;
        /* Check each position in history for matching prefix */
        for (int i = 0; i <= hist_len - prefix_len - 1; i++) {
            /* Compare prefix: hist[i..i+prefix_len-1] vs hist[hist_len-prefix_len..hist_len-1] */
            int match = 1;
            for (int j = 0; j < prefix_len; j++) {
                if (hist[i + j] != hist[hist_len - prefix_len + j]) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                /* Ban the token that follows this prefix */
                int banned = hist[i + prefix_len];
                if (banned >= 0 && banned < vocab) {
                    logits[banned] = -1e30f;
                }
            }
        }
    }

    /* Warmup ban tokens (min_new_tokens) — same semantics as the fast path */
    for (int b = 0; b < ctx->dec_argmax_n_ban; b++) {
        int t = ctx->dec_argmax_ban[b];
        if (t >= 0 && t < vocab) logits[t] = -1e30f;
    }

    /* Find argmax from logits */
    int token = 0;
    float best = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > best) {
            best = logits[i];
            token = i;
        }
    }

    /* Temperature sampling (if enabled)
     * softmax(logits/T)：T→0 趋近 argmax，T 越大分布越平；随后按累积分布
     * 做逆变换采样（inverse-CDF）：r∈[0,1) 落在哪个累积区间就选哪个 token。
     * max_logit 用带惩罚后的 best——先减最大值防 exp 上溢。 */
    if (ctx->temperature > 0.0f) {
        /* Apply temperature, compute softmax, sample */
        float temp = ctx->temperature;
        float max_logit = best;
        double sum_exp = 0.0;
        for (int i = 0; i < vocab; i++) {
            logits[i] = expf((logits[i] - max_logit) / temp);
            sum_exp += logits[i];
        }
        float inv_sum = (float)(1.0 / sum_exp);
        for (int i = 0; i < vocab; i++) logits[i] *= inv_sum;

        /* Weighted random sampling */
        float r = (float)rand() / (float)RAND_MAX;
        float cum = 0.0f;
        token = vocab - 1;
        for (int i = 0; i < vocab; i++) {
            cum += logits[i];
            if (cum >= r) { token = i; break; }
        }
    }

    return token;
}

/* ========================================================================
 * Batch Decode: process N tokens through all layers at once.
 * Each token attends to all prior KV cache + prior tokens in batch.
 * Uses batched matvec (sgemm) for QKV projections and LM head,
 * shared KV cache reads across all N tokens per layer.
 * ======================================================================== */

/* ds_decoder_forward_batch — 批量解码（v0.8 新增）
 *
 * 一次性处理 N 个 token 通过全部 12 层解码器。
 * 与 decode 路径（逐 token 串行 matvec）不同，batch 路径使用 sgemm
 * 批量矩阵乘法，对 QKV、RoPE、注意力、FFN 等所有操作批量处理。
 *
 * 【适用场景】
 * Lookahead decoding / 推测解码 (speculative decoding)：
 * 用一个小 draft 模型预测后续 N 个 token，然后用大模型验证。
 * 批量解码比逐 token 解码快约 3-5×，因为 sgemm 的计算利用率更高。
 *
 * 【注意事项】
 * 每个 token 可以 attend 到之前的所有 token（包括本批次中更早的 token），
 * 所以本批次中的 token 之间有因果依赖——即 token[i] 可以看到 token[0..i-1]
 * 但看不到 token[i+1..]。
 *
 * 当前 MoE 专家的前向仍然是逐 token 的（折衷：MLP 占比 ~30%，
 * 批量化的注意力、QKV、RMSNorm 占比 ~70%，整体仍有收益）。*/
void ds_decoder_forward_batch(ds_ctx_t *ctx, const float *input_embeds,
                                int n_tokens, int *tokens_out) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_stride = ctx->_kv_row_stride;

    if (ctx->kv_cache_len + n_tokens > ctx->kv_cache_max) {
        /* 返回类型为 void，越界只能就地报错并静默返回（不推进缓存）；
         * 与 prefill 的返回 -1 语义不同，调用方需自行判断。 */
        fprintf(stderr, "ds_decoder_forward_batch: %d + cached %d exceeds KV cache %d\n",
                n_tokens, ctx->kv_cache_len, ctx->kv_cache_max);
        return;
    }

    /* Allocate batch buffers */
    float *x = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));
    float *x_norm = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));
    float *Q = (float *)malloc((size_t)n_tokens * q_dim * sizeof(float));
    float *K = (float *)malloc((size_t)n_tokens * kv_dim * sizeof(float));
    float *V = (float *)malloc((size_t)n_tokens * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)n_tokens * q_dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));

    memcpy(x, input_embeds, (size_t)n_tokens * hidden * sizeof(float));

    for (int l = 0; l < cfg->dec_layers; l++) {
        ds_dec_layer_t *layer = &dec->layers[l];
        int cache_offset = ctx->kv_cache_len;

        /* Input RMSNorm (batched) */
        ds_rms_norm(x_norm, x, layer->input_norm, n_tokens, hidden, cfg->dec_rms_norm_eps);

        /* QKV projections — batched sgemm for all n_tokens at once.
         * This is the key advantage: one sgemm call instead of n_tokens separate matvecs. */
        ds_linear_nobias_bf16(Q, x_norm, layer->wq_weight_bf16, n_tokens, hidden, q_dim);
        ds_linear_nobias_bf16(K, x_norm, layer->wk_weight_bf16, n_tokens, hidden, kv_dim);
        ds_linear_nobias_bf16(V, x_norm, layer->wv_weight_bf16, n_tokens, hidden, kv_dim);

        /* Per-head Q/K RMSNorm (optional) */
        if (layer->q_norm_weight)
            ds_rms_norm_per_head(Q, layer->q_norm_weight, n_tokens, n_heads, head_dim, cfg->dec_rms_norm_eps);
        if (layer->k_norm_weight)
            ds_rms_norm_per_head(K, layer->k_norm_weight, n_tokens, n_kv_heads, head_dim, cfg->dec_rms_norm_eps);

        /* RoPE */
        ds_apply_rope_neox(Q, ctx->rope_cache_cos + cache_offset * head_dim,
                           ctx->rope_cache_sin + cache_offset * head_dim,
                           n_tokens, n_heads, head_dim);
        ds_apply_rope_neox(K, ctx->rope_cache_cos + cache_offset * head_dim,
                           ctx->rope_cache_sin + cache_offset * head_dim,
                           n_tokens, n_kv_heads, head_dim);

        /* Store K, V in F32 cache */
        for (int s = 0; s < n_tokens; s++) {
            int pos = cache_offset + s;
            ds_kv_store_row(ctx, ds_kv_k_row(ctx, l, pos), K + s * kv_dim, kv_dim);
            ds_kv_store_row(ctx, ds_kv_v_row(ctx, l, pos), V + s * kv_dim, kv_dim);
        }

        /* Causal attention — batched, with aligned stride */
        {
            int total_kv_len = cache_offset + n_tokens;
            float *k_base = ds_kv_k_layer(ctx, l);
            float *v_base = ds_kv_v_layer(ctx, l);
            ds_causal_attention_aligned(attn_out, Q, k_base, v_base,
                                n_tokens, total_kv_len,
                                n_heads, n_kv_heads, head_dim, scale, cache_offset, kv_stride);

            /* Output projection */
            ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, n_tokens, q_dim, hidden);
        }

        /* Residual add + post-attention RMSNorm (fused) */
        for (int s = 0; s < n_tokens; s++) {
            float *x_s = x + s * hidden;
            float *xn_s = x_norm + s * hidden;
            float *po_s = proj_out + s * hidden;
            ds_residual_rms_norm(x_s, xn_s, x_s, po_s, layer->post_attn_norm, hidden, cfg->dec_rms_norm_eps);
        }

        /* MLP */
        if (l < cfg->dec_first_k_dense) {
            /* Dense FFN — batched sgemm */
            float *gate_buf = (float *)malloc((size_t)n_tokens * cfg->dec_intermediate * sizeof(float));
            float *up_buf = (float *)malloc((size_t)n_tokens * cfg->dec_intermediate * sizeof(float));
            float *swiglu_buf = (float *)malloc((size_t)n_tokens * cfg->dec_intermediate * sizeof(float));
            float *mlp_out = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));

            ds_linear_nobias_bf16(gate_buf, x_norm, layer->dense_gate_weight_bf16,
                                   n_tokens, hidden, cfg->dec_intermediate);
            ds_linear_nobias_bf16(up_buf, x_norm, layer->dense_up_weight_bf16,
                                   n_tokens, hidden, cfg->dec_intermediate);
            ds_swiglu_direct(swiglu_buf, gate_buf, up_buf, n_tokens, cfg->dec_intermediate);
            ds_linear_nobias_bf16(mlp_out, swiglu_buf, layer->dense_down_weight_bf16,
                                   n_tokens, cfg->dec_intermediate, hidden);

            for (int s = 0; s < n_tokens; s++) {
                ds_vec_add(x + s * hidden, x + s * hidden, mlp_out + s * hidden, hidden);
            }
            free(gate_buf); free(up_buf); free(swiglu_buf); free(mlp_out);
        } else {
            /* MoE — per-token (same as decode path, no batch benefit for MoE routing) */
            for (int s = 0; s < n_tokens; s++) {
                float *mlp_out = ctx->dec_expert_out;
                mlp_forward(mlp_out, x_norm + s * hidden, layer, cfg, ctx, l,
                            ctx->dec_dense_gate, ctx->dec_dense_up, ctx->dec_dense_swiglu,
                            ctx->moe_expert_gate_buf, ctx->moe_expert_up_buf,
                            ctx->moe_expert_gate_up_buf, ctx->moe_expert_hidden_buf,
                            ctx->moe_shared_gate_buf, ctx->moe_shared_up_buf,
                            ctx->moe_shared_gate_up_buf, ctx->moe_shared_swiglu_buf,
                            ctx->moe_shared_out_buf,
                            ctx->moe_expert_outputs);
                ds_vec_add(x + s * hidden, x + s * hidden, mlp_out, hidden);
            }
        }
    }

    /* Advance KV cache */
    ctx->kv_cache_len += n_tokens;

    /* Final RMSNorm + LM head for each token
     * batch 场景选 sgemm 全量 logits 而非流式 argmax：F32 权重转换
     * （一次性 ~631MB→1.26GB）被 n_tokens 摊薄，且后续可能需要
     * 对 logits 施加重复惩罚——流式 argmax 拿不到全量分数。 */
    float *norm_buf = (float *)malloc(hidden * sizeof(float));
    for (int s = 0; s < n_tokens; s++) {
        ds_rms_norm(norm_buf, x + s * hidden, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);

        int token;
        if (ctx->dec_logits) {
            if (dec->lm_head_bf16) {
                if (!ctx->lm_head_f32_ready) {
                    size_t n = (size_t)cfg->vocab_size * hidden;
                    ctx->lm_head_f32 = ds_bf16_to_f32_alloc(dec->lm_head_bf16, n);
                    ctx->lm_head_f32_ready = 1;
                    {
                        extern void ds_drop_source_pages(const void *p, size_t bytes);
                        ds_drop_source_pages(dec->lm_head_bf16, n * sizeof(uint16_t));
                    }
                }
                ds_f32_matvec_sgemm(ctx->dec_logits, norm_buf, ctx->lm_head_f32, hidden, cfg->vocab_size);
            } else {
                if (!ctx->tok_emb_f32_ready) {
                    size_t n = (size_t)cfg->vocab_size * hidden;
                    ctx->tok_emb_f32 = ds_bf16_to_f32_alloc(dec->tok_embeddings_bf16, n);
                    ctx->tok_emb_f32_ready = 1;
                }
                ds_f32_matvec_sgemm(ctx->dec_logits, norm_buf, ctx->tok_emb_f32, hidden, cfg->vocab_size);
            }

            /* Apply repetition penalty */
            float rp = ctx->repeat_penalty;
            if (rp > 1.0f && ctx->token_history && ctx->token_history_len > 0) {
                for (int i = 0; i < ctx->token_history_len; i++) {
                    int tid = ctx->token_history[i];
                    if (tid >= 0 && tid < cfg->vocab_size) {
                        ctx->dec_logits[tid] = (ctx->dec_logits[tid] > 0) ?
                            ctx->dec_logits[tid] / rp : ctx->dec_logits[tid] * rp;
                    }
                }
            }

            /* Argmax */
            token = 0;
            float best = ctx->dec_logits[0];
            for (int i = 1; i < cfg->vocab_size; i++) {
                if (ctx->dec_logits[i] > best) { best = ctx->dec_logits[i]; token = i; }
            }
        } else {
            if (dec->lm_head_bf16)
                token = ds_argmax_matvec_bf16(norm_buf, dec->lm_head_bf16, hidden, cfg->vocab_size);
            else
                token = ds_argmax_matvec_bf16(norm_buf, dec->tok_embeddings_bf16, hidden, cfg->vocab_size);
        }
        tokens_out[s] = token;
    }
    free(norm_buf);

    /* Record all tokens in history
     * 批量 token 按生成顺序追加，供后续步骤的重复惩罚/n-gram 阻断使用；
     * 超出 token_history_cap 时静默截断（惩罚只需近期历史）。 */
    if (ctx->token_history) {
        for (int s = 0; s < n_tokens; s++) {
            if (ctx->token_history_len < ctx->token_history_cap) {
                ctx->token_history[ctx->token_history_len++] = tokens_out[s];
            }
        }
    }

    free(x); free(x_norm); free(Q); free(K); free(V);
    free(attn_out); free(proj_out);
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int ds_decoder_load(ds_ctx_t *ctx) {
    /* 权重加载的实际实现位于 ds_ocr.c 的 load_all_weights()（BF16 mmap
     * 零拷贝 + F32 norm），本函数仅保留 API 占位以维持 ds_decoder_* 命名
     * 空间的完整性；ctx 未用，显式 (void) 消除 -Wunused-parameter 告警。 */
    (void)ctx;
    return 0;
}
