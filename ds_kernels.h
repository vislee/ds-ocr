/*
 * ds_kernels.h - Math kernels for DeepSeek-OCR inference
 * ds_kernels.h — DeepSeek-OCR 推理引擎数学内核头文件
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * 底层数学运算。所有操作均基于行主序(float32)张量。
 *
 * Adapted from antirez/qwen-asr project.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【三级内核架构】generic → NEON/AVX → BLAS
 *   第一级 (generic): 纯C99可移植实现，任何平台均可运行
 *   第二级 (NEON/AVX): SIMD加速，ARM NEON(Apple Silicon)或x86 AVX2+FMA
 *   第三级 (BLAS): cblas_sgemm加速大规模矩阵运算(macOS Accelerate/Linux OpenBLAS)
 *   平台分发决策树见 docs/09 §6.1（bfdot/vshll 按指令集细分，运行时探测）；
 *   本头文件是三级共享的公共接口，impl 细节在 ds_kernels_impl.h。
 *
 * 【BF16→F32 on-the-fly 转换】(docs/03 §3.2-3.3；docs/09 §2.2)
 *   权重以BF16格式mmap加载(零拷贝)，matvec计算时即时转F32，节省50%存储。
 *   BF16↔F32 只差一条左移：BF16 位模式 << 16 即 F32 位模式（指数位对齐），
 *   转换发生在寄存器内，权重永远不落盘为 F32。
 *
 * 【关键函数对照表】(对应文档理论概念)
 *   ds_linear_nobias_bf16()     — BF16权重线性变换(解码器核心操作)
 *   ds_causal_attention()       — 因果注意力(解码器，带GQA支持)
 *   ds_rswa_attention_aligned() — R-SWA注意力(V3，reference+window两段)
 *   ds_rms_norm()               — RMSNorm(解码器归一化)
 *   ds_rms_norm_per_head()      — Per-head RMSNorm(DeepSeek特色Q/K归一化)
 *   ds_swiglu_multiply()        — SwiGLU激活(SiLU(gate)*up)
 *   ds_moe_router_bf16()        — MoE路由器(gate+softmax+topK)
 *   ds_argmax_matvec_bf16()     — 流式argmax(不分配129280个logits)
 *   ds_compute_rope_neox()      — RoPE位置编码预计算
 *   ds_apply_rope_neox()        — 应用RoPE旋转位置编码
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_KERNELS_H
#define DS_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

/* Maximum experts per layer (for static array sizing) */
#define DS_MAX_EXPERTS 64

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

/* 以下四个均为 O(n) 逐元素原地操作，线程安全（无共享状态），无同步开销 */
void ds_add_inplace(float *a, const float *b, int n);  /* a[i] += b[i]（残差累加） */
void ds_mul_inplace(float *a, const float *b, int n);  /* a[i] *= b[i] */
void ds_scale(float *x, float s, int n);               /* x[i] *= s */
void ds_copy(float *dst, const float *src, int n);     /* memcpy 语义 dst ← src */

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void ds_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
void ds_linear(float *y, const float *x, const float *W, const float *b,
               int seq_len, int in_dim, int out_dim);

/* 同 ds_linear 但 b=NULL（Transformer 投影均无 bias）；F32 权重版 */
void ds_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim);

/* bf16 weight variants — BF16 权重线性层，内部按 seq_len 自动选路：
 * seq_len==1 走平台 BF16 matvec（带宽优化），seq_len>1 转 F32 后走 BLAS
 * sgemm（吞吐优化）。详见 ds_kernels.c 中同名函数的实现注释。 */
void ds_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                    const float *b, int seq_len, int in_dim, int out_dim);

void ds_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim);

/* seq=1 decoder fast path: compute Q/K/V matvecs with one threaded dispatch */
void ds_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                const uint16_t *Wq_bf16,
                                const uint16_t *Wk_bf16,
                                const uint16_t *Wv_bf16,
                                int in_dim, int q_dim, int kv_dim);

/* ds_matmul_t 的 BF16 权重版：C[M,N] = A[M,K] @ B_bf16[N,K]^T；
 * M==1 退化为 matvec，M>1 转 F32 视图后进 sgemm */
void ds_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N);

/* ========================================================================
 * 2D Convolution (for vision tokenizer SAM patch embed + conv compression)
 * ======================================================================== */

/* im2col + sgemm 的标准卷积：in/out 均为 [C,H,W] 通道优先；
 * h_out=(h_in+2*padding-kh)/stride+1（w_out 同理）；weight[c_out,c_in,kh,kw]。
 * 实现细节见 ds_kernels.c。 */
void ds_conv2d(float *out, const float *in, const float *weight, const float *bias,
               int c_in, int c_out, int h_in, int w_in,
               int kh, int kw, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* LayerNorm with bias: out = (x - mean) / sqrt(var + eps) * weight + bias */
void ds_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                   int seq_len, int hidden, float eps);

/* RMS Normalization: out = x / rms(x) * weight
 * rms(x)=sqrt(mean(x^2))；out/x 均为 [seq_len,hidden]；weight=γ[hidden]；
 * eps 防除零（解码器 1e-6 量级）。线程安全。 */
void ds_rms_norm(float *out, const float *x, const float *weight,
                 int seq_len, int hidden, float eps);

/* Fused residual add + RMS norm: out = x + residual, then norm_out = rms_norm(out, weight).
 * Avoids separate loop for residual and saves one pass over the data. */
void ds_residual_rms_norm(float *out, float *norm_out,
                           const float *x, const float *residual,
                           const float *weight, int hidden, float eps);

/* Vector add: dst[i] = a[i] + b[i] — used for residual connections.
 * This replaces the scalar loops in decoder_layer_forward. */
void ds_vec_add(float *dst, const float *a, const float *b, int n);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq_len, n_heads*head_dim] 原地修改——统计范围是单个 [head_dim] 段
 * 而非整行（DeepSeek-V2 的 QK-Norm，稳定注意力 logit 量级） */
void ds_rms_norm_per_head(float *x, const float *weight,
                           int seq_len, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* 原地激活：silu(x)=x·σ(x)=x/(1+e^-x)；gelu 为精确 erf 版（0.5x(1+erf(x/√2))） */
void ds_silu(float *x, int n);
void ds_gelu(float *x, int n);

/* 逐行 softmax（原地，x:[rows,cols]），内部 max-subtract 防上溢；线程安全 */
void ds_softmax(float *x, int rows, int cols);

/* out[seq,inter] = SiLU(gate_up[seq,2*inter][:,even]) * gate_up[:,odd] */
void ds_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);

/* Direct SiLU(gate) * up without interleaving.
 * gate_buf and up_buf are separate contiguous arrays (not interleaved).
 * out[i] = SiLU(gate_buf[i]) * up_buf[i] for i in [0, intermediate).
 * This avoids the gate_up interleaving pattern which causes non-sequential writes. */
void ds_swiglu_direct(float *out, const float *gate_buf, const float *up_buf,
                       int seq_len, int intermediate);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Bidirectional attention (encoder).
 * Q, K, V: [seq, n_heads * head_dim]
 * out: [seq, n_heads * head_dim]
 * scale = 1/sqrt(head_dim)（调用方算好）；无掩码；online softmax O(1) 额外内存
 */
void ds_bidirectional_attention(float *out, const float *Q, const float *K,
                                const float *V, int seq, int n_heads,
                                int head_dim, float scale);

/*
 * Causal attention with GQA (decoder).
 * Q: [seq_q, n_heads * head_dim]
 * K: [seq_k, n_kv_heads * head_dim]  (stride = kv_dim between rows)
 * V: [seq_k, n_kv_heads * head_dim]
 * q_offset: global position of first query (for causal mask)
 */
void ds_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int q_offset);

/*
 * Causal attention with aligned KV cache stride (decoder).
 * Same as ds_causal_attention but K/V rows are at kv_stride (≥ kv_dim) apart,
 * allowing cache-line aligned KV cache access.
 */
void ds_causal_attention_aligned(float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int q_offset, int kv_stride);

/*
 * Reference Sliding Window Attention (R-SWA) for Unlimited-OCR.
 * During decode, attention only attends to two non-contiguous KV ranges:
 *   1. Reference: positions [0..prefill_len-1] (visual + prompt tokens, always kept)
 *   2. Window: positions [window_start..kv_seq_len-1] (recent text tokens)
 * Positions [prefill_len..window_start-1] are skipped (out of window).
 *
 * For seq_q=1 decode: compute Q@K^T for both ranges, softmax over combined,
 * then compute weighted sum over V from both ranges.
 */
void ds_rswa_attention_aligned(float *out, const float *Q, const float *K_base, const float *V_base,
                                int seq_q, int n_heads, int n_kv_heads, int head_dim,
                                float scale, int kv_stride,
                                int prefill_len, int window_start, int kv_seq_len);

/* BF16 KV 感知的 R-SWA 入口（r2 任务2）: K/V 为 void*（bf16 时按 uint16 行读，
 * 元素 = bf16 精确展开；warmup（window_start==prefill_len）与 F32 全量因果逐位一致） */
void ds_rswa_attention_aligned_bf16(float *out, const float *Q, const void *K_base, const void *V_base,
                                    int seq_q, int n_heads, int n_kv_heads, int head_dim,
                                    float scale, int kv_stride,
                                    int prefill_len, int window_start, int kv_seq_len, int bf16);

/*
 * Mixed attention for DeepEncoder V2: visual tokens use bidirectional,
 * causal flow queries use causal attention.
 * visual_len: number of visual tokens (bidirectional)
 * total_len: visual_len + n_causal_queries (visual + causal flow queries)
 */
void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

/* RoPE: compute cos/sin for positions (NeoX-style) */
void ds_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                           int seq, int head_dim, float theta);

/* Apply RoPE to Q or K (in-place) */
void ds_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                         int seq, int n_heads, int head_dim);

/* 2D position embeddings for vision tokens (row + column) */
void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols,
                                        int embed_dim);

/* ========================================================================
 * MoE Operations
 * ======================================================================== */

/* Compute router gate scores: scores[n_experts] = gate_weight[n_experts, hidden] @ x[hidden] */
void ds_moe_router(float *scores, const float *x, const float *gate_weight,
                   int hidden, int n_experts);
void ds_moe_router_bf16(float *scores, const float *x, const uint16_t *gate_weight_bf16,
                          int hidden, int n_experts);

/* Select top-K experts from gate scores. Returns expert indices and weights.
 * 无放回选出得分最高的 top_k 个；top_weights 经选中子集上的 softmax
 * 重归一化（Σ=1）。本模型典型 n_experts=64、top_k=6。
 * 注意：解码器真实路径在 ds_moe_decoder.c 内联实现（全体 softmax 后取
 * 概率、不 renorm，对齐 Python），本函数供测试/工具使用。 */
void ds_moe_top_k(int *top_indices, float *top_weights, const float *scores,
                  int n_experts, int top_k);

/* Single expert SwiGLU forward: out = down(SiLU(gate(x)) * up(x)) */
void ds_expert_forward(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate,
                       float *gate_buf, float *up_buf,
                       float *gate_up_buf, float *hidden_buf);

/* Fused gate+up expert forward: gate_up = [gate_W; up_W] @ x, then split & SwiGLU.
 * gate_up_fused_bf16 is [2*intermediate, hidden] with gate rows first, then up rows.
 * Uses one matvec instead of two — better cache reuse of x, fewer thread dispatches. */
void ds_expert_forward_fused(float *out, const float *x,
                              const uint16_t *gate_up_fused_bf16,
                              const uint16_t *down_bf16,
                              int hidden, int intermediate,
                              float *gate_up_buf, float *gate_buf,
                              float *up_buf, float *hidden_buf);

/* ds_expert_forward 的 F32 权重 + sgemm 实验路径（DS_USE_F32_EXPERTS）；
 * 数值与 matvec 路径舍入序列不同，取不到 F32 视图时自动回退 */
void ds_expert_forward_f32(float *out, const float *x,
                            const uint16_t *gate_bf16, const uint16_t *up_bf16,
                            const uint16_t *down_bf16,
                            int hidden, int intermediate,
                            float *gate_buf, float *up_buf,
                            float *gate_up_buf, float *hidden_buf);

/* Legacy wrapper that allocates internally */
void ds_expert_forward_legacy(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate);

/* Combine expert outputs with router weights
 * output[hidden] = Σ_k top_weights[k]·expert_outputs[k,hidden]；
 * expert_outputs 按 [top_k, hidden] 排布，未选中槽位权重应为 0 */
void ds_expert_combine(float *output, const float *expert_outputs,
                       const int *top_indices, const float *top_weights,
                       int top_k, int hidden);

/* ========================================================================
 * Sampling
 * ======================================================================== */

/* Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits */
int ds_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                           int in_dim, int out_dim);

/* Streaming argmax with token ID exclusion (for V3 det tag suppression).
 * Same as ds_argmax_matvec_bf16 but skips the specified token IDs. */
int ds_argmax_matvec_bf16_excluding(const float *x, const uint16_t *W_bf16,
                                       int in_dim, int out_dim,
                                       const int *exclude_ids, int n_exclude);

/* Compute dot product of x with a single row of W_bf16 (for selective logit).
 * Returns the logit value for token at row index `tok_id`. */
float ds_bf16_dot_row(const float *x, const uint16_t *W_bf16,
                      int in_dim, int tok_id);

/* Compute logits = W_bf16 @ x + bias, storing full [out_dim] vector */
void ds_bf16_matvec_pub(float *y, const float *x, const uint16_t *W_bf16,
                         const float *b, int in_dim, int out_dim);

/* Grouped row-parallel matvec (decode MoE fast path, r4):
 * out[u * rows_per_unit + r] = W[u][r] · X[u]  for all u in [0,n_units), r in [0,rows_per_unit)
 *
 * 所有单元的输出行合并为一个线程池并行区——旧实现每专家 2-3 次独立 matvec，
 * 每层 ~19 个微小型并行区，barrier 同步开销大于矩阵本身（t1 与 t14 解码同速）。
 * 输入按单元给出（X[u]）: gate/up 共享同一 x，down 用各自专家的 SwiGLU 输出。
 * 行对（2r, 2r+1）永不跨线程拆分，每行仍由同一 NEON 内核按成对路径累加
 * ——落单行会走不同的求和顺序（2 组累加器 vs 8 组），故必须以行对为切分粒度
 * 才能保证与逐专家串行版本逐位一致。rows_per_unit 必须为偶数。 */
void ds_grouped_matvec_bf16(float *out, const float *const *X, const uint16_t *const *W,
                             int n_units, int rows_per_unit, int in_dim);

/* ========================================================================
 * Threading
 * ======================================================================== */

/* Set number of threads for parallel operations (default: 1)
 * n∈[1,16]；非线程安全，须在推理开始前调用（内部会销毁重建线程池） */
void ds_set_threads(int n);
int  ds_get_threads(void);

/* Get number of available CPU cores (macOS sysctl / Linux sysconf，失败回退 1) */
int ds_get_num_cpus(void);

/* Global verbose flag
 * ds_verbose: 日志级别（≥1 打印 dispatch 选择，≥2 打印线程池/缓存配置）；
 * ds_bf16_simulate_python: 置 1 时在专家前向的关键中间量上做 BF16 截断，
 *   逐位对齐 Python 的 BF16 中间精度（调试用，有性能代价） */
extern int ds_verbose;
extern int ds_bf16_simulate_python;

/* Thread pool (public for use by ds_quantize.c etc.)
 * ds_parallel_fn_t: 并行区函数——tid∈[0,n_threads) 各领一个分片，分片写区间
 *   必须互不相交；ds_parallel_for: fork-join 入口（n_threads≤1 时零开销直调，
 *   实现细节见 ds_kernels.c 的 ds_worker_loop/ds_parallel_for 注释） */
typedef void (*ds_parallel_fn_t)(int tid, int n_threads, void *arg);
void ds_dispatch_ensure_init(void);
void ds_parallel_for(ds_parallel_fn_t fn, void *arg);
int ds_get_threads(void);   /* 与上方重复声明（历史遗留），语义相同 */

/* ========================================================================
 * Timing Utilities
 * ======================================================================== */

/* High-resolution timer in seconds (uses gettimeofday) */
static inline double ds_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================
 * Fast Math Approximations
 * ======================================================================== */

/* Fast exp approximation (Schraudolph 1999).
 * Relative error < 1.5%, ~5x faster than expf() on ARM NEON.
 * Available as inline for use across translation units.
 * 原理：利用 IEEE754 位布局，把"指数字段 = 127·2^23 + x·2^23/ln2"直接当成
 * float 读出——指数域给出 e^x 的量级，尾数域线性近似 2^frac（分段线性逼近）。
 * 12102203.0f ≈ 2^23/ln2，1064866805.0f 为 Schraudolph 调优的偏置校正常数
 * （非纯 127<<23）。注意力在线 softmax 与 SwiGLU 使用；编码器精度红线场景
 * 不用（docs/05 §10.2）。 */
static inline float ds_fast_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;
    union { uint32_t i; float f; } u;
    u.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return u.f;
}

/* ── BF16 存储的 K/V 行 × F32 Q 点积（V3/V4 BF16 KV 缓存读取路径）──────
 * bf16→f32 展开是精确的（左移 16 位），点积顺序与 F32 存储一致；
 * 数值差异仅来自存入时的 bf16 舍入（任务书 r2 任务 2 授权的有界实验）。 */
float ds_dot_bf16xf32(const uint16_t *k, const float *q, int n);
void ds_axpy_bf16xf32(float *o, const uint16_t *v, float w, int n);

/* ========================================================================
 * BF16 → F32 Weight Conversion (for sgemm cache)
 * ======================================================================== */

/* Convert BF16 weights to F32 buffer. Caller must free the result.
 * 转换是精确的位展开（<<16），无舍入 */
float *ds_bf16_to_f32_alloc(const uint16_t *src, size_t n);

/* Batched BF16→F32 conversion into pre-allocated dst
 * 大缓冲（≥64K 元素）自动切线程池并行 */
void ds_bf16_to_f32_convert(float *dst, const uint16_t *src, size_t n);

/* NEON-optimized batch BF16↔F32 conversion (auto-dispatched)
 * 反向（F32→BF16）走 +0x8000 进位舍入；NEON 平台用向量化实现 */
void ds_bf16_to_f32_batch(float *dst, const uint16_t *src, size_t n);
void ds_f32_to_bf16_batch(uint16_t *dst, const float *src, size_t n);

/* ========================================================================
 * SGEMM-accelerated matvec (for large output dims like LM head)
 * ======================================================================== */

/* y[out_dim] = W_f32[out_dim, in_dim] @ x[in_dim] using cblas_sgemm.
 * W_f32 must be pre-converted from BF16. Thread-safe. */
void ds_f32_matvec_sgemm(float *y, const float *x, const float *W_f32,
                          int in_dim, int out_dim);

/* NEON-vectorized BF16 roundtrip: truncate F32 array to BF16 precision.
 * Used by DS_BF16_SIMULATE_PYTHON to match Python's BF16 intermediate values
 * without the overhead of per-element scalar bit manipulation. */
void ds_bf16_truncate_array(float *data, int n);

/* ========================================================================
 * Runtime BF16 kernel dispatch
 * ======================================================================== */

/* Initialize optimal BF16 matvec kernel for current CPU.
 * Called automatically on first matvec, or explicitly via ds_load().
 * On Apple M2+: uses bfdot (2× throughput). On M1: uses vshll+vfma. */
void ds_kernels_dispatch_init(void);

#endif /* DS_KERNELS_H */
