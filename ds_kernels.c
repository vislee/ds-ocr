/*
 * ds_kernels.c - Math kernels for DeepSeek-OCR inference
 * ds_kernels.c — DeepSeek-OCR 推理引擎数学内核实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】所有数学运算的底层实现
 * ─────────────────────────────────────────────────────────────────────
 * 提供矩阵乘法、注意力计算、归一化、激活函数、MoE路由等核心运算。
 * 是整个推理引擎中调用最频繁的模块——12层解码器每层都调用数十次。
 *
 * 【三级内核架构】(详见 ds_kernels_impl.h)
 *   本文件(ds_kernels.c) = 公共逻辑 + 平台无关的参考实现
 *   ds_kernels_generic.c  = 纯C99可移植实现
 *   ds_kernels_neon.c     = ARM NEON SIMD优化（Apple Silicon）
 *   ds_kernels_avx.c      = x86 AVX2+FMA优化（Intel/AMD）
 *
 * 编译时通过宏自动选择最优实现:
 *   #ifdef __ARM_NEON → 使用 NEON 版本
 *   #elif __AVX2__    → 使用 AVX2 版本
 *   #else             → 使用 generic 版本
 *
 * 【关键函数对照表】
 *   ds_linear_nobias_bf16()  — BF16权重矩阵×F32向量（解码器核心操作）
 *   ds_causal_attention()    — 因果注意力（解码器，带GQA支持）
 *   ds_rswa_attention_aligned() — R-SWA注意力（V3，reference+window两段）
 *   ds_rms_norm()            — RMSNorm（解码器归一化）
 *   ds_rms_norm_per_head()   — Per-head RMSNorm（DeepSeek特色Q/K归一化）
 *   ds_swiglu_multiply()     — SwiGLU激活（SiLU(gate)*up）
 *   ds_moe_router_bf16()     — MoE路由器（gate + softmax + topK）
 *   ds_argmax_matvec_bf16()  — 流式argmax（不分配129280个logits）
 *   ds_compute_rope_neox()   — RoPE位置编码预计算
 *   ds_apply_rope_neox()     — 应用RoPE旋转
 *
 * Adapted from antirez/qwen-asr project.
 * ═══════════════════════════════════════════════════════════════════════
 */

#include "ds_kernels.h"
#include "ds_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

/* Helper: truncate F32 to BF16 precision and back (matches Python BF16 intermediate precision)
 * 【BF16 位级原理】BF16 = F32 截掉低 16 位：两者共用 1 位符号 + 8 位指数，BF16 只留高 7 位尾数。
 *   F32→BF16：给低 16 位加上"半个最低保留位" 0x8000 再清零低 16 位——进位即入舍入
 *   （tie 时向绝对值增大方向舍入，与 IEEE 就近偶数舍入仅在恰好 tie 时不同）；
 *   BF16→F32：位模式 <<16 低位补零，这一步是【精确】的，只还原不损失。
 * Scalar reference for the SIMD rounding paths (e.g. the vaddq/vandq BF16 conversion);
 * kept even when unused so numerical debugging can call it directly. */
static __attribute__((unused)) inline float ds_f32_to_bf16_to_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u = (u + 0x8000) & 0xFFFF0000u;  /* 就近舍入（精确半途恒进位 = ties away from zero，非严格 RNE，见 docs/03 §3.7） */
    float r;
    memcpy(&r, &u, sizeof(r));
    return r;
}

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ds_fast_expf now provided by ds_kernels.h (inline) */

/* ========================================================================
 * Runtime BF16 dispatch — selects optimal matvec kernel at startup.
 *
 * On Apple M2+ (FEAT_BF16), uses bfdot instructions for 2× throughput.
 * On Apple M1 or other ARM without FEAT_BF16, falls back to vshll+vfma.
 * On x86, the AVX/generic path is selected at compile time.
 * ======================================================================== */

#ifdef __ARM_NEON
ds_bf16_matvec_fused_fn_t ds_bf16_matvec_fused_impl_ptr = ds_bf16_matvec_fused_neon;
ds_argmax_bf16_range_fn_t ds_argmax_bf16_range_impl_ptr = ds_argmax_bf16_range_neon;
#else
/* Non-ARM platforms: compile-time dispatch (no runtime check needed) */
ds_bf16_matvec_fused_fn_t ds_bf16_matvec_fused_impl_ptr = ds_bf16_matvec_fused_generic;
ds_argmax_bf16_range_fn_t ds_argmax_bf16_range_impl_ptr = ds_argmax_bf16_range_generic;
#endif

static int ds_dispatch_initialized = 0;

/* ds_kernels_dispatch_init - 运行时探测 CPU 能力，把 matvec/argmax 函数指针切到最优实现
 * 【算法】ARM 上读 hw.optional.arm.FEAT_BF16（macOS sysctl）或 AT_HWCAP2 的
 *   HWCAP2_BF16 位（Linux getauxval）：支持 FEAT_BF16 则选 bfdot 路径
 *   （一条指令 8 个 BF16 MAC，吞吐约为 vshll+vfma 路径的 2×），否则回退 vshll 加宽；
 *   x86 由 -march 在编译期定死 AVX2/AVX-512，无运行时探测需求。
 * 【为什么】同一份 Apple 二进制要同时跑 M1（无 bfdot）与 M2+（有），编译期无法
 *   二选一，只能把两份内核都编入、启动时切指针（"编译期定候选、运行时定人选"，
 *   详见 ds_kernels_impl.h 顶部注释）。
 * 【线程安全】初始化标志非原子，但各线程探测结果恒相同（写同值），实践无害。
 * 【参考】docs/03-推理引擎实战篇 §4（三级内核）；docs/09-量化与部署篇 §6。 */
void ds_kernels_dispatch_init(void) {
    if (ds_dispatch_initialized) return;
    ds_dispatch_initialized = 1;

#ifdef __ARM_NEON
    int has_bf16 = 0;
#ifdef __APPLE__
    /* sysctl for FEAT_BF16: hw.optional.arm.FEAT_BF16 == 1 on M2+ */
    size_t len = sizeof(has_bf16);
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &has_bf16, &len, NULL, 0) != 0)
        has_bf16 = 0;
#else
    /* Linux: check HWCAP2_BF16 from auxiliary vector */
#ifdef __linux__
#include <sys/auxv.h>
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1 << 14)
#endif
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    has_bf16 = (hwcap2 & HWCAP2_BF16) != 0;
#endif /* __linux__ */
#endif /* __APPLE__ */

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    if (has_bf16) {
        ds_bf16_matvec_fused_impl_ptr = ds_bf16_matvec_fused_neon_bf16;
        ds_argmax_bf16_range_impl_ptr = ds_argmax_bf16_range_neon_bf16;
        if (ds_verbose >= 1)
            fprintf(stderr, "BF16 dispatch: bfdot path (FEAT_BF16 detected)\n");
    } else
#endif
    {
        /* Fallback: vshll+vfma (no FEAT_BF16, e.g. Apple M1) */
        ds_bf16_matvec_fused_impl_ptr = ds_bf16_matvec_fused_neon;
        ds_argmax_bf16_range_impl_ptr = ds_argmax_bf16_range_neon;
        if (ds_verbose >= 1)
            fprintf(stderr, "BF16 dispatch: vshll path (no FEAT_BF16)\n");
    }
#elif defined(__AVX2__) && defined(__FMA__)
    ds_bf16_matvec_fused_impl_ptr = ds_bf16_matvec_fused_avx;
    ds_argmax_bf16_range_impl_ptr = ds_argmax_bf16_range_avx;
    if (ds_verbose >= 1)
        fprintf(stderr, "BF16 dispatch: AVX2 path\n");
#else
    if (ds_verbose >= 1)
        fprintf(stderr, "BF16 dispatch: generic path\n");
#endif
}

/* Auto-initialize on first matvec call (thread-safe via simple flag).
 * 惰性初始化：上层（ds_load 等）可显式提前调用，也可完全不调——第一次
 * matvec 经 bf16_matvec_fused → ds_dispatch_ensure_init 自动完成探测，
 * 保证任何路径都不会用到未初始化的函数指针。 */
void ds_dispatch_ensure_init(void) {
    if (!ds_dispatch_initialized) ds_kernels_dispatch_init();
}

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define DS_MAX_THREADS 16

typedef void (*ds_parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[DS_MAX_THREADS - 1];
    int tids[DS_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    ds_parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

/* ds_worker_loop - 常驻 worker 线程主循环（pthread 版 fork-join 的 worker 半边）
 * 【参数】arg 指向 tp.tids[i]（i∈[1,n_threads)），即本线程的 tid；主线程固定 tid=0。
 * 【协议】① 在 cond_work 上等 generation 前进（拿锁轮询→睡）；② 被唤醒后领取
 *   本轮代数/函数/参数并解锁执行 fn(tid, nt, arg)；③ 完成后 n_done++，最后一个
 *   完成者 signal cond_done 唤醒主线程（见 ds_parallel_for）。
 * 【为什么常驻】线程创建/销毁 ~几十 μs，而单次 matvec 并行区只有百 μs 级——
 *   复用线程把摊销成本降为 0。generation 代数计数让旧广播信号自然失效，
 *   替代了"每轮锁/解锁整套任务数组"的复杂簿记。
 * 【参考】docs/03-推理引擎实战篇 §8；qwen-asr 同名模式。 */
static void *ds_worker_loop(void *arg) {
    /* 【理论】常驻 worker 的 fork-join 模式（消除线程创建开销）：
     * - generation 计数器区分"第几轮任务"，worker 醒来后比对代数，
     *   旧唤醒信号被自然丢弃（避免条件变量竞态）。
     * - 主线程跑 fn(0,...) 同时 worker 跑 fn(tid,...)，join 用 n_done 计数 +
     *   cond_done 单向通知，barrier 开销 ~微秒级。
     * - 与 cblas_sgemm 的内部分线程"纵向"并行不同，这里按输出"横向"切分。 */
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)      /* while 而非 if：防虚假唤醒 */
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;      /* 领取本轮任务代号 */
        ds_parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);   /* 执行期间不持锁，各区间的写入互不相交 */

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)      /* 最后一个完成者唤醒主线程 */
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

/* ========================================================================
 * MoE Operations
 * ======================================================================== */

/* ds_moe_router - MoE 路由器：gate 线性层（F32 权重版，测试/对照用）
 * 【算法/数学】scores[e] = Σ_i W_gate[e][i]·x[i]，即 y = x @ W_gateᵀ；
 *   W_gate: [n_experts, hidden]（本模型 [64, 1280]）→ scores: [n_experts]。
 * 【注意】本函数【只做线性层，不含 softmax】——softmax 在调用方完成
 *   （解码器真实路径 ds_moe_decoder.c 的 moe_forward 内联做"全体 64 专家
 *   softmax 后再选 top-6"，与 Python/DeepSeek-V2 语义一致）。
 *   docs/02-MoE混合专家篇 §3.3 把 softmax 写进了 router 描述，与代码不符，以代码为准。
 * 【参数】scores[n_experts] 输出；x[hidden] 输入；gate_weight 行主序 F32 权重。
 * 【参考】docs/02 §3（路由机制）；PyTorch 等价: nn.Linear(hidden, n_experts, bias=False)。 */
void ds_moe_router(float *scores, const float *x, const float *gate_weight,
                   int hidden, int n_experts) {
    /* scores[e] = gate_weight[e, hidden] @ x[hidden] */
    ds_linear_nobias(scores, x, gate_weight, 1, hidden, n_experts);
}

/* ds_moe_router_bf16 - MoE 路由器（BF16 权重版，解码器真实路径使用）
 * 【算法】同 ds_moe_router，但权重为 BF16、matvec 时在线转 F32。
 * 【为什么用 BF16】路由决策必须与 Python 的 BF16 精度逐位对齐：两个排名相近的
 *   专家在 F32/BF16 下得分次序可能互换，一旦选错专家，输出就系统性偏离
 *   （docs/02 §3.4"精度细节"）。范围无损（8 位指数同 F32），只牺牲尾数。
 * 【参数】scores[n_experts] 输出原始得分（未 softmax）；x[hidden] F32 输入；
 *   gate_weight_bf16: [n_experts, hidden] 行主序 BF16。
 * 【参考】docs/02 §3.4；ds_moe_decoder.c moe_forward Step 1-2。 */
void ds_moe_router_bf16(float *scores, const float *x, const uint16_t *gate_weight_bf16,
                          int hidden, int n_experts) {
    /* scores[e] = gate_weight_bf16[e, hidden] @ x[hidden] (BF16 weights for precision matching) */
    ds_linear_nobias_bf16(scores, x, gate_weight_bf16, 1, hidden, n_experts);
}

/* ds_moe_top_k - 从专家得分中选 top-K 并做重归一化（独立工具函数）
 * 【算法/数学】① 无放回地选出得分最高的 top_k 个专家（选择法，K 轮扫描）；
 *   ② 对选中的 K 个分数再做一次 max-subtract softmax：w_k = e^(s_k-max)/Σe^(s_j-max)
 *   ——因为 exp 单调，这等价于"把已 softmax 过的概率在 top-K 子集内重归一化"，
 *   保证 Σw_k = 1 且未选中专家权重为 0。
 * 【参数】top_indices[top_k] 输出专家编号；top_weights[top_k] 输出归一化权重；
 *   scores[n_experts] 为输入得分（本工程用 softmax 后的概率，原始 logits 亦可，
 *   softmax 内部自带 max-subtract 对平移不变）；n_experts=64、top_k=6（典型值）。
 * 【注意】解码器真实路径（ds_moe_decoder.c）为对齐 Python/DeepSeek-V2 语义，
 *   改为"全体 64 专家 softmax 后直接取概率、不再 renorm"，未走本函数；
 *   本函数保留供测试（test_unit/test.c）与未来切换 renorm 语义使用。
 * 【复杂度】O(top_k × n_experts) = 6×64=384 次比较，<1μs，无需堆/partial sort。
 * 【参考】docs/02 §3.2（top-K 伪代码）。
 *
 * 【说人话】top-K 就是"64 个候选人里挑 6 个最能干的"。做法很土：先扫一遍
 *   找最高分（64 次比较），记下来；再扫一遍找第二高（跳过刚选中的），
 *   重复 6 次 → 6×64=384 次比较。为什么不排序？因为排序要 O(64 log 64)≈384
 *   次比较 + 额外内存，而 384 次比较在 3 GHz CPU 上不到 1 微秒——为了省这点
 *   时间引入堆/partial sort 的复杂度不划算。
 *   选完再 softmax 归一化，让 6 个权重加起来等于 1（"6 张票凑满 1 票"）。
 * 【边界】本函数是"先选后归一"的旧语义，解码器真实路径（ds_moe_decoder.c）
 *   已经改成"先对全部 64 个专家 softmax、再取 top-6、不再 renorm"——因为
 *   softmax 的分母必须含全部 64 个专家，先选后归一会让权重系统性偏大，
 *   与 Python 参考实现对不上。本函数保留给测试和未来切换用，别当成解码路径。
 *   另外 n_experts < top_k 时多余槽位填权重 0，combine 时乘 0 无贡献——
 *   这是防御分支，正常配置（64 专家 / top-6）不会走到。 */
void ds_moe_top_k(int *top_indices, float *top_weights, const float *scores,
                  int n_experts, int top_k) {
    /* 选择 top-K 路由专家（n_experts=64, top_k=6）
     *
     * K=6 ≈ n_experts 的 10%，top-6 覆盖了绝大多数的路由概率质量。
     * 仅计算选中的 6 个专家，跳过 58 个不活跃的专家，显著节省计算。
     *
     * 【当前实现：O(k·n_experts) 简单选择】
     * 优点：代码简单，无额外内存开销。
     * 缺点：对 64 个专家做 6 次扫描（6×64=384 次比较），不是最优的。
     * 优化方向：可以用 partial sort (nth_element) 将复杂度降到 O(n_experts)。
     * 当前 K=6, n_experts=64 规模下，384 次比较开销微不足道（< 1μs），不需要优化。
     *
     * 【路由权重处理】
     * top-6 的原始 softmax 分数经过 max-subtract 归一化后，
     * 再做 softmax 得到最终权重。这样权重之和 = 1，且只有 top-6 非零。*/
    int selected[DS_MAX_EXPERTS];
    float selected_scores[DS_MAX_EXPERTS];
    int n_selected = 0;

    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_score = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            /* Skip already selected */
            int already = 0;
            for (int j = 0; j < n_selected; j++) {
                if (selected[j] == e) { already = 1; break; }
            }
            if (already) continue;
            if (scores[e] > best_score) {
                best_score = scores[e];
                best = e;
            }
        }
        if (best >= 0) {
            selected[n_selected] = best;
            selected_scores[n_selected] = best_score;
            n_selected++;
        }
    }

    /* Softmax over top-K scores */
    /* max-subtract：先减最大值再 exp，防止 e^s 上溢（s-e^max ≤ 0，结果落在 [0,1]），
     * 平移不改变 softmax 结果（分子分母同乘 e^-max 被约掉） */
    float max_score = -1e30f;
    for (int k = 0; k < n_selected; k++) {
        if (selected_scores[k] > max_score) max_score = selected_scores[k];
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < n_selected; k++) {
        selected_scores[k] = expf(selected_scores[k] - max_score);
        sum_exp += selected_scores[k];
    }
    float inv_sum = 1.0f / sum_exp;

    for (int k = 0; k < top_k; k++) {
        if (k < n_selected) {
            top_indices[k] = selected[k];
            top_weights[k] = selected_scores[k] * inv_sum;   /* 乘 1/Σ 比除法快且省一次除法 */
        } else {
            /* n_experts < top_k 的防御分支：多余槽位填权重 0（combine 时乘 0 无贡献） */
            top_indices[k] = 0;
            top_weights[k] = 0.0f;
        }
    }
}

/* ds_expert_forward - 单个路由专家的 SwiGLU 前向（gate/up 权重分离版，CPU decode 路径）
 * 【参数】out[hidden] 输出；x[hidden] 输入；gate_bf16/up_bf16: [intermediate, hidden]、
 *   down_bf16: [hidden, intermediate] 均为行主序 BF16（本模型 hidden=1280、
 *   intermediate=moe_inter=896）；四个 buf 为调用方预分配的复用缓冲
 *   （gate_buf/up_buf/hidden_buf 各 [intermediate]，gate_up_buf 当前未用，仅保签名兼容）。
 * 【为什么复用缓冲】MoE 串行处理各专家，同一缓冲可跨专家复用，避免每专家
 *   malloc/free（docs/02 §8.2）。
 * 【参考】docs/02 §4（专家=小型 SwiGLU FFN）；docs/05 §4.5。 */
void ds_expert_forward(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate,
                       float *gate_buf, float *up_buf,
                       float *gate_up_buf, float *hidden_buf) {
    /* 分离 gate 和 up 的专家前向传播
     *
     * 架构: output = W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))
     * gate 和 up 分别做 BF16 matvec 得到 gate_buf 和 up_buf，
     * 然后 SwiGLU(gate_buf, up_buf) → hidden_buf，再做 down matvec。
     *
     * 【为什么有分离版本？】
     * 1. 直接读取 gate_bf16 和 up_bf16 两块分离的权重
     * 2. gate 和 up 的中间维度不同时也需要分离版本
     * 3. 对使用 INT4 量化时，gate/up 有各自的 scale/offset
     *
     * 此函数是 CPU 路径的专家前向，每路由专家调用一次。
     * 性能方面，gate 和 up 的 BF16 matvec 各占 ~50% 的计算时间。*/
    (void)gate_up_buf;  /* 保留在签名中以兼容调用方；SwiGLU 已改用 ds_swiglu_direct */
    ds_linear_nobias_bf16(gate_buf, x, gate_bf16, 1, hidden, intermediate);
    ds_linear_nobias_bf16(up_buf, x, up_bf16, 1, hidden, intermediate);

    /* Truncate gate/up to BF16 precision to match Python's BF16 matmul output.
     * Python: BF16 × BF16 → BF16 (linear output is BF16 before next op).
     * C:      F32 × BF16 → F32 (NEON matvec gives F32 output).
     * Without truncation, gate_buf and up_buf have extra F32 mantissa bits
     * that Python doesn't, causing SwiGLU results to diverge. */
    extern int ds_bf16_simulate_python;
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(gate_buf, intermediate);
        ds_bf16_truncate_array(up_buf, intermediate);
    }

    /* Fused gate+up for SwiGLU — use ds_swiglu_direct to avoid interleaving.
     * Previously: gate_up_buf[2*i] = gate[i], gate_up_buf[2*i+1] = up[i],
     * then ds_swiglu_multiply. The interleaving is cache-unfriendly.
     * ds_swiglu_direct computes SiLU(gate[i]) * up[i] directly. */
    ds_swiglu_direct(hidden_buf, gate_buf, up_buf, 1, intermediate);   /* SiLU(gate)⊙up [896] */

    /* Truncate SwiGLU output to BF16 (Python's down projection input is BF16) */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(hidden_buf, intermediate);
    }

    /* down projection */
    ds_linear_nobias_bf16(out, hidden_buf, down_bf16, 1, intermediate, hidden);

    /* Truncate final output to BF16 */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(out, hidden);
    }
}

/* Backward-compatible wrapper that allocates internally
 * ds_expert_forward_legacy - ds_expert_forward 的自分配包装（遗留接口）
 * 每次 4 次 malloc/free，仅供离线调试/脚本调用；真实推理请用外部
 * 预分配缓冲的 ds_expert_forward / ds_expert_forward_fused。 */
void ds_expert_forward_legacy(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate) {
    float *gate_buf = (float *)malloc(intermediate * sizeof(float));
    float *up_buf = (float *)malloc(intermediate * sizeof(float));
    float *gate_up_buf = (float *)malloc(2 * intermediate * sizeof(float));
    float *hidden_buf = (float *)malloc(intermediate * sizeof(float));

    if (!gate_buf || !up_buf || !gate_up_buf || !hidden_buf) {
        fprintf(stderr, "ds_expert_forward: allocation failed\n");
        free(gate_buf); free(up_buf); free(gate_up_buf); free(hidden_buf);
        return;
    }

    ds_expert_forward(out, x, gate_bf16, up_bf16, down_bf16,
                      hidden, intermediate, gate_buf, up_buf, gate_up_buf, hidden_buf);

    free(gate_buf); free(up_buf); free(gate_up_buf); free(hidden_buf);
}

/* Fused gate+up expert forward — one matvec instead of two.
 * gate_up_fused_bf16 layout: [2*intermediate, hidden] BF16
 *   rows 0..intermediate-1 = gate_W
 *   rows intermediate..2*intermediate-1 = up_W
 *
 * Pipeline:
 *   1. gate_up = gate_up_fused_bf16 @ x   [1280 → 1792] single matvec
 *   2. gate = gate_up[0:intermediate], up = gate_up[intermediate:2*intermediate]
 *   3. BF16 truncate (if ds_bf16_simulate_python)
 *   4. swiglu = SiLU(gate) * up
 *   5. BF16 truncate swiglu
 *   6. out = down_bf16 @ swiglu            [896 → 1280]
 *   7. BF16 truncate out
 *
 * 【为什么融合】x[1280] 在两次 matvec 间常驻 L1，一次读入喂 1792 行输出，
 *   且线程池只派发 1 个并行区（分离版要 2 个），减少 barrier 次数。
 * 【参数】out[hidden]；x[hidden]；gate_up_fused_bf16: [2*intermediate, hidden]
 *   （前 intermediate 行 = gate_W，后 = up_W）；其余同 ds_expert_forward。
 * 【参考】docs/05 §4.5（Fused Expert Forward）。 */
void ds_expert_forward_fused(float *out, const float *x,
                              const uint16_t *gate_up_fused_bf16,
                              const uint16_t *down_bf16,
                              int hidden, int intermediate,
                              float *gate_up_buf, float *gate_buf,
                              float *up_buf, float *hidden_buf) {
    /* Step 1: Fused gate+up projection — single matvec [hidden → 2*intermediate] */
    ds_linear_nobias_bf16(gate_up_buf, x, gate_up_fused_bf16, 1, hidden, 2 * intermediate);

    /* Step 2: Split fused output into gate and up */
    memcpy(gate_buf, gate_up_buf, intermediate * sizeof(float));
    memcpy(up_buf, gate_up_buf + intermediate, intermediate * sizeof(float));

    /* Step 3: BF16 truncate gate and up to match Python's BF16 precision */
    extern int ds_bf16_simulate_python;
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(gate_buf, intermediate);
        ds_bf16_truncate_array(up_buf, intermediate);
    }

    /* Step 4: SwiGLU = SiLU(gate) * up */
    ds_swiglu_direct(hidden_buf, gate_buf, up_buf, 1, intermediate);

    /* Step 5: BF16 truncate SwiGLU output */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(hidden_buf, intermediate);
    }

    /* Step 6: Down projection */
    ds_linear_nobias_bf16(out, hidden_buf, down_bf16, 1, intermediate, hidden);

    /* Step 7: BF16 truncate final output */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(out, hidden);
    }
}

/* ds_expert_combine - 用路由权重加权合并 top-K 个专家的输出（MoE 最后一步）
 * 【算法/数学】output[i] = Σ_k top_weights[k] · expert_outputs[k][i]，
 *   即 MoE(x) = Σ w_k·Expert_k(x)；expert_outputs 按 [top_k, hidden] 排布
 *   （本模型 top_k=6、hidden=1280）。
 * 【参数】output[hidden] 输出（先清零再累加）；top_indices 未用——top_weights
 *   中未选中槽位已置 0，乘 0 自然消去。
 * 【为什么先 memset】专家输出是"加法贡献"而非覆盖，必须从 0 起步；
 *   外层 k 内层 i 的顺序保证对 expert_outputs 的读取连续（cache 友好）。
 * 【参考】docs/02 §3.2（output = Σ top_k_probs·Expert(x)）。 */
void ds_expert_combine(float *output, const float *expert_outputs,
                       const int *top_indices, const float *top_weights,
                       int top_k, int hidden) {
    /* output[hidden] = sum_k(top_weights[k] * expert_outputs[k, hidden]) */
    (void)top_indices;  /* 未选中专家的权重已置零，无需按索引跳过 */
    memset(output, 0, hidden * sizeof(float));
    for (int k = 0; k < top_k; k++) {
        float w = top_weights[k];
        const float *expert_out = expert_outputs + (size_t)k * hidden;
        for (int i = 0; i < hidden; i++) {
            output[i] += w * expert_out[i];
        }
    }
}

/* ========================================================================
 * Mixed Attention (DeepEncoder V2: bidirectional for visual, causal for flow queries)
 * ======================================================================== */

/* ds_mixed_attention - DeepEncoder V2 混合注意力：视觉 token 双向 + 因果流查询因果
 * 【算法/数学】同一组 K/V 上按行施加两种掩码：
 *   - 行 i∈[0, visual_len)：双向——Q_i 与【全部】visual_len 个 K/V 做注意力；
 *   - 行 i∈[visual_len, total_len)：因果——Q_i 只看 [0, i]（全部视觉 token
 *     + 之前的流查询，k_end=i+1）。
 *   每行用 online softmax（O(1) 额外内存）：维护运行最大值 m 与指数和 Σ，
 *   遇到更大 score 时以 correction=e^(m_old-m_new) 整体重缩放已累加输出——
 *   数学恒等，无需物化 [total_len,total_len] 注意力矩阵（docs/01 §5）。
 * 【为什么 F64】V2 编码输出对浮点扰动极敏感：近并列 argmax 一旦翻转即级联
 *   放大，回归门控实测后定为红线"编码侧注意力权重精度不可妥协"
 *   （docs/05 §10.2），故 QK 点积与 V 累加全程 double、最后一步才舍入 F32。
 * 【参数】out/Q/K/V 均为 [total_len, n_heads*head_dim] 行主序；每 head 切片
 *   独立计算；scale=1/sqrt(head_dim) 由调用方传入（稳定 softmax 梯度）。
 * 【参考】docs/01 §9.3（混合注意力）；PyTorch 等价: 双向/因果两次 attention 的逐行拼合。 */
void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale) {
    int hidden = n_heads * head_dim;
    double dscale = (double)scale;

    /* Allocate double buffers for V accumulation (float64 precision)
     * o_row_d: 当前 (head, 行) 的加权 V 累加器 [head_dim]；v_buf: 供重缩放
     * 的临时行。仅 head_dim×8B×2，栈外小分配，换取整行 F64 精度。 */
    double *o_row_d = (double *)malloc(head_dim * sizeof(double));
    double *v_buf = (double *)malloc(head_dim * sizeof(double));

    for (int h = 0; h < n_heads; h++) {
        /* Visual tokens: bidirectional attention over visual tokens */
        for (int i = 0; i < visual_len; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            double max_score = -1e30;
            double sum_exp = 0.0;
            for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

            /* 双向注意力：j 遍历全部视觉 token，无掩码（docs/01 §9.1） */
            for (int j = 0; j < visual_len; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                /* Float64 QK dot product */
                double score = 0.0;
                for (int d = 0; d < head_dim; d++)
                    score += (double)q_row[d] * (double)k_row[d];
                score *= dscale;   // 缩放 1/sqrt(head_dim)：把 QKᵀ 方差压回 ~1，防 softmax 饱和

                /* online softmax 分支一：出现新的行最大值。
                 * 以 correction=e^(m_old-m_new) 把旧累加器与旧 Σ 整体重缩放
                 * （Σ 还要 +1 计入当前行，其 exp 权重恰为 1）；
                 * 本行 V 也乘同一 correction 后并入——展开即
                 * o = (Σ旧·e^旧 + v·e^新)，与直接算 e^(score-m) 数学恒等，
                 * 却保证指数参数永远 ≤ 0，不会溢出。 */
                if (score > max_score) {
                    double correction = exp(max_score - score);
                    sum_exp = sum_exp * correction + 1.0;
                    for (int d = 0; d < head_dim; d++) {
                        v_buf[d] = (double)v_row[d] * correction;
                        o_row_d[d] = o_row_d[d] * correction + v_buf[d];
                    }
                    max_score = score;
                } else {
                    /* online softmax 分支二（常见路径）：score ≤ 当前最大值，
                     * 权重 wt=e^(score-m) ∈ (0,1]，直接累加即可 */
                    double wt = exp(score - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] += wt * (double)v_row[d];
                }
            }

            /* 收尾：除以 Σ 得到归一化输出（未除前 o 是"未归一的指数加权和"） */
            if (sum_exp > 0.0) {
                double inv_sum = 1.0 / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    o_row[d] = (float)(o_row_d[d] * inv_sum);
            } else {
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
            }
        }

        /* Causal flow queries: attend to all visual tokens + previous causal queries */
        for (int i = visual_len; i < total_len; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            double max_score = -1e30;
            double sum_exp = 0.0;
            for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

            /* 因果掩码：流查询 i 只能看 [0, i]——k_end=i+1 覆盖全部视觉 token
             * 与此前到达的流查询（"按序收集信息"，docs/01 §9.3）。
             * 下面 j 循环内的 online softmax 与视觉段逐行相同，不再重复注释 */
            int k_end = i + 1;
            if (k_end > total_len) k_end = total_len;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                /* Float64 QK dot product */
                double score = 0.0;
                for (int d = 0; d < head_dim; d++)
                    score += (double)q_row[d] * (double)k_row[d];
                score *= dscale;

                if (score > max_score) {
                    double correction = exp(max_score - score);
                    sum_exp = sum_exp * correction + 1.0;
                    for (int d = 0; d < head_dim; d++) {
                        v_buf[d] = (double)v_row[d] * correction;
                        o_row_d[d] = o_row_d[d] * correction + v_buf[d];
                    }
                    max_score = score;
                } else {
                    double wt = exp(score - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] += wt * (double)v_row[d];
                }
            }

            if (sum_exp > 0.0) {
                double inv_sum = 1.0 / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    o_row[d] = (float)(o_row_d[d] * inv_sum);
            } else {
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
            }
        }
    }

    free(o_row_d);
    free(v_buf);
}

/* ========================================================================
 * 2D Position Embeddings for vision tokens
 * ======================================================================== */

/* ds_compute_2d_position_embeddings - 视觉 token 的 2D 正弦位置编码（行/列两半拼接）
 * 【算法/数学】embed_dim 对半拆：前半编码行号 row、后半编码列号 col，
 *   每半用 1D 正弦公式（docs/01 §4.2）：
 *     PE(pos, 2i)   = sin(pos · f_i)      f_i = 10000^(-2i/half),  half=embed_dim/2
 *     PE(pos, 2i+1) = cos(pos · f_i)
 *   频率随维度指数衰减：低维高频区分相邻行/列，高维低频区分远距位置。
 * 【参数】pos_embed[n_rows*n_cols, embed_dim] 输出，token 顺序 = patch 网格
 *   行优先展平（row = i/n_cols，col = i%n_cols）。
 * 【为什么 2D】图像 token 有 (row,col) 两个正交坐标，1D 序号表达不了
 *   "同行/同列"邻接关系；SAM/DeepEncoder 均为行列分别编码再拼接。
 * 【参考】docs/01 §4.4；docs/06-视觉编码器篇。 */
void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols,
                                         int embed_dim) {
    /* Split embedding dimension in half: row position + column position */
    int half = embed_dim / 2;
    int total = n_rows * n_cols;

    for (int i = 0; i < total; i++) {
        int row = i / n_cols;
        int col = i % n_cols;
        float *emb = pos_embed + i * embed_dim;

        /* Row position encoding (first half) */
        for (int d = 0; d < half; d++) {
            // f_d = 10000^(-2⌊d/2⌋/half)：偶 d 与奇 d+1 共用同一频率，
            // 构成一对 sin/cos（旋转正交基的经典写法）
            float freq = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)half);
            float angle = (float)row * freq;
            if (d % 2 == 0) emb[d] = sinf(angle);
            else emb[d] = cosf(angle);
        }

        /* Column position encoding (second half) */
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)half);
            float angle = (float)col * freq;
            if (d % 2 == 0) emb[half + d] = sinf(angle);
            else emb[half + d] = cosf(angle);
        }
    }
}

/* ds_set_threads - 设置线程池线程数（含旧线程池的销毁重建）
 * 【参数】n∈[1, DS_MAX_THREADS=16]，越界截断；n=1 为纯串行模式
 *   （ds_parallel_for 直接在主线程执行，零同步开销）。
 * 【协议】已有 >1 线程时：置 shutdown → broadcast 唤醒全部 worker →
 *   逐个 join → 按 n 重建（tid=1..n-1，主线程恒为 tid=0）。
 * 【为什么整池重建】pthread 线程数无法原地增减；推理期通常只调一次，
 *   重建成本可忽略。
 * 【参考】docs/03 §8（线程池与并行计算）。 */
void ds_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > DS_MAX_THREADS) n = DS_MAX_THREADS;

    /* Shutdown existing workers
     * shutdown=1 + broadcast：worker 在 cond_work 醒来后看到 shutdown 即退出
     * （见 ds_worker_loop），join 保证资源完全回收后再重建，避免句柄泄漏 */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) return;

    /* 只创建 n-1 个 worker：主线程自己充当 tid=0，并行区少一次线程切换 */
    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, ds_worker_loop, &tp.tids[i]);
    }

    if (ds_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

/* ds_get_threads - 读取当前线程池线程数（含主线程；1 = 纯串行模式） */
int ds_get_threads(void) {
    return tp.n_threads;
}

/* ds_get_num_cpus - 查询可用 CPU 核数（macOS sysctl hw.ncpu / Linux sysconf），
 * 供 main/ds_load 决定 ds_set_threads 的默认值；查询失败回退 1。 */
int ds_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* Dispatch work to all threads; main thread is tid=0
 * ds_parallel_for - fork-join 并行区入口：全部线程执行同一 fn(tid, n_threads, arg)
 * 【协议】主线程：锁 → 发布 fn/arg、n_done=0、generation++ → broadcast 唤醒
 *   worker → 解锁 → 自己跑 fn(0,...) → 等 n_done 达 n_threads-1（出口 barrier）。
 *   worker 侧协议见 ds_worker_loop；generation++ 使上一轮的迟到唤醒自然失效。
 * 【分片约定】fn 内部按 tid 自行切分数据（惯用 chunk=ceil(N/n_threads)），
 *   各分片写区间互不相交 → 执行期不持锁；同步开销仅入口广播 + 出口 barrier。
 * 【为什么主线程也干活】避免主线程空转等 barrier；n_threads≤1 时退化为
 *   直接调用 fn(0,1,arg)，完全零同步开销。
 * 【参考】docs/03 §8；docs/05 §10.3（按输出维度切分的"逐位一致"并行原则）。
 *
 * 【说人话】这就是"工头派活"：工头（主线程）在公告栏贴一张任务单（fn+arg），
 *   敲一下铃（broadcast），然后自己也抄起工具干活——不站在旁边看。8 个工人
 *   各自按工号 tid 认领自己那一段（第 tid 段），互不重叠，所以干活时不用
 *   互相打招呼（无锁）。干完各自在签到本上划一笔（n_done++），最后一个
 *   划完的人负责叫醒工头（signal）。工头收工前必须确认 8 个人都划了才走
 *   （出口 barrier），否则下一批活会踩到上一批没干完的数据。
 *   为什么主线程也要干活？8 个工人 + 1 个监工 = 9 份工资只出 8 份活，浪费。
 * 【边界】这套"静态切分"要求每段活的工作量大致均匀。如果某段特别重
 *   （比如 MoE 里某个专家特别慢），快的工人干完只能干等——这时该换
 *   动态任务队列（work stealing），代价是每次取任务要加锁。本仓库的
 *   matvec/attention 都是规整的等长行，静态切分正好，所以没上队列。
 *   另外 n_threads≤1 时直接串行调用，一行同步开销都不付（单核机器/调试）。 */
void ds_parallel_for(ds_parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    /* 出口 barrier：while（非 if）防虚假唤醒；最后一个完成的 worker 负责唤醒 */
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

/* ds_add_inplace - 原地逐元素加 a[i] += b[i]（残差累加，省一份输出缓冲） */
void ds_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

/* ds_mul_inplace - 原地逐元素乘 a[i] *= b[i]（如按位掩码/门控） */
void ds_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

/* ds_scale - 原地标量缩放 x[i] *= s（如注意力输出乘 1/Σexp） */
void ds_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

/* ds_copy - memcpy 包装：dst ← src（n 个 F32，编译器可向量化） */
void ds_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* ds_matmul_t - C = A @ Bᵀ 的批量矩阵乘（prefill 阶段的主力原语）
 * 【算法/数学】C[m,n] = Σ_k A[m,k]·B[n,k]；A[M,K]、B[N,K] 均行主序，
 *   B 以"转置参与乘法"的布局存储 → C[M,N]。prefill 的 QKV/Out 投影、
 *   共享专家 gate/up/down 全部收敛到这里。
 * 【为什么 B 存 [N,K]】权重天然按"每个输出元素一行"排布（与 matvec 通用），
 *   CblasTrans 让 BLAS 内部挑最优微内核，用户无需物化转置副本。
 * 【为什么 BLAS 快 5-10×】Accelerate/OpenBLAS 做了 cache 分块 + 寄存器分块
 *   + 指令级并行（docs/03 §4.3）；naive 三重循环仅为无 BLAS 平台保底。
 * 【参考】docs/05 §3（prefill 批量 sgemm，30s→0.93s 的关键）、docs/03 §4.3（BLAS 分块）。
 *
 * 【说人话】矩阵乘法是"两班人马对表"。朴素写法是：C 的每个格子都要把 A 的
 *   一整行、B 的一整列读一遍——算 2K 次乘加，却要搬 2K 个数，算术强度只有
 *   0.25 FLOP/B，CPU 大部分时间在等内存。
 *   BLAS 的做法是"分块"：把 A 切一块 64×Kc、B 切一块 Kc×64，算 64×64 的
 *   C 块。这一块里 A 和 B 的数据能塞进 L1 缓存，反复用 64 次——算术强度
 *   涨到 16 FLOP/B，翻过屋脊点（约 5），从"等内存"变成"算力吃满"。
 *   这就是为什么 prefill（几百个 token 一起算，M 大）用 BLAS 飞快，而
 *   decode（M=1，无从分块）用 BLAS 毫无收益——本仓库 decode 走自写 matvec。
 * 【边界】分块收益依赖矩阵足够大。M=1 时分块退化成原样；M=8（小 batch）
 *   强度只到 8，仍在屋脊点左侧。所以"用 BLAS 加速 decode"是无效的。
 *   另外 CblasTrans 让 BLAS 内部挑最优微内核，不需要用户物化转置副本——
 *   权重天然按"每个输出元素一行"排布（[N,K]），正好匹配。 */
void ds_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    /* 朴素三重循环（无 BLAS 平台的保底实现）：循环顺序 m→n→k 与行主序内存
     * 布局对齐——最内层 k 沿 A 的第 m 行、B 的第 n 行连续滑动，硬件预取器
     * 跟得上顺序访问；若把 n 放最内层则每步跳一整行 K 个 float，缓存行
     * 利用率崩塌（这就是"行主序遍历对缓存友好"的具体含义） */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {   /* C[m,n] = A 的第 m 行 · B 的第 n 行 */
            float sum = 0.0f;           /* 寄存器累加器：K 次乘加期间不碰内存 */
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];   /* 行主序线性索引：A[m*K+k] 即 A[m,k] */
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

/* ds_linear - 全连接 y = x @ Wᵀ + b（F32 权重，编码器/共享专家批量路径用）
 * 【参数】x[seq_len,in_dim]、W[out_dim,in_dim] 行主序、b[out_dim] 可 NULL、
 *   y[seq_len,out_dim]。BLAS 路径：sgemm 算完再逐行加 bias——sgemm 的 β 语义
 *   要求先把 b 广播进 C（多一趟写），不如算完一遍加干净。
 * 【参考】PyTorch: nn.Linear(in_dim, out_dim, bias)。 */
void ds_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    /* 朴素实现：逐 token、逐输出行做点积。x_row/w_row/y_row 先取行指针，
     * 内层循环免掉 s*in_dim、o*in_dim 的乘法地址计算；bias 直接作累加器
     * 初值（sum 从 b[o] 起步），省一趟单独的"逐行加 bias"扫描 */
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;      /* 第 s 个 token 的输入行 */
        float *y_row = y + s * out_dim;           /* 第 s 个 token 的输出行 */
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;  /* W 第 o 行 = 第 o 个输出特征的权重向量 */
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];   /* 两条行指针同步右移：顺序访问对缓存友好 */
            }
            y_row[o] = sum;
        }
    }
#endif
}

/* ds_linear_nobias - ds_linear 的 b=NULL 版；Transformer 的 QKV/FFN 投影
 * 均不带 bias（RMSNorm/LayerNorm 已含可学习偏移，extra bias 冗余，
 * 见 docs/01 §3.5）。F32 权重版路由器/专家也走这里。 */
void ds_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    ds_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer.
 * Large conversions (expert weights during prefill, LM head F32 fallback)
 * are memory-bound — split across the thread pool. Prefill converts ~5GB of
 * expert weights per run; the serial loop made that ~1.4s of pure copy. */
typedef struct {
    uint32_t *dst;
    const uint16_t *src;
    size_t n;
} ds_bf16_conv_task_t;

/* ds_bf16_conv_worker - BF16→F32 转换的线程体：按 tid 切连续区间，写互不相交
 * 【核心一行】dst[i] = src[i] << 16 —— BF16 位模式就是 F32 的高 16 位，
 *   左移补零即得精确 F32（无舍入、无查表，1 条移位指令完成"转换"）。 */
static void ds_bf16_conv_worker(int tid, int n_threads, void *arg) {
    ds_bf16_conv_task_t *t = (ds_bf16_conv_task_t *)arg;
    size_t chunk = (t->n + n_threads - 1) / n_threads;   // 每线程均分，向上取整
    size_t start = (size_t)tid * chunk;
    size_t end = start + chunk;
    if (end > t->n) end = t->n;
    for (size_t i = start; i < end; i++)
        t->dst[i] = ((uint32_t)t->src[i]) << 16;
}

/* ds_bf16_to_f32_buf - BF16→F32 缓冲转换的统一入口（大缓冲并行、小缓冲串行）
 * 【为什么 64K 阈值】65536 元素 = 256KB，此时转换耗时（内存带宽受限）才
 *   值得付一次 barrier 的同步开销；小转换进线程池反而被调度拖慢。 */
static void ds_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    if (n >= (size_t)1 << 16) {
        ds_bf16_conv_task_t task = { d, src, n };
        ds_parallel_for(ds_bf16_conv_worker, &task);
        return;
    }
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Reusable scratch buffer for bf16->f32 conversion
 * 进程级单例暂存区：cap 只增不减（几何扩容），prefill 每层反复转换
 * 同一批权重时避免反复 malloc/free MB 级缓冲。
 * 注意：无锁单例，依赖调用方串行使用（当前调用点均在单线程 prefill /
 * DS_USE_F32_EXPERTS 实验路径）。 */
static float *ds_bf16_scratch = NULL;
static size_t ds_bf16_scratch_cap = 0;

/* ds_bf16_get_scratch - 取能容纳 n 个 F32 的暂存区；不足时换更大的，
 * malloc 失败返回 NULL（调用方需回退到非转换路径）。 */
static float *ds_bf16_get_scratch(size_t n) {
    if (n > ds_bf16_scratch_cap) {
        free(ds_bf16_scratch);
        ds_bf16_scratch = (float *)malloc(n * sizeof(float));
        ds_bf16_scratch_cap = ds_bf16_scratch ? n : 0;
    }
    return ds_bf16_scratch;
}

typedef struct {
    const uint16_t *src;
    size_t n;
    float *dst_f32;
} ds_bf16_cache_entry_t;

static ds_bf16_cache_entry_t *bf16_cache = NULL;
static int bf16_cache_len = 0;
static int bf16_cache_cap = 0;
static size_t bf16_cache_bytes = 0;
static size_t bf16_cache_limit_bytes = 0;
static int bf16_cache_limit_init = 0;

/* bf16_cache_init_limit - 读取 DS_BF16_CACHE_MB 环境变量，设定 F32 缓存字节上限
 * 【策略】默认 0 = 关闭（省内存优先）；显式设 DS_BF16_CACHE_MB=<n> 后，
 *   缓存至多 n MB 的"已转换 F32 权重视图"。惰性执行一次。 */
static void bf16_cache_init_limit(void) {
    if (bf16_cache_limit_init) return;
    bf16_cache_limit_init = 1;

    /* Default OFF. Override with DS_BF16_CACHE_MB=<n> to enable. */
    unsigned long long mb = 0;
    const char *env = getenv("DS_BF16_CACHE_MB");
    if (env && env[0] != '\0') {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = v;
    }
    bf16_cache_limit_bytes = (size_t)(mb * 1024ULL * 1024ULL);

    if (ds_verbose >= 2) {
        fprintf(stderr, "BF16 cache: limit=%llu MB\n", mb);
    }
}

/* ds_bf16_get_cached_f32 - BF16 权重的 F32 副本缓存（命中免重复转换）
 * 【命中策略】以 (src 指针, n) 二元组线性查找——mmap 权重地址在进程生命周期
 *   内稳定，指针等值即同一张量，无需哈希。
 * 【淘汰策略】没有淘汰：总字节超限即拒绝新条目（返回 NULL，调用方走 scratch
 *   或 matvec 路径）。权重访问是周期性的（12 层轮流），全量缓存价值有限，
 *   简单"满了就不存"避免 LRU 的簿记开销。
 * 【线程安全】同 scratch：无锁，依赖串行调用路径。 */
static const float *ds_bf16_get_cached_f32(const uint16_t *src, size_t n) {
    bf16_cache_init_limit();

    for (int i = 0; i < bf16_cache_len; i++) {
        if (bf16_cache[i].src == src && bf16_cache[i].n == n) {
            return bf16_cache[i].dst_f32;
        }
    }

    if (bf16_cache_limit_bytes == 0) return NULL;

    size_t bytes = n * sizeof(float);
    if (bytes > bf16_cache_limit_bytes) return NULL;
    if (bf16_cache_bytes + bytes > bf16_cache_limit_bytes) return NULL;

    float *dst = (float *)malloc(bytes);
    if (!dst) return NULL;
    ds_bf16_to_f32_buf(dst, src, n);

    if (bf16_cache_len == bf16_cache_cap) {
        int new_cap = bf16_cache_cap > 0 ? bf16_cache_cap * 2 : 256;
        ds_bf16_cache_entry_t *tmp = (ds_bf16_cache_entry_t *)realloc(
            bf16_cache, (size_t)new_cap * sizeof(ds_bf16_cache_entry_t));
        if (!tmp) {
            free(dst);
            return NULL;
        }
        bf16_cache = tmp;
        bf16_cache_cap = new_cap;
    }

    bf16_cache[bf16_cache_len].src = src;
    bf16_cache[bf16_cache_len].n = n;
    bf16_cache[bf16_cache_len].dst_f32 = dst;
    bf16_cache_len++;
    bf16_cache_bytes += bytes;
    return dst;
}

/* ds_bf16_get_f32_view - 返回 BF16 权重的可读 F32 视图：缓存命中 → 缓存；
 * 未命中/未启用 → 一次性转入 scratch 返回（下次调用会覆盖）。
 * 【为什么需要 F32 视图】sgemm 只吃 F32：prefill（seq_len>1）时先把权重
 *   转成 F32 再进 cblas_sgemm，转换成本被多行输出摊薄（docs/05 §3）。 */
static const float *ds_bf16_get_f32_view(const uint16_t *src, size_t n) {
    const float *cached = ds_bf16_get_cached_f32(src, n);
    if (cached) return cached;

    float *scratch = ds_bf16_get_scratch(n);
    if (!scratch) return NULL;
    ds_bf16_to_f32_buf(scratch, src, n);
    return scratch;
}

/* Expert forward using F32 weights + BLAS sgemm (high precision, slower).
 * Used for prefill when DS_USE_F32_EXPERTS is set.
 * ds_expert_forward_f32 - 专家前向的 F32/sgemm 实验路径（DS_USE_F32_EXPERTS 启用）
 * 【差异】三个权重经 ds_bf16_get_f32_view 转 F32 后走 ds_linear_nobias（BLAS
 *   sgemm），数值与 matvec 路径的舍入序列不同（各输出行独立全精度累加）；
 *   任一视图取不到（缓存关闭且 scratch 不足）则整体回退 ds_expert_forward。
 * 【为什么单 token 不用】decode 的 matvec 是带宽受限，转 F32 反而多一倍
 *   读流量；仅 prefill 的批量 sgemm 能赚回转换成本（见 ds_linear_nobias_bf16）。 */
void ds_expert_forward_f32(float *out, const float *x,
                            const uint16_t *gate_bf16, const uint16_t *up_bf16,
                            const uint16_t *down_bf16,
                            int hidden, int intermediate,
                            float *gate_buf, float *up_buf,
                            float *gate_up_buf, float *hidden_buf) {
    size_t n_gate = (size_t)intermediate * hidden;
    size_t n_up = (size_t)intermediate * hidden;
    size_t n_down = (size_t)hidden * intermediate;
    const float *gate_f32 = ds_bf16_get_f32_view(gate_bf16, n_gate);
    const float *up_f32 = ds_bf16_get_f32_view(up_bf16, n_up);
    const float *down_f32 = ds_bf16_get_f32_view(down_bf16, n_down);
    if (!gate_f32 || !up_f32 || !down_f32) {
        ds_expert_forward(out, x, gate_bf16, up_bf16, down_bf16,
                          hidden, intermediate, gate_buf, up_buf, gate_up_buf, hidden_buf);
        return;
    }
    ds_linear_nobias(gate_buf, x, gate_f32, 1, hidden, intermediate);
    ds_linear_nobias(up_buf, x, up_f32, 1, hidden, intermediate);
    ds_swiglu_direct(hidden_buf, gate_buf, up_buf, 1, intermediate);
    ds_linear_nobias(out, hidden_buf, down_f32, 1, intermediate, hidden);
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 * bf16_matvec_fused - BF16 matvec 的统一入口：经函数指针跳到运行时选定的
 * 平台内核（bfdot / vshll+vfma / AVX / generic，见 ds_kernels_impl.h）。
 * 【为什么融合】BF16→F32 在线转换 + 点积 + bias 一趟完成，权重保持 BF16
 *   流式读取——decode 的 matvec 是内存带宽受限（算术强度≈1 FLOP/B），
 *   省下的就是赚到的（docs/03 §4.4、docs/09 §1）。
 * 【参考】docs/03 §4.4、docs/09 §1。
 *
 * 【说人话】BF16 matvec 是"边拆快递边用"：权重以 BF16 存在内存里（每个数
 *   2 字节，比 F32 省一半），算的时候现拆现用——读进来立刻展开成 F32 参与
 *   乘加，不落回内存。为什么不先全部转成 F32 再算？因为 3B 参数转 F32 要
 *   12 GB，翻倍；而"现拆现用"只需要几个寄存器，内存占用不变。
 *   为什么"融合"很重要？分开做的话，转换结果得先写回一块 F32 缓冲、再读回来
 *   做点积，多出 2 倍权重体积的内存流量。decode 的 matvec 本来就是内存带宽
 *   受限（算术强度≈1 FLOP/B，读一个数只做一次乘加），多一趟往返就是多一倍
 *   时间——所以融合省的不是"算"，是"搬"。
 * 【边界】"融合"只在 decode（单 token）时是决定性的。prefill 走 sgemm，
 *   一次读权重能摊到几百行输出上，算术强度翻几十倍，转换开销被摊薄，
 *   融不融合差别不大。另外这里经函数指针跳到运行时选定的内核（bfdot /
 *   vshll+vfma / AVX / generic），首次调用会惰性探测 CPU 能力——所以
 *   第一次 matvec 比后续慢一点点（微秒级，可忽略）。 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    ds_dispatch_ensure_init();
    ds_bf16_matvec_fused_impl_ptr(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads
 * 按【输出行】切分是多线程 matvec 的标准姿势：每线程处理一段连续行
 * （W 偏移 + 行数），x 向量只读、天然共享，无任何写冲突。 */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} ds_matvec_task_t;

/* ds_matvec_worker - matvec 线程体：tid 号线程算输出行 [start, end)，
 * 权重指针偏移 start*in_dim 对齐到自己的首行，bias 同步偏移 */
static void ds_matvec_worker(int tid, int n_threads, void *arg) {
    ds_matvec_task_t *t = (ds_matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

/* ds_bf16_matvec_threaded - 单/多线程 matvec 分派：n_threads≤1 直接调内核，
 * 否则把 out_dim 行按线程均分进并行区（decode 每步十几次 matvec 共用此入口） */
static void ds_bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    ds_matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    ds_parallel_for(ds_matvec_worker, &task);
}

/* ── Grouped row-parallel matvec（解码 MoE 快路径，r4）──
 * 以"行对"为切分粒度（见 ds_kernels.h 注释）：线程边界永远落在偶数行对齐处，
 * 每行的累加顺序与整段调用 bf16_matvec_fused 时完全一致 → 数值逐位不变。
 * 输入指针按单元给出：gate/up 共享同一 x，down 用各自专家的 SwiGLU 输出。 */
typedef struct {
    float *out;
    const float *const *X;
    const uint16_t *const *W;
    int n_units;
    int rows_per_unit;
    int in_dim;
} ds_grouped_matvec_task_t;

/* ds_grouped_matvec_worker - grouped matvec 线程体：把 n_units×rows_per_unit
 * 的总行数折叠成"行对序列"后按 tid 均分；跨 unit 边界的线程块被 while 循环
 * 拆成逐 unit 的子调用（每次调用保证 unit 内行对连续、输出指针正确偏移） */
static void ds_grouped_matvec_worker(int tid, int n_threads, void *arg) {
    ds_grouped_matvec_task_t *t = (ds_grouped_matvec_task_t *)arg;
    int pairs_per_unit = t->rows_per_unit / 2;
    int total_pairs = t->n_units * pairs_per_unit;
    int chunk = (total_pairs + n_threads - 1) / n_threads;
    int p = tid * chunk;                 // 本线程的起始行对（全局行对编号）
    int p_end = p + chunk;
    if (p_end > total_pairs) p_end = total_pairs;

    while (p < p_end) {
        int unit = p / pairs_per_unit;
        int pair0 = p - unit * pairs_per_unit;
        int pairs_here = pairs_per_unit - pair0;
        if (pairs_here > p_end - p) pairs_here = p_end - p;
        bf16_matvec_fused(t->out + (size_t)unit * t->rows_per_unit + 2 * pair0,
                          t->X[unit], t->W[unit] + (size_t)2 * pair0 * t->in_dim,
                          NULL, t->in_dim, 2 * pairs_here);
        p += pairs_here;
    }
}

/* ds_grouped_matvec_bf16 - 分组 BF16 matvec：n_units 个单元（专家/批行）合并
 * 进一个并行区（MoE prefill Gather-Compute-Scatter 的 gather 侧也复用此入口）
 * 【形状】out[n_units × rows_per_unit]、W 同布局、每单元 X[unit][in_dim]
 * 【为什么合并分发】契约与 barrier 开销分析见 ds_kernels.h 顶部长注释。 */
void ds_grouped_matvec_bf16(float *out, const float *const *X, const uint16_t *const *W,
                             int n_units, int rows_per_unit, int in_dim) {
    /* 实现 = 一个并行区跑完所有单元（契约见 ds_kernels.h 顶部长注释）：
     * 旧实现每专家独立 matvec → 每层 ~19 个微并行区，barrier 开销淹没矩阵
     * 本身；合并后 barrier 次数 O(1)，且行对粒度保证与串行版逐位一致 */
    ds_grouped_matvec_task_t task = { out, X, W, n_units, rows_per_unit, in_dim };
    ds_parallel_for(ds_grouped_matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_ds_matvec_task_t;

/* qkv_ds_matvec_worker - QKV 融合 matvec 线程体
 * 【分片方式】把 q_dim + 2*kv_dim 三段输出折叠成一个 [0, total_dim) 全局行号
 *   空间按 tid 均分，再由每线程把 [start,end) 拆解为落在 Q/K/V 三块权重里的
 *   子区间——一个并行区同时喂三张权重表，barrier 从 3 次降到 1 次；
 *   三块写缓冲（q/k/v）互不相交，无写冲突。 */
static void qkv_ds_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_ds_matvec_task_t *t = (qkv_ds_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;          /* 三段输出在全局行号空间 [0,total_dim) 中的分界 */
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {           /* 段 1：本线程区间头部落在 Q 权重行（块内行号 = 全局行号） */
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {   /* 段 2：与 K 权重行的交集；块内行号 = 全局行号 − Q 段宽度 */
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {   /* 段 3：与 V 权重行的交集，同理减去前两段宽度 */
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

/* ds_linear_nobias_bf16_qkv - Q/K/V 三投影融合 matvec（decode 单 token 快路径）
 * 【算法】q = Wq@x、k = Wk@x、v = Wv@x（各 [in_dim]→[q_dim]/[kv_dim]，无 bias）；
 *   本模型 in_dim=1280、q_dim=1280（10 头×128）、kv_dim=1280（MHA）。
 * 【为什么融合派发】x[1280] 在三次 matvec 间常驻 L1，且 3 次并行区合并为
 *   1 次（省 2 个 barrier）；多线程时行号空间折叠见 qkv_ds_matvec_worker。
 * 【参考】docs/01 §5.5（QKV 投影与 kernel fusion）。 */
void ds_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_ds_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    ds_parallel_for(qkv_ds_matvec_worker, &task);
}

/* ds_linear_nobias_bf16 - BF16 权重线性层 y = x @ Wᵀ 的统一入口（按 seq_len 分派）
 * 【契约】y[seq_len,out_dim]、x[seq_len,in_dim]、W_bf16[out_dim,in_dim] 行主序。
 * 【两条路径】
 *   seq_len==1（decode）：NEON/AVX BF16 matvec 流式点积。单 token 时这是
 *     内存带宽问题——权重读一次用一次，先转 F32 再 sgemm 要多付出"写出+读回"
 *     两倍流量，反而更慢。
 *   seq_len>1（prefill）：先转 F32（并行，见 ds_bf16_get_f32_view），再进
 *     cblas_sgemm。多 token 时算术强度上升为吞吐问题，sgemm 的寄存器分块
 *     远胜逐 token matvec（实测 prefill 30s→0.93s，docs/05 §3）。
 * 【参数】y/x 见契约；W_bf16 指向 mmap 的 BF16 权重（零拷贝）。
 * 【参考】docs/03 §3.3、§6（Prefill vs Decode）；docs/05 §3。 */
void ds_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    /* BF16 权重线性层 y = x @ Wᵀ 的统一入口，按 seq_len 分派：
     *
     * ┌ seq_len==1（decode）：NEON/AVX BF16 matvec，逐行流式点积。
     * │  理论：单 token 时这是内存带宽问题——权重一次读入即用，
     * │  转 F32 再 sgemm 反而要写出+读回 2 倍流量。
     * └ seq_len>1（prefill）：先转 F32（并行），再 cblas_sgemm。
     *    理论：多 token 时算术强度↑，变成吞吐问题，sgemm 的
     *    寄存器分块/重排序远胜逐 token matvec（实测 100×）。 */
    if (seq_len == 1) {
        ds_bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = ds_bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    ds_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

/* ds_linear_bf16 - ds_linear 的 BF16 权重版（y = x@Wᵀ + b，b 可 NULL）：
 * 分派逻辑与 ds_linear_nobias_bf16 相同，仅 decode 路径多传一个 bias 指针 */
void ds_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        ds_bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = ds_bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    ds_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* Public wrapper: compute logits = W_bf16 @ x + bias (single token, full output vector)
 * ds_bf16_matvec_pub - 对外暴露的完整 matvec（物化全部 out_dim 个输出）。
 * 与 ds_argmax_matvec_bf16 的区别：后者只回传 argmax 不物化 logits；
 * 需要完整 logits（重复惩罚/n-gram 阻断/温度采样）时用本函数。
 * decode 的 LM head 调用点在 ds_moe_decoder.c（lm_head 或 tied embeddings）。 */
void ds_bf16_matvec_pub(float *y, const float *x, const uint16_t *W_bf16,
                         const float *b, int in_dim, int out_dim) {
    ds_bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused).
 * argmax_bf16_range - 行区间流式 argmax 入口（转平台内核，区间接口供线程分块） */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    ds_dispatch_ensure_init();
    ds_argmax_bf16_range_impl_ptr(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[DS_MAX_THREADS];
    float best_val[DS_MAX_THREADS];
} ds_argmax_task_t;

/* ds_argmax_worker - argmax 线程体：把 out_dim 行按 tid 均分，各自扫出区间内
 * 最大值写入 best_idx/best_val[tid]，主线程归约（见
 * ds_argmax_matvec_bf16_excluding）。区间空时填 (0, -1e30f) 保证归约正确 */
static void ds_argmax_worker(int tid, int n_threads, void *arg) {
    ds_argmax_task_t *t = (ds_argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

/* ds_bf16_dot_row - 计算单个 token 行的点积 logit = W[tok_id]·x（选择性 logit）
 * 【算法】定点取出 lm_head 的第 tok_id 行做 BF16×F32 点积；
 *   NEON 路径：vshll_n_u16 把 4 个 BF16 左移加宽成 F32 位模式（精确转换），
 *   vfmaq_f32 乘加，双累加器 acc0/acc1 打破 FMA 依赖链；行尾标量收尾。
 * 【为什么存在】Selective Repetition Penalty（docs/05 §4.2）只需重算历史中
 *   出现过的 ~100 个 token 的 logit（vs 全词表 129280 个），每步省一次
 *   全量 LM head 扫描。
 * 【参数】x[in_dim]；W_bf16[out_dim,in_dim]；tok_id 为行号（token ID）。
 * 【参考】docs/05 §4.2；HF 语义: logits[tid]。 */
float ds_bf16_dot_row(const float *x, const uint16_t *W_bf16,
                      int in_dim, int tok_id) {
    const uint16_t *w_row = W_bf16 + (size_t)tok_id * in_dim;
    float sum = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= in_dim; k += 8) {
        // 一次载 8 个 BF16：低/高半各 4 个经 vshll 左移 16 位加宽成精确 F32，
        // 再与 x 的对应 4 元素 FMA；两个累加器交替填 FMA 的延迟气泡
        uint16x8_t bf = vld1q_u16(w_row + k);
        acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                         vld1q_f32(x + k));
        acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                         vld1q_f32(x + k + 4));
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < in_dim; k++) {
        uint32_t bits = ((uint32_t)w_row[k]) << 16;
        float wv;
        memcpy(&wv, &bits, sizeof(float));
        sum += wv * x[k];
    }
#else
    for (int k = 0; k < in_dim; k++) {
        uint32_t bits = ((uint32_t)w_row[k]) << 16;
        float wv;
        memcpy(&wv, &bits, sizeof(float));
        sum += wv * x[k];
    }
#endif
    return sum;
}

/* ds_argmax_matvec_bf16 - 流式 argmax：找 argmax(W_bf16 @ x) 但【不物化 logits】
 * 【算法】逐行点积 + 边算边比较，只留当前最大值（见 *_excluding 完整注释）。
 * 【为什么】129280×1280 的 lm_head ≈ 331MB BF16，贪心解码只需要 argmax；
 *   物化 logits 还要额外 129280×4B 缓冲与一遍写读（docs/03 §3.4、docs/05 §4.1，
 *   v0.5→v0.9 单步 ~63ms→8ms 的主要来源）。 */
int ds_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    return ds_argmax_matvec_bf16_excluding(x, W_bf16, in_dim, out_dim, NULL, 0);
}

/* True if id appears in the exclusion list (n is tiny, linear scan) */
static int ds_id_excluded(const int *exclude_ids, int n_exclude, int id) {
    for (int e = 0; e < n_exclude; e++)
        if (exclude_ids[e] == id) return 1;
    return 0;
}

/* Exact argmax over [0, out_dim) with excluded rows skipped.
 * Splits the range into contiguous segments around the sorted/deduped
 * excluded ids and runs the fast range kernel on each segment, so the
 * result is the TRUE best non-excluded row. The previous per-thread /
 * per-chunk fallbacks dropped a thread's (or chunk's) runner-up when its
 * best was excluded and could return a lower-ranked token. Cost: one
 * extra full-vocab scan (~8ms) on the rare ban-hit step. */
static int ds_argmax_rescan_excluding(const float *x, const uint16_t *W_bf16,
                                      int in_dim, int out_dim,
                                      const int *exclude_ids, int n_exclude) {
    int ex[64];
    int n = n_exclude < 64 ? n_exclude : 64;   // 禁用表极小（n-gram/EOS ban），截断保安全
    for (int i = 0; i < n; i++) ex[i] = exclude_ids[i];
    /* Insertion sort + dedupe (n is tiny)
     * 排序去重是为了把禁用 ID 转成一组"要跳过的行号区间"，从而把词表切成
     * 若干连续段逐段调用快速 range 内核——每段的行内累加顺序与全量扫描
     * 完全一致，保证拿到的是真正的次优 token（见函数头注释） */
    for (int i = 1; i < n; i++) {
        int v = ex[i], j = i - 1;
        while (j >= 0 && ex[j] > v) { ex[j + 1] = ex[j]; j--; }
        ex[j + 1] = v;
    }
    int m = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || ex[i] != ex[i - 1]) ex[m++] = ex[i];

    int best = -1;
    float best_val = -1e30f;
    int start = 0;                       // 当前尚未扫描段的左端
    for (int k = 0; k < m; k++) {
        int e = ex[k];
        if (e < start) continue;         /* duplicate or unsorted leftover */
        if (e > out_dim - 1) break;      /* beyond vocab */
        if (e > start) {
            // 扫 [start, e)：禁用行 e 之前的连续段；e 本身被跳过（不参与比较）
            int chunk_best;
            float chunk_val;
            argmax_bf16_range(x, W_bf16, in_dim, start, e, &chunk_best, &chunk_val);
            if (chunk_val > best_val) { best_val = chunk_val; best = chunk_best; }
        }
        start = e + 1;
    }
    if (start < out_dim) {
        int chunk_best;
        float chunk_val;
        argmax_bf16_range(x, W_bf16, in_dim, start, out_dim, &chunk_best, &chunk_val);
        if (chunk_val > best_val) { best_val = chunk_val; best = chunk_best; }
    }
    return best >= 0 ? best : 0;
}

/* Argmax with token ID exclusion list (n-gram blocking, EOS warmup ban).
 * The common case (winner not excluded) costs one streaming argmax; when
 * the winner IS excluded we rescan exactly around the excluded rows.
 * ds_argmax_matvec_bf16_excluding - 带禁用表的流式 argmax（V3 det 标签抑制等）
 * 【算法】① 全词表流式 argmax（多线程分块归约或单线程直扫）；
 *   ② 冠军若不在禁用表 → 直接返回（零额外成本，绝大多数步）；
 *   ③ 否则 ds_argmax_rescan_excluding 精确重扫。
 * 【参数】exclude_ids[n_exclude] 禁用的 token ID（可为 NULL）；n_exclude=0 等价
 *   于普通 argmax。
 * 【为什么"重扫"而非"跳过"】早年的分线程/分块回退方案在"本线程最大值被禁"
 *   时会丢失该线程的次优解，可能返回排名更低的 token；按禁用行切段重扫
 *   保证数学上的真 argmax，代价只是禁用命中的极少数步多 ~8ms 全表扫描。 */
int ds_argmax_matvec_bf16_excluding(const float *x, const uint16_t *W_bf16,
                                       int in_dim, int out_dim,
                                       const int *exclude_ids, int n_exclude) {
    int best;
    if (tp.n_threads > 1) {
        ds_argmax_task_t task;
        task.x = x;
        task.W_bf16 = W_bf16;
        task.in_dim = in_dim;
        task.out_dim = out_dim;
        ds_parallel_for(ds_argmax_worker, &task);

        best = task.best_idx[0];
        float best_val = task.best_val[0];
        for (int i = 1; i < tp.n_threads; i++) {
            if (task.best_val[i] > best_val) {
                best_val = task.best_val[i];
                best = task.best_idx[i];
            }
        }
    } else {
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
    }

    if (n_exclude > 0 && ds_id_excluded(exclude_ids, n_exclude, best))
        return ds_argmax_rescan_excluding(x, W_bf16, in_dim, out_dim,
                                          exclude_ids, n_exclude);
    return best;
}

/* ds_matmul_t_bf16 - ds_matmul_t 的 BF16 权重版：C = A @ B_bf16ᵀ
 * M==1 时退化为 matvec（走 fused 内核，不物化 F32 副本）；
 * M>1 时转 F32 视图后进 sgemm（与 ds_linear_nobias_bf16 同一取舍） */
void ds_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M == 1) {
        ds_bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        const float *B_f32 = ds_bf16_get_f32_view(B_bf16, n);
        if (!B_f32) return;
        ds_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * BF16 → F32 Weight Conversion
 * ======================================================================== */

/* ds_bf16_to_f32_alloc - 一次性把 n 个 BF16 精确展开为 F32（malloc，调用方 free）
 * 转换即位模式左移 16 位，无舍入；用于编码器等需要整块 F32 权重的场景 */
float *ds_bf16_to_f32_alloc(const uint16_t *src, size_t n) {
    float *dst = (float *)malloc(n * sizeof(float));
    if (!dst) return NULL;
    ds_bf16_to_f32_convert(dst, src, n);
    return dst;
}

/* ds_bf16_to_f32_convert - 对外转换入口：转发到 ds_bf16_to_f32_buf
 * （内部按 64K 元素阈值决定是否并行），调用方自备 dst 容量 ≥ 4n 字节 */
void ds_bf16_to_f32_convert(float *dst, const uint16_t *src, size_t n) {
    /* Route through the shared converter — parallel for large buffers */
    ds_bf16_to_f32_buf(dst, src, n);
}

/* NEON-optimized batch conversion dispatch
 * 批量 BF16↔F16 转换的编译期双实现：__ARM_NEON 下直连 ds_kernels_neon.c 的
 * 向量化版本（每轮 8 元素 vshll/vsubn 级流水），否则回落到上方通用转换器
 * （其内部仍有 64K 阈值的线程池并行）。KV 缓存写入（ds_kv_store_row）和
 * 编码器预转换都经由此对入口。 */
#ifdef __ARM_NEON
extern void ds_bf16_to_f32_neon(float *dst, const uint16_t *src, size_t n);
extern void ds_f32_to_bf16_neon(uint16_t *dst, const float *src, size_t n);

/* ds_bf16_to_f32_batch - BF16→F32 批量展开（NEON 直连版）
 * 转换 = 位模式 <<16，无舍入、无误差；dst 容量须 ≥ 4n 字节 */
void ds_bf16_to_f32_batch(float *dst, const uint16_t *src, size_t n) {
    ds_bf16_to_f32_neon(dst, src, n);
}
/* ds_f32_to_bf16_batch - F32→BF16 批量压缩（NEON 直连版）
 * 就近舍入到 BF16（tie 向绝对值增大方向），与下方标量回退逐位一致 */
void ds_f32_to_bf16_batch(uint16_t *dst, const float *src, size_t n) {
    ds_f32_to_bf16_neon(dst, src, n);
}
#else
/* ds_bf16_to_f32_batch - 标量回退：转发通用转换器（内部按量并行） */
void ds_bf16_to_f32_batch(float *dst, const uint16_t *src, size_t n) {
    ds_bf16_to_f32_convert(dst, src, n);
}
/* ds_f32_to_bf16_batch 的标量回退：u+0x8000 进位实现"就近舍入"（tie 向绝对值
 * 增大方向），再右移 16 位取回高半——与 NEON vaddq/vandq 路径逐位一致 */
void ds_f32_to_bf16_batch(uint16_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, src + i, sizeof(u));
        u = (u + 0x8000) & 0xFFFF0000u;
        dst[i] = (uint16_t)(u >> 16);
    }
}
#endif

/* ========================================================================
 * SGEMM-accelerated matvec (for large output dims like LM head)
 * ======================================================================== */

/* ds_f32_matvec_sgemm - F32 权重 matvec 的 BLAS 封装：y = W_f32 @ x
 * 【用途】LM head 这类大输出维（本模型 129280）且权重已转成 F32 的 matvec。
 * 【行主序/转置陷阱】CBLAS 传统接口是 Fortran 列主序，这里用 CblasRowMajor
 *   显式声明行主序布局；W_f32 按 [out_dim, in_dim] 存储（每个输出元素一行，
 *   与各 matvec 内核通用），而数学上 y[1,out] = x[1,in] @ W[in,out] 需要
 *   "W 的转置参与乘法"，故 B 侧必须传 CblasTrans + ldb=in_dim——BLAS 内部
 *   按自己的存储序挑最优微内核，无需物化转置副本。若误传 NoTrans 会静默
 *   算出错误结果（行/列语义颠倒），这是 CBLAS 最常见的调用坑。
 * 【参数】y[out_dim]、x[in_dim]、W_f32[out_dim,in_dim] 行主序。
 * 【参考】docs/03 §4.3（BLAS 约定与陷阱）。 */
void ds_f32_matvec_sgemm(float *y, const float *x, const float *W_f32,
                          int in_dim, int out_dim) {
#ifdef USE_BLAS
    /* y = W_f32 @ x where W_f32 is [out_dim, in_dim], x is [in_dim], y is [out_dim]
     * Using sgemm: y[1, out_dim] = x[1, in_dim] @ W_f32^T[in_dim, out_dim]
     * cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
     *             M=1, N=out_dim, K=in_dim, 1.0, x, in_dim, W_f32, in_dim, 0.0, y, out_dim) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, out_dim, in_dim, 1.0f, x, in_dim, W_f32, in_dim, 0.0f, y, out_dim);
#else
    /* Fallback: naive matvec */
    for (int o = 0; o < out_dim; o++) {
        const float *w_row = W_f32 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += x[k] * w_row[k];
        y[o] = sum;
    }
#endif
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 *
 * 把每个输出位置对应的 kH×kW 感受野展平成 cols 的一列：列号 (ic,ki,kj) →
 * 行 col_row，列内按输出位置 (oh,ow) 排布；padding 越界处补 0（与卷积的
 * 零填充语义一致）。展开后卷积就是一次普通 sgemm（见 ds_conv2d）。
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;   /* 感受野左上角 = oh·stride−padding，再偏移 ki 到本核行 */
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;   /* 列方向同理；越界即 padding 补 0 */
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

/* ds_conv2d - 标准 2D 卷积（im2col + sgemm 实现，SAM patch embed / neck 压缩用）
 * 【算法/数学】out[c_out, H_out, W_out] = Σ_c,k w[c_out,c,kh,kw]·in[c,ih,iw]，
 *   H_out=(H_in+2p-kh)/s+1（同 W_out）；im2col 展开后等价于
 *   weight[c_out, C_in·kh·kw] @ cols[C_in·kh·kw, H_out·W_out] → 一次 sgemm。
 * 【为什么 im2col】把卷积降维成 GEMM 就能复用 BLAS 的极致优化，代价是
 *   patch_size×spatial_out 的中间缓冲（stride>1 时远小于输入本身）。
 * 【布局】in/out 均为 [C, H, W] 通道优先；bias 加在输出通道维。
 * 【参考】PyTorch: nn.Conv2d(..., bias)；docs/06-视觉编码器篇。 */
void ds_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* DEBUG: dump im2col for any 3x3 s1 conv with c_in=256
     * 【调试路径】设 DS_DUMP_CONV2_IM2COL=1 才激活，用于 C↔Python 数值对拍
     * （落盘 dump/c_conv2_im2col.bin）；未设环境变量时每次调用只付一次
     * getenv 的开销，正常推理不受影响 */
    {
        static int _conv2d_call_count = 0;
        _conv2d_call_count++;
        if (getenv("DS_DUMP_CONV2_IM2COL")) {
            fprintf(stderr, "[DUMP] ds_conv2d call #%d: c_in=%d c_out=%d h=%d w=%d kh=%d kw=%d s=%d p=%d h_out=%d w_out=%d\n",
                    _conv2d_call_count, c_in, c_out, h_in, w_in, kh, kw, stride, padding, h_out, w_out);
            if (c_in == 256 && c_out == 256 && kh == 3 && stride == 1 && padding == 1) {
                FILE *df = fopen("dump/c_conv2_im2col.bin", "wb");
                if (df) { fwrite(cols, sizeof(float), (size_t)patch_size * spatial_out, df); fclose(df); }
                fprintf(stderr, "[DUMP] *** neck conv2 im2col saved: %d x %d ***\n", patch_size, spatial_out);
            }
        }
    }

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int s = 0; s < spatial_out; s++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
            }
            out[oc * spatial_out + s] = sum;
        }
    }
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* ds_layer_norm - 标准 LayerNorm（SAM ViT / qwen-asr 编码器使用）
 * 【算法/数学】out = (x-μ)/sqrt(σ²+ε)·γ + β；每行两遍扫描（先 μ 再 σ²），
 *   AVX-512/AVX2 路径用 FMA 加速方差累加；ε 防止除零（调用方传 1e-5 等量级）。
 * 【参数】out/x: [seq_len, hidden]；weight=γ、bias=β 各 [hidden]（逐维仿射）。
 * 【与 RMSNorm 的区别】LayerNorm 减均值 + 加 β，RMSNorm 两者都省
 *   （见 ds_rms_norm 顶部理论块）；解码器用 RMSNorm，编码器用 LayerNorm。
 * 【参考】docs/01 §8.2；PyTorch: nn.LayerNorm(hidden, eps)。 */
void ds_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
        /* LayerNorm 是"逐 token 归一化"：统计量（μ、σ²）只在单行 hidden 维内
         * 计算，同一 seq 的其他行互不影响。三条 SIMD 路径只是把求和向量化，
         * 语义与标量版完全一致（水平求和：AVX-512 用 reduce_add，AVX2 用 hadd） */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;   /* μ = Σx / n，第一遍扫描完毕 */

        /* Compute variance */
        /* σ² = Σ(x-μ)²/n：第二遍扫描。两遍式比单遍 E[x²]−μ² 数值更稳——
         * 后者是大数减小数相消（catastrophic cancellation），μ² 接近 E[x²]
         * 时有效位被抹掉，可能出现负方差 */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        /* inv_std = 1/sqrt(σ²+ε)：ε 防方差为 0 的行除零；先求倒数，
         * 第三遍归一化循环里只需乘法（除法指令延迟 ~20 周期 vs 乘法 ~4） */
        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

/* ========================================================================
 * RMSNorm — Root Mean Square Layer Normalization
 * ========================================================================
 * 【理论】(Zhang & Sennrich, 2019)
 *   RMSNorm(x) = x / sqrt(mean(x²) + ε) · g
 * 与 LayerNorm 的区别：不减均值 μ、不算方差 σ²，只除以均方根 RMS。
 * 省去两次遍历（求 μ、求 σ²），在 LLM 推理里通常快 ~10-15%，
 * 且效果与 LayerNorm 相当——DeepSeek/Qwen/LLaMA 全系采用。
 *
 * 【本实现】hidden=1280，每次 decode 调 2×12+1=25 次（每层输入+post-attn+最终）。
 * 三条 SIMD 路径（AVX-512 / AVX2 / NEON）+ 标量回退，逻辑相同：
 *   第一遍：累加 Σx²      （FMA：乘加同一条指令）
 *   第二遍：x·rms_inv·g   （rms_inv = 1/sqrt(Σx²/n + ε)，先求倒数再乘，除法只做一次）
 *
 * 【说人话】RMSNorm 是"音量归一化旋钮"。一段话里有的词喊得震天响（数值大），
 *   有的词细声细气（数值小），直接送进下一层会让大数淹没小数。RMSNorm 先量
 *   一下这段话的"平均音量"（均方根 = 各元素平方的平均再开方），然后把整段
 *   按比例调到标准音量，最后乘一个可学习的"音色旋钮" g 让模型自己决定每个
 *   维度该放大还是缩小。
 *   和 LayerNorm 的差别就一句：LayerNorm 还要先把整段话"居中"（减均值），
 *   RMSNorm 不居中，只调音量。省掉"求均值"这一趟扫描，所以快 10-15%。
 *   为什么可以不居中？因为下一层的矩阵乘法里本来就有 bias 项，居中这件事
 *   模型自己会学——省一趟扫描，效果几乎不变。
 * 【边界】RMSNorm 对"整体偏移"不敏感，所以如果输入存在系统性直流偏置
 *   （比如某层输出整体偏大），它不会帮你消掉，只能靠后续层自己学。
 *   另外 ε 不能省：全零向量时 RMS=0，除零会出 NaN/Inf，ε 是兜底。
 *   数值上先算 1/sqrt 再乘（而不是逐个除），是因为除法延迟远高于乘法，
 *   1280 个元素能省下 1279 次除法。 */
void ds_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__ARM_NEON)
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            float32x4_t v0 = vld1q_f32(x_row + i), v1 = vld1q_f32(x_row + i + 4);
            acc0 = vfmaq_f32(acc0, v0, v0);
            acc1 = vfmaq_f32(acc1, v1, v1);
        }
        float sum_sq = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        /* rms_inv = 1/sqrt(mean(x²)+ε)：RMSNorm 的核心缩放因子。
         * eps（本模型 1e-5 量级，必须与 Python config 的 rms_norm_eps 逐位一致）
         * 防两类退化：全零行 sum_sq=0 时除零；极小行时除以近 0 数导致输出爆表。
         * 先求倒数再乘：归一化循环里 n 次乘法替代 n 次除法（除法 ~20 周期 vs 乘法 ~4） */
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__ARM_NEON)
        float32x4_t s = vdupq_n_f32(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            float32x4_t vx0 = vld1q_f32(x_row + j), vx1 = vld1q_f32(x_row + j + 4);
            float32x4_t vw0 = vld1q_f32(weight + j), vw1 = vld1q_f32(weight + j + 4);
            vst1q_f32(out_row + j, vmulq_f32(vmulq_f32(vx0, vw0), s));
            vst1q_f32(out_row + j + 4, vmulq_f32(vmulq_f32(vx1, vw1), s));
        }
        for (; j + 4 <= hidden; j += 4) {
            float32x4_t vx = vld1q_f32(x_row + j);
            float32x4_t vw = vld1q_f32(weight + j);
            vst1q_f32(out_row + j, vmulq_f32(vmulq_f32(vx, vw), s));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

/* ds_rms_norm_per_head - 对 Q/K 的每个注意力头【独立】做 RMSNorm（DeepSeek-V2 的 QK-Norm）
 * 【契约】x 为 [seq_len, n_heads*head_dim]，原地修改；weight/eps 语义同
 *   ds_rms_norm，但均值统计范围是单个 [head_dim] 段（本模型 10 头×128 维）。
 * 【为什么按头】注意力 logit = q·k/√d 的量级由各头自己的向量决定，逐头
 *   归一化把每个头的 logit 拉回稳定区间、抑制注意力熵塌缩；整行归一化
 *   做不到这一点。
 * 【参考】docs/01 §8.5；Dehghani et al. 2023（QK-Norm）。 */
void ds_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment
     * 【理论】QK-Norm（Dehghani et al. 2023）：对每个 head 的 Q/K 向量独立做
     * RMSNorm。作用是把注意力分数 logit = q·k/√d 的量级拉回稳定区间，
     * 抑制注意力熵塌缩。本工程 DeepSeek-V2 解码器 10 head × 128 维。*/
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            /* 1/sqrt(mean(x²)+eps)：统计范围仅本 head 的 head_dim 维（每个头
             * 有自己的 RMS）；eps 与解码器全局 RMSNorm 同源、防零向量除零，
             * 取值须与 Python 侧 qk-norm 的 eps 逐位一致 */
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Fused Residual Add + RMS Norm (NEON/AVX optimized)
 * out = x + residual,  norm_out = rms_norm(out, weight)
 * One pass for add + compute sum_sq, then second pass for norm.
 * Saves one full hidden-dim loop vs separate add then norm.
 * ======================================================================== */

/* ds_residual_rms_norm - 融合算子：out = x + residual，norm_out = RMSNorm(out)·weight
 * 【参数】out/norm_out/x/residual/weight 均 [hidden]（decode 单 token 的层尾路径）；
 *   out 与 norm_out 必须是两个不同缓冲（out 供下一层残差链使用），eps 语义同
 *   ds_rms_norm。
 * 【为什么融合】第一遍"边加边顺手累加 Σout²"把残差加与求和扫描合并成一趟，
 *   比先 ds_add_inplace 再 ds_rms_norm 少一遍 hidden 全量读；三条 SIMD 路径
 *   + 标量回退，公式与 ds_rms_norm 完全一致。
 * 【参考】docs/01 §8.4（残差 + 归一化的层结构）。 */
void ds_residual_rms_norm(float *out, float *norm_out,
                           const float *x, const float *residual,
                           const float *weight, int hidden, float eps) {
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= hidden; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        float32x4_t r0 = vld1q_f32(residual + i), r1 = vld1q_f32(residual + i + 4);
        float32x4_t s0 = vaddq_f32(x0, r0), s1 = vaddq_f32(x1, r1);
        vst1q_f32(out + i, s0);
        vst1q_f32(out + i + 4, s1);
        acc0 = vfmaq_f32(acc0, s0, s0);
        acc1 = vfmaq_f32(acc1, s1, s1);
    }
    float sum_sq = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < hidden; i++) {
        out[i] = x[i] + residual[i];
        sum_sq += out[i] * out[i];
    }
    float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
    float32x4_t s = vdupq_n_f32(rms_inv);
    int j = 0;
    for (; j + 8 <= hidden; j += 8) {
        float32x4_t v0 = vld1q_f32(out + j), v1 = vld1q_f32(out + j + 4);
        float32x4_t w0 = vld1q_f32(weight + j), w1 = vld1q_f32(weight + j + 4);
        vst1q_f32(norm_out + j, vmulq_f32(vmulq_f32(v0, w0), s));
        vst1q_f32(norm_out + j + 4, vmulq_f32(vmulq_f32(v1, w1), s));
    }
    for (; j < hidden; j++) norm_out[j] = out[j] * rms_inv * weight[j];
#elif defined(__AVX512F__) && defined(__FMA__)
    __m512 accv = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= hidden; i += 16) {
        __m512 sv = _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(residual + i));
        _mm512_storeu_ps(out + i, sv);
        accv = _mm512_fmadd_ps(sv, sv, accv);
    }
    float sum_sq = _mm512_reduce_add_ps(accv);
    for (; i < hidden; i++) { out[i] = x[i] + residual[i]; sum_sq += out[i] * out[i]; }
    float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
    __m512 scale = _mm512_set1_ps(rms_inv);
    int j = 0;
    for (; j + 16 <= hidden; j += 16) {
        __m512 v = _mm512_loadu_ps(out + j), w = _mm512_loadu_ps(weight + j);
        _mm512_storeu_ps(norm_out + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
    }
    for (; j < hidden; j++) norm_out[j] = out[j] * rms_inv * weight[j];
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 accv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= hidden; i += 8) {
        __m256 sv = _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(residual + i));
        _mm256_storeu_ps(out + i, sv);
        accv = _mm256_fmadd_ps(sv, sv, accv);
    }
    __m128 a128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
    a128 = _mm_hadd_ps(a128, a128); a128 = _mm_hadd_ps(a128, a128);
    float sum_sq = _mm_cvtss_f32(a128);
    for (; i < hidden; i++) { out[i] = x[i] + residual[i]; sum_sq += out[i] * out[i]; }
    float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
    __m256 scale = _mm256_set1_ps(rms_inv);
    int j = 0;
    for (; j + 8 <= hidden; j += 8) {
        __m256 v = _mm256_loadu_ps(out + j), w = _mm256_loadu_ps(weight + j);
        _mm256_storeu_ps(norm_out + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
    }
    for (; j < hidden; j++) norm_out[j] = out[j] * rms_inv * weight[j];
#else
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden; i++) { out[i] = x[i] + residual[i]; sum_sq += out[i] * out[i]; }
    float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
    for (int i = 0; i < hidden; i++) norm_out[i] = out[i] * rms_inv * weight[i];
#endif
}

/* ds_vec_add - 向量加 dst[i] = a[i] + b[i]（SIMD 版残差连接）
 * 【模式】主循环每次处理 8/16 个 F32（NEON 两条 4 宽 / AVX-512 一条 16 宽），
 *   尾部不足一个向量宽度的元素标量收尾——全文件 SIMD 内核的标准骨架。
 *   逐元素运算无跨元素依赖，因此 dst 与 a/b 允许重叠（本工程调用方不重叠）。 */
void ds_vec_add(float *dst, const float *a, const float *b, int n) {
#if defined(__ARM_NEON)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; i++) dst[i] = a[i] + b[i];
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) dst[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
#endif
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* ds_silu - SiLU/Swish 激活（原地）：x·σ(x) = x/(1+e^-x)
 * SwiGLU 的门控非线性；零点附近平滑、负半轴软抑制（区别于 ReLU 硬截断）。
 * 【参考】docs/01 §7.3；PyTorch: nn.SiLU()。 */
void ds_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        /* val 很负（<-88）时 expf(-val)=Inf，val/Inf = -0——恰好是正确的极限值 0，
         * 无需特判；val 很大时 e^-val→0，val/1 = val，饱和为恒等。
         * 即 SiLU 对任意输入数值安全（对比直接算 e^x 会在正向溢出成 Inf） */
        x[i] = val / (1.0f + expf(-val));
    }
}

/* ds_gelu - 精确 GELU 激活（原地），SAM ViT / qwen-asr 编码器 FFN 使用
 * 【公式】GELU(x) = x·Φ(x) = 0.5·x·(1 + erf(x/√2))，Φ 是标准正态分布 CDF；
 *   0.7071067811865475f 即 1/√2。用 erf 的"精确版"而非 tanh 近似，
 *   与 PyTorch nn.GELU(approximate='none') 逐位对齐；erf 是 libm 特殊函数，
 *   比 SiLU 的 exp 慢，但编码器只在 prefill 阶段运行、可接受。 */
void ds_gelu(float *x, int n) {
    /* Exact GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
     * Matches PyTorch nn.GELU(approximate='none') */
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float arg = val * 0.7071067811865475f; /* 1/sqrt(2) */
        float erf_val = erff(arg);
        x[i] = 0.5f * val * (1.0f + erf_val);
    }
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} ds_swiglu_task_t;

/* ds_swiglu_worker - 交错布局 SwiGLU 线程体：按 token 切区间（每 token 的
 * 2*inter 输入互不重叠）；alias_inplace=1 表示 out 与 gate_up 是同一缓冲
 * （decode 路径省一次分配），此时必须逐元素就地覆盖、不能走先整行写 -g
 * 再 vvexpf 的两遍式（会破坏尚未读取的输入） */
static void ds_swiglu_worker(int tid, int n_threads, void *arg) {
    ds_swiglu_task_t *t = (ds_swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): use fast exp approximation. */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + ds_fast_expf(-g)); /* SiLU with fast exp */
                o[j] = g * u;
            }
        }
    }
}

/* ds_swiglu_multiply - SwiGLU（交错布局版）：输入 [seq, 2*inter] 按
 * [g0,u0,g1,u1,...] 排列（qwen-asr 风格 gate_up_fused 权重的输出布局），
 * 输出 out[s,i] = SiLU(gu[2i])·gu[2i+1]
 * 【并行条件】seq≥2 且 inter≥256 才进线程池——decode 单 token 时任务太小，
 *   barrier 开销不划算（与 ds_swiglu_direct 同一阈值）。
 * 【新代码建议】分离布局请用 ds_swiglu_direct（顺序读写，cache 更友好）。
 * 【参考】docs/01 §7.3（SwiGLU 数学与 kernel fusion）。 */
void ds_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    ds_swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        ds_parallel_for(ds_swiglu_worker, &task);
    } else {
        ds_swiglu_worker(0, 1, &task);
    }
}

/* ========================================================================
 * Direct SwiGLU: SiLU(gate) * up without interleaving
 * Eliminates the gate_up_buf interleaving pattern which causes
 * non-sequential writes: gate_up[2*i] = gate[i], gate_up[2*i+1] = up[i].
 * Instead, compute SiLU(gate[i]) * up[i] directly from separate buffers.
 * ======================================================================== */

typedef struct {
    float *out;
    const float *gate;
    const float *up;
    int seq_len;
    int intermediate;
} ds_swiglu_direct_task_t;

static void ds_swiglu_direct_worker(int tid, int n_threads, void *arg) {
    /* 每个 worker 处理 [s0, s1) 区间的 token，区间互斥无需锁。
     * SiLU 恒等式: z·sigmoid(z) = z / (1 + e^(-z))——把"乘 sigmoid"化成一次除法；
     * Apple 平台用 vForce 向量化 exp，其余平台用多项式快速 exp。 */
    ds_swiglu_direct_task_t *t = (ds_swiglu_direct_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    for (int s = s0; s < s1; s++) {
        const float *g = t->gate + (size_t)s * inter;
        const float *u = t->up + (size_t)s * inter;
        float *o = t->out + (size_t)s * inter;
#if defined(__APPLE__) && defined(USE_BLAS)
        /* Vectorized exp(-g) using Accelerate vForce — fastest on Apple Silicon. */
        for (int j = 0; j < inter; j++) o[j] = -g[j];
        int n = inter;
        vvexpf(o, o, &n);
        for (int j = 0; j < inter; j++) {
            o[j] = (g[j] / (1.0f + o[j])) * u[j];
        }
#else
        /* Fast exp approximation — avoids expf() overhead on non-Accelerate. */
        for (int j = 0; j < inter; j++) {
            o[j] = (g[j] / (1.0f + ds_fast_expf(-g[j]))) * u[j];
        }
#endif
    }
}

void ds_swiglu_direct(float *out, const float *gate_buf, const float *up_buf,
                       int seq_len, int intermediate) {
    /* 【理论】SwiGLU（Shazeer 2020, GLU 变体）：FFN(x) = SiLU(x·W_g) ⊙ (x·W_u)
     *   SiLU(z) = z·sigmoid(z)（即 silu/swish，平滑版 ReLU）
     * 门控 ⊙ 让网络"选择性地"放大部分激活，实践上比 GELU-FFN 收敛更好。
     * 本工程中间维 6848（dense）/1792（expert）。
     * 【实现】out[i] = silu(gate[i]) * up[i]，单遍顺序读写（cache 友好）；
     * 多 token（prefill）且量够大时拆给线程池。 */
    ds_swiglu_direct_task_t task = {
        .out = out, .gate = gate_buf, .up = up_buf,
        .seq_len = seq_len, .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        ds_parallel_for(ds_swiglu_direct_worker, &task);
    } else {
        ds_swiglu_direct_worker(0, 1, &task);
    }
}

/* ds_softmax - 逐行 softmax（原地）：p_i = e^(x_i-max)/Σ e^(x_j-max)
 * 【为什么减 max】e^x 上溢极快：F32 最大值 ~3.4e38，expf(89) 已经是 Inf。
 *   反例：两个 logits 90 与 89，直接 exp 得 Inf/Inf = NaN——整个分布报废；
 *   减去行最大值 90 后变 e^0=1 与 e^-1≈0.368，得 0.731/0.269，完全正常。
 *   减 max 后指数参数 ≤0，结果落在 (0,1]；分子分母同乘 e^-max 相互约掉，
 *   平移不改变 softmax 分布（数学恒等）。下溢方向无害：极小分数的 exp 项
 *   下溢为 0 只是丢弃可忽略的概率尾巴，不会产生 NaN。
 * 【参数】x[rows, cols] 行主序；MoE 路由 64 列、编码器分类头等小规模场景
 * 使用；注意力的大行 softmax 内联在各 attention 内核里（online softmax）。
 * 【参考】docs/01 §5.2；PyTorch: torch.softmax(dim=-1)。 */
void ds_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];   // 第一遍：行最大值（防 exp 上溢的锚点）
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);   // 就地覆写为分子 e^(x-max)，同时累加分母
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/* ds_dot_f32 - F32 点积包装：转发到编译期选定的平台实现（NEON/AVX/generic，
 * 见 ds_kernels_impl.h 的宏分发），注意力 QK 打分等高频小向量运算用 */
static inline float ds_dot_f32(const float *a, const float *b, int n) {
    return ds_dot_f32_impl(a, b, n);
}

/* dst = dst * scale
 * ds_vec_scale_inplace - 原地缩放；注意力 online softmax 收尾时乘 1/Σexp */
static inline void ds_vec_scale_inplace(float *dst, float scale, int n) {
    ds_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src
 * ds_vec_axpy_inplace - 原地 axpy；online softmax 常规分支累加 wt·V 行 */
static inline void ds_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    ds_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src
 * ds_vec_scale_add - 融合"重缩放旧和 + 并入新 V 行"：online softmax 换最大值
 * 时一次遍历完成 o = o·e^(m_old-m_new) + v_new（省一趟单独的 scale 扫描） */
static inline void ds_vec_scale_add(float *dst, const float *src, float correction, int n) {
    ds_vec_scale_add_impl(dst, src, correction, n);
}

/* ds_bidirectional_attention - 双向注意力（编码器：每个 token 看所有 token）
 * 【算法/数学】head_h、行 i：score_j = (Q_i·K_j)·scale ∈ [0,seq)，
 *   out_i = Σ_j softmax(score)_j · V_j（全序列归一化，无掩码）。
 *   online softmax：单遍扫描同时维护 max/Σ/输出累加器，O(1) 额外内存
 *   （不物化 [seq,seq] 注意力矩阵）。
 * 【参数】Q/K/V: [seq, n_heads*head_dim] 行主序，Q/K/V 同长（无 GQA）；
 *   scale=1/sqrt(head_dim) 由调用方传入。
 * 【参考】docs/01 §9.1；SAM/DeepEncoder 的标准编码路径。
 *
 * 【说人话】双向注意力就是"开卷考试，所有同学互相抄"：每个 token（学生）
 *   都能看到全班的答案（所有 token），然后按"谁跟我更相关"加权抄一份。
 *   "更相关"用点积算——两个向量方向越一致，点积越大，抄得越多。
 *   除以 sqrt(head_dim) 是"防爆表"：head_dim=128 时点积是 128 项相加，
 *   量级会涨到 sqrt(128)≈11 倍，不除的话 softmax 会变成"一家独大"（几乎
 *   全押一个 token），梯度也消失。除以 sqrt(d) 把方差拉回 1，softmax 才有
 *   正常的"温度"。
 * 【边界】双向注意力只能用在编码器（看图/看音频），不能用在生成模型——
 *   生成时后面的 token 还没出生，没法"抄后面的答案"。生成必须用因果掩码
 *   （ds_causal_attention），每个 token 只能看自己和之前的。
 *   另外这里的 online softmax 是单遍扫描：不存 [seq,seq] 的注意力矩阵，
 *   只维护"当前最大值/当前和/当前输出"三个量。seq=4096 时能省 4096²×4B
 *   = 64 MB 内存，代价是数值结果与"先算全部再归一"的两遍版有 ULP 级差异。 */
void ds_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq,
                                   int n_heads, int head_dim, float scale) {
    int hidden = n_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            /* Online softmax（流式）+ 双向注意力：普通 softmax 要先扫一遍求行
             * 最大值、再扫一遍求 exp 和，还要物化 [seq,seq] 分数矩阵；online 版
             * 单遍扫描同时维护三件运行状态——max_score（当前行最大，初值 -1e30
             * 保证首个 score 必走"新最大"分支）、sum_exp（指数和）、o_row
             * （未归一化的加权 V 累加器），O(1) 额外内存，无分数矩阵 */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            /* 双向：j 遍历全部 seq 个 K/V——编码器无因果掩码，token 两两互看 */
            for (int j = 0; j < seq; j++) {
                const float *k_row = K + j * hidden + h * head_dim;   /* 行 = 位置 j×缓存行宽，列偏移 = 本 head 切片 */
                const float *v_row = V + j * hidden + h * head_dim;

                float score = ds_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    /* 分支一（出现新的行最大）：旧权重基于旧 max，须统一换基准——
                     * 以 correction=e^(m_old-m_new) 整体重缩放旧 Σ 与旧 o；
                     * 当前行权重恰为 e^0=1（故 Σ 再 +1）。指数参数恒 ≤0，绝不溢出 */
                    float correction = ds_fast_expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    ds_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    /* 分支二（常见路径）：wt=e^(score-m) ∈ (0,1]，直接累加 */
                    float wt = ds_fast_expf(score - max_score);
                    sum_exp += wt;
                    ds_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                ds_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

/* ds_causal_attention_heads - 因果注意力核心（支持 GQA），head 区间 [head_start, head_end)
 * 【算法/数学】行 i（全局位置 global_pos=q_offset+i）只看 K/V 的 [0, min(global_pos, seq_k))：
 *   k_end=global_pos+1 即因果掩码（未来的 token 不存在/不许看）。
 *   GQA 映射：kv_h = h / heads_per_kv——每 heads_per_kv 个 Q 头共享一个 KV 头
 *   （本工程 V2 编码器 14 Q 头/2 KV 头 → heads_per_kv=7）。
 * 【q_offset】prefill 时 = 已缓存的序列前缀长度；decode 单 token 时 = 当前位置。
 * 【参考】docs/01 §9.2（因果掩码）、§6.3（GQA）。
 *
 * 【说人话】因果注意力是"只能看自己写过的日记"：写第 100 个字时，你只能
 *   翻前 99 个字，不能偷看第 101 个（它还没写）。这就是 k_end = global_pos+1
 *   的含义——循环上界随位置增长，天然形成下三角掩码。
 *   GQA 则是"7 个学生共用一本参考书"：14 个 Q 头（学生）两两分组，每组 7 个
 *   共用同一份 K/V（参考书）。为什么能共用？因为训练后这 7 个 Q 头学到的
 *   "该查什么"高度相似，各存一份 K/V 是浪费。省下来的是 KV 缓存——V2 编码器
 *   从 14×64 降到 2×64，缓存直接缩 7 倍。
 * 【边界】GQA 只省"存 K/V"和"算 K/V 投影"，不省 QK 点积次数（Q 头数没变），
 *   所以它是省内存不是省算力。压到 n_kv_heads=1 就是 MQA，缓存最小但质量
 *   掉得明显（14 个学生抢一本书）；实践中 4-8 个 KV 头是甜点。
 *   另外 kv_h = h / heads_per_kv 是"连续分组"而非"交错分配"——连续分组让
 *   同一 KV 头服务的 7 个 Q 头在内存上相邻，k_row/v_row 地址对它们完全一样，
 *   L1 命中率天然高。交错分配会让 7 个 Q 头跳着访问，缓存行利用率下降。 */
static void ds_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;   /* GQA 映射：q_head ÷ heads_per_kv = kv_head
                                          （本工程 V2 编码器 14 个 Q 头共享 2 个 KV 头，每 7 个 Q 头共用一份 K/V） */

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;   /* 本行 token 在整个序列中的绝对位置 */
            /* 因果掩码（下三角判断）：允许的 K/V 下标 j 须满足 j ≤ global_pos，
             * 即注意力矩阵的下三角（含对角线）。k_end=global_pos+1 把"逐 j 判断
             * 掩码位"压缩成"循环上界"——被掩掉的右上角根本不访问。
             * 超出缓存已有长度 seq_k 时截断（防御） */
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            /* KV cache 读索引：行 = j·kv_hidden（位置 pos × 缓存行宽，即 pos*dim
             * 线性索引），列 = kv_h·head_dim（本 KV 头切片；GQA 下多个 Q 头读同一 KV 头） */
            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = ds_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = ds_fast_expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    ds_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = ds_fast_expf(score - max_score);
                    sum_exp += wt;
                    ds_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                ds_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} ds_causal_attn_task_t;

/* ds_causal_attn_worker - 因果注意力线程体：按 head 切分（每 head 的整行
 * 计算互不依赖，写区间互不相交）；按 head 而非按行切，保证每个 (head,行)
 * 的在线 softmax 序列与串行版一致（逐位可复现） */
static void ds_causal_attn_worker(int tid, int n_threads, void *arg) {
    ds_causal_attn_task_t *t = (ds_causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    ds_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

/* ========================================================================
 * Fused Decode Attention: single-token, standard MHA (no GQA)
 * ======================================================================== */

typedef struct {
    float *out; const float *Q; const float *K; const float *V;
    int n_heads; int head_dim; float scale; int k_end;
} ds_dec_attn_task_t;

/* ds_dec_attn_worker - 融合 decode 注意力线程体（seq_q=1、标准 MHA）：
 * 每 head 独立扫描 k_end 个缓存 K/V 行做在线 softmax；K/V 行距 = 整个
 * hidden（同一 head 的数据跨行等距，点积/累加都命中固定列带） */
static void ds_dec_attn_worker(int tid, int nt, void *arg) {
    ds_dec_attn_task_t *t = (ds_dec_attn_task_t *)arg;
    int chunk = (t->n_heads + nt - 1) / nt;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    int hidden = t->n_heads * t->head_dim;
    for (int h = h0; h < h1; h++) {
        const float *q = t->Q + h * t->head_dim;
        const float *k_base = t->K + h * t->head_dim;
        const float *v_base = t->V + h * t->head_dim;
        float *o = t->out + h * t->head_dim;
        float max_s = -1e30f, sum_e = 0.0f;
        for (int d = 0; d < t->head_dim; d++) o[d] = 0.0f;
        for (int j = 0; j < t->k_end; j++) {
            float score = ds_dot_f32(q, k_base + j * hidden, t->head_dim) * t->scale;
            if (score > max_s) {
                float corr = ds_fast_expf(max_s - score);
                sum_e = sum_e * corr + 1.0f;
                ds_vec_scale_add(o, v_base + j * hidden, corr, t->head_dim);
                max_s = score;
            } else {
                float wt = ds_fast_expf(score - max_s);
                sum_e += wt;
                ds_vec_axpy_inplace(o, v_base + j * hidden, wt, t->head_dim);
            }
        }
        if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, t->head_dim);
    }
}

/* ds_causal_attention - 因果注意力统一入口（decode 快路径 + GQA 通用路径）
 * 【契约】Q: [seq_q, n_heads*head_dim]；K/V: [seq_k, n_kv_heads*head_dim]，
 *   行距 = kv_hidden（非 aligned 版本）；q_offset 为首 query 的全局位置。
 * 【路径选择】
 *   seq_q==1 且 n_heads==n_kv_heads（标准 MHA decode，最常见）→ 融合快路径：
 *     每 head 单遍扫完 QK→在线 softmax→VO，中间结果全在寄存器/累加器，
 *     不写出注意力分数；≥2 线程时按 head 并行。
 *   其余（prefill 批量 / GQA）→ ds_causal_attention_heads（可按 head 并行）。
 * 【参考】docs/03 §6.3（decode 注意力）；docs/01 §5、§6.3。 */
void ds_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    /* Fast path: single-token decode with standard MHA (no GQA).
     * Fused QK+softmax+VO in one pass per head, avoiding intermediate writes.
     * This is the most common decode pattern: seq_q=1, n_kv_heads=n_heads. */
    if (seq_q == 1 && n_heads == n_kv_heads && n_heads >= 2) {
        /* 因果边界：全局位置 q_offset 的 query 只看前 q_offset+1 个 K/V，
         * 超出 seq_k 截断（防御） */
        int k_end = q_offset + 1;
        if (k_end > seq_k) k_end = seq_k;

        if (tp.n_threads > 1) {
            ds_dec_attn_task_t task = {
                .out = out, .Q = Q, .K = K, .V = V,
                .n_heads = n_heads, .head_dim = head_dim,
                .scale = scale, .k_end = k_end
            };
            ds_parallel_for(ds_dec_attn_worker, &task);
            return;
        }

        /* Single-threaded fused decode attention
         * KV cache 读索引：第 j 行基址 = j·hidden（位置 pos × 缓存行宽，
         * 即 pos*dim 的线性索引），列偏移 h·head_dim 取本 head 的切片；
         * q/o 的行宽同为 hidden（seq_q=1，只有一行） */
        int hidden = n_heads * head_dim;
        for (int h = 0; h < n_heads; h++) {
            const float *q = Q + h * head_dim;
            float *o = out + h * head_dim;
            float max_s = -1e30f, sum_e = 0.0f;
            for (int d = 0; d < head_dim; d++) o[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;
                float score = ds_dot_f32(q, k_row, head_dim) * scale;

                if (score > max_s) {
                    float corr = ds_fast_expf(max_s - score);
                    sum_e = sum_e * corr + 1.0f;
                    ds_vec_scale_add(o, v_row, corr, head_dim);
                    max_s = score;
                } else {
                    float wt = ds_fast_expf(score - max_s);
                    sum_e += wt;
                    ds_vec_axpy_inplace(o, v_row, wt, head_dim);
                }
            }
            if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, head_dim);
        }
        return;
    }

    /* General path with GQA support */
    if (tp.n_threads > 1 && n_heads >= 2) {
        ds_causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        ds_parallel_for(ds_causal_attn_worker, &task);
        return;
    }

    ds_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Aligned KV Cache Attention: same logic but with kv_stride for aligned
 * cache access. K/V base pointers point to row 0; each subsequent row
 * is at base + j * kv_stride. This supports the cache-line aligned
 * F32 KV cache layout where kv_stride ≥ kv_dim.
 * ======================================================================== */

/* ds_causal_attention_heads_aligned - 因果注意力（aligned KV 缓存版），head 区间 [head_start, head_end)
 * 与 ds_causal_attention_heads 唯一的区别：K/V 行距用 kv_stride（≥ kv_dim，
 * 由缓存分配方对齐到 cache line），换取跨行访问的可预测地址模式；
 * 因果掩码/GQA 映射/在线 softmax 与非 aligned 版逐行相同 */
static void ds_causal_attention_heads_aligned(float *out, const float *Q,
                                                const float *K, const float *V,
                                                int seq_q, int seq_k,
                                                int n_heads, int n_kv_heads,
                                                int head_dim, float scale,
                                                int q_offset, int kv_stride,
                                                int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_stride + kv_h * head_dim;
                const float *v_row = V + j * kv_stride + kv_h * head_dim;

                /* Software prefetch next K/V rows for sequential access pattern.
                 * Prefetch 2 rows ahead to hide L2 cache latency (~10ns on Apple M2). */
                if (j + 2 < k_end) {
                    __builtin_prefetch(K + (j + 2) * kv_stride + kv_h * head_dim, 0, 1);
                    __builtin_prefetch(V + (j + 2) * kv_stride + kv_h * head_dim, 0, 1);
                }

                float score = ds_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = ds_fast_expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    ds_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = ds_fast_expf(score - max_score);
                    sum_exp += wt;
                    ds_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                ds_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

/* Aligned decode attention task with kv_stride */
typedef struct {
    float *out; const float *Q; const float *K; const float *V;
    int n_heads; int head_dim; float scale; int k_end; int kv_stride;
} ds_dec_attn_aligned_task_t;

/* ds_dec_attn_aligned_worker - aligned decode 注意力线程体（kv_stride 行距 +
 * 2 行软件预取，其余与 ds_dec_attn_worker 相同） */
static void ds_dec_attn_aligned_worker(int tid, int nt, void *arg) {
    ds_dec_attn_aligned_task_t *t = (ds_dec_attn_aligned_task_t *)arg;
    int chunk = (t->n_heads + nt - 1) / nt;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    for (int h = h0; h < h1; h++) {
        const float *q = t->Q + h * t->head_dim;
        const float *k_base = t->K + h * t->head_dim;
        const float *v_base = t->V + h * t->head_dim;
        float *o = t->out + h * t->head_dim;
        float max_s = -1e30f, sum_e = 0.0f;
        for (int d = 0; d < t->head_dim; d++) o[d] = 0.0f;
        for (int j = 0; j < t->k_end; j++) {
            const float *k_row = k_base + j * t->kv_stride;
            const float *v_row = v_base + j * t->kv_stride;
            /* Prefetch 2 rows ahead */
            if (j + 2 < t->k_end) {
                __builtin_prefetch(k_base + (j + 2) * t->kv_stride, 0, 1);
                __builtin_prefetch(v_base + (j + 2) * t->kv_stride, 0, 1);
            }
            float score = ds_dot_f32(q, k_row, t->head_dim) * t->scale;
            if (score > max_s) {
                float corr = ds_fast_expf(max_s - score);
                sum_e = sum_e * corr + 1.0f;
                ds_vec_scale_add(o, v_row, corr, t->head_dim);
                max_s = score;
            } else {
                float wt = ds_fast_expf(score - max_s);
                sum_e += wt;
                ds_vec_axpy_inplace(o, v_row, wt, t->head_dim);
            }
        }
        if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, t->head_dim);
    }
}

typedef struct {
    float *out; const float *Q; const float *K; const float *V;
    int seq_q, seq_k; int n_heads, n_kv_heads;
    int head_dim; float scale; int q_offset; int kv_stride;
} ds_causal_attn_aligned_task_t;

/* ds_causal_attn_aligned_worker - aligned 因果注意力线程体（按 head 切分，
 * 同 ds_causal_attn_worker，多传 kv_stride） */
static void ds_causal_attn_aligned_worker(int tid, int n_threads, void *arg) {
    ds_causal_attn_aligned_task_t *t = (ds_causal_attn_aligned_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    ds_causal_attention_heads_aligned(t->out, t->Q, t->K, t->V,
                                        t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                        t->head_dim, t->scale, t->q_offset, t->kv_stride,
                                        h0, h1);
}

/* ds_causal_attention_aligned - 因果注意力（aligned KV 缓存版统一入口）
 * 【契约】K/V 行距为 kv_stride（而非紧凑 kv_hidden），其余参数语义同
 *   ds_causal_attention；路径选择也一致（MHA decode 快路径 + GQA 通用路径）。
 * 【为什么对齐行距】KV 缓存每行补齐到 cache line 边界后，行起点不再被
 *   前一行"劈开"，预取与替换的粒度更整（见下方 __builtin_prefetch）。
 * 【参考】docs/03 §5（KV 缓存布局）。 */
void ds_causal_attention_aligned(float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int q_offset, int kv_stride) {
    /* Fast path: single-token decode with standard MHA (no GQA). */
    if (seq_q == 1 && n_heads == n_kv_heads && n_heads >= 2) {
        int k_end = q_offset + 1;
        if (k_end > seq_k) k_end = seq_k;

        if (tp.n_threads > 1) {
            ds_dec_attn_aligned_task_t task = {
                .out = out, .Q = Q, .K = K, .V = V,
                .n_heads = n_heads, .head_dim = head_dim,
                .scale = scale, .k_end = k_end, .kv_stride = kv_stride
            };
            ds_parallel_for(ds_dec_attn_aligned_worker, &task);
            return;
        }

        /* Single-threaded fused decode attention with aligned stride */
        for (int h = 0; h < n_heads; h++) {
            const float *q = Q + h * head_dim;
            const float *k_base = K + h * head_dim;
            const float *v_base = V + h * head_dim;
            float *o = out + h * head_dim;
            float max_s = -1e30f, sum_e = 0.0f;
            for (int d = 0; d < head_dim; d++) o[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = k_base + j * kv_stride;
                const float *v_row = v_base + j * kv_stride;
                if (j + 2 < k_end) {
                    __builtin_prefetch(k_base + (j + 2) * kv_stride, 0, 1);
                    __builtin_prefetch(v_base + (j + 2) * kv_stride, 0, 1);
                }
                float score = ds_dot_f32(q, k_row, head_dim) * scale;
                if (score > max_s) {
                    float corr = ds_fast_expf(max_s - score);
                    sum_e = sum_e * corr + 1.0f;
                    ds_vec_scale_add(o, v_row, corr, head_dim);
                    max_s = score;
                } else {
                    float wt = ds_fast_expf(score - max_s);
                    sum_e += wt;
                    ds_vec_axpy_inplace(o, v_row, wt, head_dim);
                }
            }
            if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, head_dim);
        }
        return;
    }

    /* General path with GQA support + aligned stride */
    if (tp.n_threads > 1 && n_heads >= 2) {
        ds_causal_attn_aligned_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset,
            .kv_stride = kv_stride
        };
        ds_parallel_for(ds_causal_attn_aligned_worker, &task);
        return;
    }

    ds_causal_attention_heads_aligned(out, Q, K, V,
                                        seq_q, seq_k, n_heads, n_kv_heads,
                                        head_dim, scale, q_offset, kv_stride,
                                        0, n_heads);
}

/* ========================================================================
 * Reference Sliding Window Attention (R-SWA) for Unlimited-OCR
 * ======================================================================== */

/* ds_dot_bf16xf32 - BF16 存储的 K 行 × F32 Q 向量的点积（BF16 KV 缓存读取路径）
 * 【数值】BF16→F32 用 union 位模式左移 16 位精确展开（无舍入），点积顺序与
 *   F32 存储路径一致——差异只来自"存入时"的 BF16 舍入（0.4% 量级，
 *   r2 有界实验授权，V3 全组合回归绿后启用；V4 打红回退，docs/05 §10.5） */
float ds_dot_bf16xf32(const uint16_t *k, const float *q, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        union { uint32_t u; float f; } e;    // 类型双关：避免未定义的指针强转别名
        e.u = (uint32_t)k[i] << 16;
        s += q[i] * e.f;
    }
    return s;
}

/* ds_axpy_bf16xf32 - o += w·(BF16 V 行展开为 F32)：R-SWA 在线 softmax 累加
 * BF16 V 行用（数学与 F32 路径 ds_vec_axpy_inplace 一致，仅输入精度不同） */
void ds_axpy_bf16xf32(float *o, const uint16_t *v, float w, int n) {
    for (int i = 0; i < n; i++) {
        union { uint32_t u; float f; } e;
        e.u = (uint32_t)v[i] << 16;
        o[i] += w * e.f;
    }
}

/* R-SWA decode: 每 head 独立（在线 softmax 只依赖本 head 的 Q/O），
 * 参考集 [0,prefill_len) + 窗口 [window_start,kv_seq_len) 两段扫描按 head 切分并行。 */
typedef struct {
    float *out; const float *Q; const float *K_base; const float *V_base;
    const uint16_t *K16; const uint16_t *V16;   /* BF16 KV 模式（r2 任务2） */
    int bf16;
    int n_heads, head_dim; float scale; int kv_stride;
    int prefill_len, window_start, kv_seq_len;
} ds_rswa_attn_task_t;

/* ds_rswa_attn_worker - R-SWA 线程体：按 head 切分，每 head 顺序扫
 * 参考段 + 滑窗段（共用同一在线 softmax 状态），bf16 标志切换 K/V 读取路径 */
static void ds_rswa_attn_worker(int tid, int nt, void *arg) {
    ds_rswa_attn_task_t *t = (ds_rswa_attn_task_t *)arg;
    int chunk = (t->n_heads + nt - 1) / nt;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    for (int h = h0; h < h1; h++) {
        const float *q = t->Q + h * t->head_dim;
        float *o = t->out + h * t->head_dim;
        const float *k_h = t->bf16 ? NULL : t->K_base + h * t->head_dim;
        const float *v_h = t->bf16 ? NULL : t->V_base + h * t->head_dim;
        const uint16_t *k16 = t->bf16 ? t->K16 + h * t->head_dim : NULL;
        const uint16_t *v16 = t->bf16 ? t->V16 + h * t->head_dim : NULL;
        float max_s = -1e30f, sum_e = 0.0f;
        for (int d = 0; d < t->head_dim; d++) o[d] = 0.0f;

        /* Range 1: reference [0, prefill_len) + Range 2: window
         * [window_start, kv_seq_len)，两段共用同一在线 softmax 状态 */
        for (int range = 0; range < 2; range++) {
            int j0 = range == 0 ? 0 : t->window_start;
            int j1 = range == 0 ? t->prefill_len : t->kv_seq_len;
            for (int j = j0; j < j1; j++) {
                float score;
                if (t->bf16) {
                    score = ds_dot_bf16xf32(k16 + (size_t)j * t->kv_stride, q, t->head_dim) * t->scale;
                } else {
                    const float *k_row = k_h + (size_t)j * t->kv_stride;
                    score = ds_dot_f32(q, k_row, t->head_dim) * t->scale;
                }
                if (j + 2 < j1) {
                    if (t->bf16) {
                        __builtin_prefetch(k16 + (size_t)(j + 2) * t->kv_stride, 0, 1);
                        __builtin_prefetch(v16 + (size_t)(j + 2) * t->kv_stride, 0, 1);
                    } else {
                        __builtin_prefetch(k_h + (size_t)(j + 2) * t->kv_stride, 0, 1);
                        __builtin_prefetch(v_h + (size_t)(j + 2) * t->kv_stride, 0, 1);
                    }
                }
                if (score > max_s) {
                    float corr = ds_fast_expf(max_s - score);
                    sum_e = sum_e * corr + 1.0f;
                    if (t->bf16) {
                        /* o = o*corr + v（与 F32 路径 ds_vec_scale_add 同一数学） */
                        ds_vec_scale_inplace(o, corr, t->head_dim);
                        ds_axpy_bf16xf32(o, v16 + (size_t)j * t->kv_stride, 1.0f, t->head_dim);
                    } else {
                        const float *v_row = v_h + (size_t)j * t->kv_stride;
                        ds_vec_scale_add(o, v_row, corr, t->head_dim);
                    }
                    max_s = score;
                } else {
                    float wt = ds_fast_expf(score - max_s);
                    sum_e += wt;
                    if (t->bf16) {
                        ds_axpy_bf16xf32(o, v16 + (size_t)j * t->kv_stride, wt, t->head_dim);
                    } else {
                        const float *v_row = v_h + (size_t)j * t->kv_stride;
                        ds_vec_axpy_inplace(o, v_row, wt, t->head_dim);
                    }
                }
            }
        }
        if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, t->head_dim);
    }
}

/* ds_rswa_attention_aligned - R-SWA（Reference Sliding Window）decode 注意力，F32 KV 版
 * 【契约】Q: [n_heads*head_dim]（seq_q=1）；K/V 缓存行距 kv_stride；
 *   注意力只覆盖两个【不连续】KV 区间：参考段 [0, prefill_len)（视觉+prompt，
 *   恒驻）+ 滑窗段 [window_start, kv_seq_len)（环形缓冲的近期文本），
 *   [prefill_len, window_start) 被滑出、跳过——KV 缓存总量由此恒定
 *   （prefill_len + W，不再随生成步数增长，docs/05 §6）。
 * 【算法】两段共用同一组 online softmax 运行状态（max/Σ/累加器）顺序扫描：
 *   跳段不破坏 softmax——指数权重只依赖 score-max，与扫描顺序无关。
 *   warmup 期 window_start==prefill_len 时两段恰好连续，等价普通因果注意力。
 * 【路径】MHA decode 快路径按 head 并行/串行；GQA 通用路径逐 head 映射 kv_h。
 * 【参考】docs/05 §6（R-SWA）；docs/14（V4/DASH 注意力）。 */
void ds_rswa_attention_aligned(float *out, const float *Q, const float *K_base, const float *V_base,
                                int seq_q, int n_heads, int n_kv_heads, int head_dim,
                                float scale, int kv_stride,
                                int prefill_len, int window_start, int kv_seq_len) {
    /* R-SWA decode attention: attend to two non-contiguous KV ranges:
     *   Range 1 (reference): positions [0, prefill_len)
     *   Range 2 (window):    positions [window_start, kv_seq_len)
     * Positions [prefill_len, window_start) are skipped (out of sliding window).
     *
     * This implements the same logic as Python's SlidingWindowLlamaAttention
     * steady-state decode path, where the KV cache has a ring buffer for
     * text tokens beyond the sliding window, but reference (visual) tokens
     * are always kept.
     *
     * For seq_q=1 decode with MHA (n_heads == n_kv_heads), we use an
     * online softmax approach: scan reference range, then window range,
     * maintaining running max/sum for numerically stable softmax. */

    if (seq_q == 1 && n_heads == n_kv_heads) {
        /* Single-token decode fast path */
        if (tp.n_threads > 1 && n_heads >= 2) {
            ds_rswa_attn_task_t task = {
                .out = out, .Q = Q, .K_base = K_base, .V_base = V_base,
                .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
                .kv_stride = kv_stride,
                .prefill_len = prefill_len, .window_start = window_start,
                .kv_seq_len = kv_seq_len
            };
            ds_parallel_for(ds_rswa_attn_worker, &task);
            return;
        }
        for (int h = 0; h < n_heads; h++) {
            const float *q = Q + h * head_dim;
            const float *k_h = K_base + h * head_dim;
            const float *v_h = V_base + h * head_dim;
            float *o = out + h * head_dim;

            float max_s = -1e30f, sum_e = 0.0f;
            for (int d = 0; d < head_dim; d++) o[d] = 0.0f;

            /* Range 1: reference tokens [0, prefill_len) */
            for (int j = 0; j < prefill_len; j++) {
                const float *k_row = k_h + (size_t)j * kv_stride;
                const float *v_row = v_h + (size_t)j * kv_stride;
                if (j + 2 < prefill_len) {
                    __builtin_prefetch(k_h + (size_t)(j + 2) * kv_stride, 0, 1);
                    __builtin_prefetch(v_h + (size_t)(j + 2) * kv_stride, 0, 1);
                }
                float score = ds_dot_f32(q, k_row, head_dim) * scale;
                if (score > max_s) {
                    float corr = ds_fast_expf(max_s - score);
                    sum_e = sum_e * corr + 1.0f;
                    ds_vec_scale_add(o, v_row, corr, head_dim);
                    max_s = score;
                } else {
                    float wt = ds_fast_expf(score - max_s);
                    sum_e += wt;
                    ds_vec_axpy_inplace(o, v_row, wt, head_dim);
                }
            }

            /* Range 2: window tokens [window_start, kv_seq_len) */
            for (int j = window_start; j < kv_seq_len; j++) {
                const float *k_row = k_h + (size_t)j * kv_stride;
                const float *v_row = v_h + (size_t)j * kv_stride;
                if (j + 2 < kv_seq_len) {
                    __builtin_prefetch(k_h + (size_t)(j + 2) * kv_stride, 0, 1);
                    __builtin_prefetch(v_h + (size_t)(j + 2) * kv_stride, 0, 1);
                }
                float score = ds_dot_f32(q, k_row, head_dim) * scale;
                if (score > max_s) {
                    float corr = ds_fast_expf(max_s - score);
                    sum_e = sum_e * corr + 1.0f;
                    ds_vec_scale_add(o, v_row, corr, head_dim);
                    max_s = score;
                } else {
                    float wt = ds_fast_expf(score - max_s);
                    sum_e += wt;
                    ds_vec_axpy_inplace(o, v_row, wt, head_dim);
                }
            }

            if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, head_dim);
        }
        return;
    }


    /* General path: compute attention over combined reference + window positions.
     * For each head, build a compact K/V array from the two ranges,
     * then use standard attention. This is slower but handles GQA correctly. */
    int n_groups = n_heads / n_kv_heads;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_groups;   /* GQA 映射：每 n_groups 个 Q 头共享一个 KV 头 */
        const float *q = Q + h * head_dim;
        const float *k_h = K_base + kv_h * head_dim;
        const float *v_h = V_base + kv_h * head_dim;
        float *o = out + h * head_dim;

        float max_s = -1e30f, sum_e = 0.0f;
        for (int d = 0; d < head_dim; d++) o[d] = 0.0f;

        /* Reference range */
        for (int j = 0; j < prefill_len; j++) {
            const float *k_row = k_h + (size_t)j * kv_stride;
            const float *v_row = v_h + (size_t)j * kv_stride;
            float score = ds_dot_f32(q, k_row, head_dim) * scale;
            if (score > max_s) {
                float corr = ds_fast_expf(max_s - score);
                sum_e = sum_e * corr + 1.0f;
                ds_vec_scale_add(o, v_row, corr, head_dim);
                max_s = score;
            } else {
                float wt = ds_fast_expf(score - max_s);
                sum_e += wt;
                ds_vec_axpy_inplace(o, v_row, wt, head_dim);
            }
        }

        /* Window range */
        for (int j = window_start; j < kv_seq_len; j++) {
            const float *k_row = k_h + (size_t)j * kv_stride;
            const float *v_row = v_h + (size_t)j * kv_stride;
            float score = ds_dot_f32(q, k_row, head_dim) * scale;
            if (score > max_s) {
                float corr = ds_fast_expf(max_s - score);
                sum_e = sum_e * corr + 1.0f;
                ds_vec_scale_add(o, v_row, corr, head_dim);
                max_s = score;
            } else {
                float wt = ds_fast_expf(score - max_s);
                sum_e += wt;
                ds_vec_axpy_inplace(o, v_row, wt, head_dim);
            }
        }

        if (sum_e > 0.0f) ds_vec_scale_inplace(o, 1.0f / sum_e, head_dim);
    }
}

/* ds_rswa_attention_aligned_bf16 - R-SWA 的 BF16 KV 感知入口（r2 任务 2）
 * 【契约】K/V 为 void*：bf16=1 时按 uint16 行解释（ds_dot_bf16xf32 /
 *   ds_axpy_bf16xf32 精确展开），bf16=0 时与 F32 版完全一致；warmup
 *   （window_start==prefill_len）下两种模式都与 F32 全量因果逐位一致。
 * 【防御】BF16 模式仅服务 V3/V4 解码（seq_q=1 且 MHA），其他形状直接
 *   清零输出并报错（不静默算错）。
 * 【性能】KV 读流量减半是动机；V3 实测 decode +8%（docs/05 §10.5）。 */
void ds_rswa_attention_aligned_bf16(float *out, const float *Q, const void *K_base, const void *V_base,
                                    int seq_q, int n_heads, int n_kv_heads, int head_dim,
                                    float scale, int kv_stride,
                                    int prefill_len, int window_start, int kv_seq_len, int bf16) {
    if (seq_q != 1 || n_heads != n_kv_heads) {
        /* BF16 模式仅 V3/V4 解码（seq_q=1, MHA）；防御性断言 */
        fprintf(stderr, "rswa_bf16: unsupported shape seq_q=%d\n", seq_q);
        memset(out, 0, (size_t)seq_q * n_heads * head_dim * sizeof(float));
        return;
    }
    ds_rswa_attn_task_t task = {
        .out = out, .Q = Q,
        .K_base = (const float *)K_base, .V_base = (const float *)V_base,
        .K16 = (const uint16_t *)K_base, .V16 = (const uint16_t *)V_base,
        .bf16 = bf16,
        .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
        .kv_stride = kv_stride,
        .prefill_len = prefill_len, .window_start = window_start,
        .kv_seq_len = kv_seq_len
    };
    if (tp.n_threads > 1 && n_heads >= 2) {
        ds_parallel_for(ds_rswa_attn_worker, &task);
        return;
    }
    ds_rswa_attn_worker(0, 1, &task);
}

/* ds_sinusoidal_pe - 经典 1D 正弦位置编码（Transformer 原版公式，qwen-asr 风格）
 * 【算法/数学】half=d_model/2 个频率按几何级数排布：
 *   inv_timescale_d = 10000^(-d/(half-1))；row[p,d]=sin(p·f_d)、
 *   row[p,half+d]=cos(p·f_d)。与 ds_compute_2d_position_embeddings 同族，
 *   差别在频率分母用 (half-1) 且 sin/cos 分列前后半。
 * 【参数】pe[n_pos, d_model] 输出；n_pos 为位置数。
 * 【注意】本函数未在 ds_kernels.h 声明（当前无调用方），保留作参考实现；
 *   如需使用请先补头文件声明。
 * 【参考】docs/01 §4.2；Vaswani 2017 §3.5。 */
void ds_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

/* ds_compute_rope_neox - 预计算 RoPE（NeoX/split-half 风格）的 cos/sin 缓存
 * 【算法/数学】维度 d∈[0, half) 的频率 f_d = theta^(-2d/head_dim)（theta 本模型
 *   解码器取 10000），位置 pos 的旋转角 angle = pos·f_d：
 *     cos_out[s,d] = cos_out[s,d+half] = cos(pos·f_d)   （前后半重复）
 *     sin_out[s,d] = sin_out[s,d+half] = sin(pos·f_d)
 *   低维 f≈1 旋转快（区分相邻位置），高维 f≈0 旋转慢（区分远距位置）——
 *   多尺度"时钟组"（docs/01 §4.3）。theta 取 10000^(-2i/d) 衰减的原因：
 *   head_dim/2 个频率需在 [1, 1/theta) 内按几何级数铺开——频率太密则各维
 *   冗余、太疏则远距位置在所有维上相位混叠，相对位置无法分辨。
 * 【为什么 split-half 重复】对齐 HF LlamaAttention 的缓存布局
 *   emb = cat(freqs, freqs)，apply 端才能用 rotate_half 公式逐位一致。
 * 【参数】positions[seq] 各 token 的绝对位置（decode 传当前位）；theta 为
 *   RoPE base；缓存可预计算一次复用整个推理过程。
 * 【参考】docs/01 §4.3；Su et al. 2021 RoFormer。 */
void ds_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    /* Split-half RoPE cos/sin cache (matches LlamaAttention used by DeepSeek-OCR-2):
     * Layout: [seq, head_dim] where cos[0:half] == cos[half:end] (repeated).
     * Python LlamaAttention's RoPE:
     *   freqs = outer(t, inv_freq)  # [seq, half]
     *   emb = cat(freqs, freqs, dim=-1)  # [seq, dim]
     *   cos_cached = emb.cos()
     */
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            // f_d = theta^(-2d/head_dim)：指数随维度线性衰减，形成多尺度频率
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Split-half: first half and second half have identical values */
            cos_out[s * head_dim + d]         = c;
            cos_out[s * head_dim + d + half]  = c;
            sin_out[s * head_dim + d]         = sn;
            sin_out[s * head_dim + d + half]  = sn;
        }
    }
}

/* ds_apply_rope_neox - 把 RoPE 旋转施加到 Q 或 K（原地）：每 head 拆成
 * head_dim/2 个 (d, d+half) 配对做 2D 旋转
 * 【算法/数学】split-half 公式（等价于逐对平面旋转）：
 *     out[d]      = x[d]·cos[d] - x[d+half]·sin[d]
 *     out[d+half] = x[d]·sin[d] + x[d+half]·cos[d]
 *   即 LlamaAttention 的 q*cos + rotate_half(q)*sin，rotate_half(x)=
 *   cat(-x[half:], x[:half])；cos/sin 取 split-half 重复布局（见
 *   ds_compute_rope_neox），故两个公式逐位一致。
 * 【关键性质】旋转后位置 m 的 Q 与位置 n 的 K 的点积只依赖 m-n——
 *   注意力天然感知相对位置（docs/01 §4.3）。只旋转 Q/K，V 不动。
 * 【复数视角】把每个 head 内的一对 (x[d], x[d+half]) 看成复数
 *   z = x[d] + i·x[d+half]，本函数的两行输出就是复数乘法 z·e^(iθ)：
 *     实部' = 实部·cosθ − 虚部·sinθ（旋转保模长 |z|，只改相位）
 *     虚部' = 实部·sinθ + 虚部·cosθ
 *   θ = pos·f_d 由位置决定——"相位即位置"。配对方式这里是 split-half
 *   （d 与 d+half 配对，跟随 HF Llama 布局）而非相邻偶奇对 (2d, 2d+1)：
 *   两者只是同一组二维平面旋转的不同元素排列，配合对应布局的 cos/sin
 *   缓存数学等价（详见 ds_compute_rope_neox 的 split-half 说明）。
 * 【参数】x: [seq, n_heads*head_dim] 原地修改；cos_vals/sin_vals: [seq, head_dim]；
 *   R-SWA 的 page 轴角度修正等场景调用方需自备 positions。
 * 【参考】docs/01 §4.3；PyTorch 等价: LlamaAttention.apply_rotary_pos_emb。 */
void ds_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * LLaMA-style split-half RoPE (matches LlamaAttention used by DeepSeek-OCR-2):
     * rotate_half(x) = cat(-x[half:], x[:half])
     * q_embed = q * cos + rotate_half(q) * sin
     *
     * For each element d in [0, half):
     *   out[d]       = x[d]*cos[d] - x[d+half]*sin[d]
     *   out[d+half]  = x[d]*sin[d] + x[d+half]*cos[d]
     *
     * Note: cos[d] == cos[d+half] and sin[d] == sin[d+half] (split-half repeated)
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

            /* Apply split-half rotate: save first half, then update in-place
             * 先取快照 x1/x2 再回写——两行输出互为对方的输入，直接覆盖会污染 */
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[d + half];
                vec[d]         = x1 * c[d]         - x2 * sn[d];
                vec[d + half]  = x1 * sn[d + half] + x2 * c[d + half];
            }
        }
    }
}

/* NEON-vectorized BF16 roundtrip: truncate F32 array to BF16 precision and back.
 * Processes 4 floats at a time using NEON vcvtnq_f32_bf16 (if available) or
 * manual bit manipulation. This is ~4× faster than the scalar loop when
 * DS_BF16_SIMULATE_PYTHON is enabled.
 *
 * ds_bf16_truncate_array - 原地把 F32 数组舍入到 BF16 精度再还原：
 *   data[i] = bf16_round(data[i])（位级 = (u+0x8000)&0xFFFF0000，保留指数、
 *   尾数仅剩高 7 位，数值上模拟"中间量是 BF16"的 Python 行为）。
 * 【为什么需要】Python 里 BF16×BF16→BF16，中间激活每步都被舍到 BF16；
 *   C 引擎全程 F32 累加，若不舍入，SwiGLU/Down 输出会带 Python 没有的
 *   尾数位，长序列后逐步偏离（见 ds_expert_forward_fused 的 7 步流水线）。
 *
 * 【说人话】这是"把话说回原来的精度"。Python 那边每算完一步就把结果四舍五入
 *   到 BF16（只有 7 位尾数，像只保留两位小数的账本）；C 这边全程 F32（23 位
 *   尾数，像记到小数点后七位）。两边算同一件事，C 会多出 Python 没有的尾数位，
 *   一步差一点点，几百步后就能看出偏差——所以要在关键节点主动"舍回 BF16"，
 *   让 C 的数值轨迹和 Python 对齐。
 *   具体做法是位运算：(u + 0x8000) & 0xFFFF0000。加 0x8000 是"四舍五入"的
 *   进位技巧（被丢弃的 16 位中点是 0x8000，加上它，超过一半的自然进位），
 *   掩码则把低 16 位清零——一次加法一次掩码，没有分支、没有浮点运算。
 * 【边界】这是"模拟 Python 行为"的调试/对齐手段，不是数学上更正确的做法。
 *   真正的 BF16 硬件指令（BFCVT）用的是严格 RNE（round-to-nearest-even），
 *   而 +0x8000 在"恰好半途"时恒进位，差半个 ULP——只影响精确半途点，
 *   工程上可接受。另外这个函数只在 DS_BF16_SIMULATE_PYTHON 打开时调用，
 *   正常推理路径不付这个开销。 */
void ds_bf16_truncate_array(float *data, int n) {
#ifdef __ARM_NEON
    /* NEON bit manipulation: round F32 to BF16 precision and back.
     * Works on all ARMv8+ (M1, M2, etc.) without FEAT_BF16.
     * Uses vaddq_u32 + vandq_u32 to match ds_f32_to_bf16_to_f32 scalar logic. */
    int i = 0;
    uint32x4_t mask = vdupq_n_u32(0xFFFF0000u);
    uint32x4_t half = vdupq_n_u32(0x8000u);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t u = vreinterpretq_u32_f32(vld1q_f32(data + i));
        u = vandq_u32(vaddq_u32(u, half), mask);  /* Round to nearest BF16 */
        vst1q_f32(data + i, vreinterpretq_f32_u32(u));
    }
    for (; i < n; i++) {
        uint32_t u;
        memcpy(&u, data + i, sizeof(u));
        u = (u + 0x8000u) & 0xFFFF0000u;
        memcpy(data + i, &u, sizeof(u));
    }
#else
    /* Generic scalar fallback */
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, data + i, sizeof(u));
        u = (u + 0x8000u) & 0xFFFF0000u;
        memcpy(data + i, &u, sizeof(u));
    }
#endif
}
