/*
 * qwen_asr_kernels_avx.c - x86 SIMD hot kernels (AVX2+FMA, with AVX-512 when available)
 *
 * bf16→f32 conversion: load 16 uint16 → zero-extend to 32-bit → shift left 16.
 * Uses AVX-512F+BW for 16-wide bf16 matvec/argmax (dominant for seq_len==1), and
 * AVX2+FMA / AVX-512F for f32 attention helpers (which operate on cache-resident data).
 *
 * The bf16 matvec processes 4 output rows simultaneously to reduce instruction
 * overhead and improve out-of-order execution on memory-bound workloads.
 */

#include "ds_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <string.h>

/* 【生效条件】本文件仅在编译器同时定义 __AVX2__ 与 __FMA__（x86 上由
 * -march/-mcpu 决定）时编入。在本项目的主目标 arm64（Apple Silicon）上
 * 两个宏都未定义，预处理器会把整个文件剔除，编译产物为空翻译单元——
 * 此时 ds_kernels_impl.h 的分发宏也不会引用任何 avx 符号，两边一致。
 * 文件内部再按有无 AVX-512 二分：__AVX512F__ && __AVX512BW__ 走 512 位
 * 实现，否则走 256 位 AVX2 实现（docs/09 §6.4）。
 *
 * 寄存器宽度速查：__m512 = 512 位 = 16 个 F32 lane；__m256i = 256 位 =
 * 32 字节 = 16 个 BF16；_mm256_* / _mm512_* 的 loadu/storeu 均不要求
 * 对齐（对应 NEON 的 vld1q/vst1q），故对权重指针无对齐假设。 */

/* =====================================================================
 * BF16 matvec - uses AVX-512F+BW when available, AVX2 fallback otherwise
 * ===================================================================== */

#if defined(__AVX512F__) && defined(__AVX512BW__)

/* ── AVX-512F+BW 路径（16 lane 宽）──────────────────────────────────────────
 * x86 没有 BF16 硬件乘加：Intel 的 AVX512-BF16 扩展（_mm512_dpbf16_ps，
 * 语义同 ARM 的 bfdot）只存在于少数服务器 CPU（Cooper Lake / Sapphire
 * Rapids 起），通用 x86 不可依赖——故统一走"u16 加宽 + 左移 16 位 +
 * FMADD"的软件转换三连，即 NEON vshll+vfma 路径的 x86 对应物
 * （docs/09 §6.4）。 */

/* ds_bf16_matvec_fused_avx - BF16 matvec 的 AVX-512F+BW 实现（512 位路径）
 * 【算法/数学】同 generic：y[o] = bias[o] + Σ_k W[o][k]·x[k]。
 * 【参数】同 generic（y/x/W_bf16/bias/in_dim/out_dim）。
 * 【为什么】一次处理 4 个输出行：2 条 512 位 x 载入（32 个 F32）被 4 行
 *   复用，权重每行每轮 2 条 256 位整数载入；4 行 × 每行 2 个累加器 =
 *   8 个 512 位累加器，FMADD 依赖链被充分交错。decode 是内存带宽受限
 *   负载（算术强度 ≈1 FLOP/B，docs/09 §1.2），压低指令开销才能把在途
 *   读请求喂满带宽。
 * 【参考】docs/09 §1、§6.4。
 */
void ds_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 4 output rows at a time */
    for (; o + 3 < out_dim; o += 4) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim; // 第 o 行行首
        const uint16_t *w1 = w0 + in_dim;                 // 第 o+1 行（行主序连续）
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;

        // 8 个 512 位 F32 累加器清零：a0/a1 归行0（各管 16 元素），其余行同理
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) { // 主循环：32 元素/次（2 条 512 位 x 载入）
            __m512 xv0 = _mm512_loadu_ps(x + k);      // x[k..k+15]（16 lane）
            __m512 xv1 = _mm512_loadu_ps(x + k + 16); // x[k+16..31]

            /* Row 0 */
            __m256i r0a = _mm256_loadu_si256((const __m256i *)(w0 + k)); // 16 个 BF16（32B）
            __m256i r0b = _mm256_loadu_si256((const __m256i *)(w0 + k + 16));
            // BF16→F32 三连（16 lane 一批），加宽→移位→重解释→FMADD：
            //   _mm512_cvtepu16_epi32：u16 零扩展成 u32（16 路）
            //   _mm512_slli_epi32(…,16)：逐 lane 左移 16 位 → F32 位模式（指数对齐）
            //   _mm512_castsi512_ps：纯位重解释，编译期零指令
            //   _mm512_fmadd_ps(a,b,c) = a*b+c：a=权重、b=x、c=累加器
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a), 16)), xv0, a0);
            a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b), 16)), xv1, a1);

            /* Row 1 */
            // 行 1-3 同构，全部复用 xv0/xv1（4 行共享 x 是"一次 4 行"的主要收益）
            __m256i r1a = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r1b = _mm256_loadu_si256((const __m256i *)(w1 + k + 16));
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a), 16)), xv0, a2);
            a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b), 16)), xv1, a3);

            /* Row 2 */
            __m256i r2a = _mm256_loadu_si256((const __m256i *)(w2 + k));
            __m256i r2b = _mm256_loadu_si256((const __m256i *)(w2 + k + 16));
            a4 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2a), 16)), xv0, a4);
            a5 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2b), 16)), xv1, a5);

            /* Row 3 */
            __m256i r3a = _mm256_loadu_si256((const __m256i *)(w3 + k));
            __m256i r3b = _mm256_loadu_si256((const __m256i *)(w3 + k + 16));
            a6 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3a), 16)), xv0, a6);
            a7 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3b), 16)), xv1, a7);
        }

        for (; k + 16 <= in_dim; k += 16) { // 次级循环：16 元素/次（in_dim%32 余数）
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(w2 + k));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(w3 + k));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0), 16)), xv, a0);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1), 16)), xv, a2);
            a4 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2), 16)), xv, a4);
            a6 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3), 16)), xv, a6);
        }

        // 水平归约：先把同行的两个累加器相加，再 _mm512_reduce_add_ps 做
        // 16→1 的加树归约（编译器按固定顺序展开，结果可复现）；bias 归约后补加
        float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[o]   : 0.0f);
        float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3)) + (bias ? bias[o+1] : 0.0f);
        float s2 = _mm512_reduce_add_ps(_mm512_add_ps(a4, a5)) + (bias ? bias[o+2] : 0.0f);
        float s3 = _mm512_reduce_add_ps(_mm512_add_ps(a6, a7)) + (bias ? bias[o+3] : 0.0f);

        for (; k < in_dim; k++) { // 标量兜底：in_dim%16 的 <16 个尾巴（同 generic 三连）
            float xk = x[k];
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            uint32_t b2=((uint32_t)w2[k])<<16, b3=((uint32_t)w3[k])<<16;
            float v0,v1,v2,v3;
            memcpy(&v0,&b0,4); memcpy(&v1,&b1,4); memcpy(&v2,&b2,4); memcpy(&v3,&b3,4);
            s0+=v0*xk; s1+=v1*xk; s2+=v2*xk; s3+=v3*xk;
        }
        y[o]=s0; y[o+1]=s1; y[o+2]=s2; y[o+3]=s3;
    }

    /* Handle 2-row remainder */
    // out_dim % 4 == 2..3 时剩下的 2 行：结构同上但只有两行（2×2 累加器）
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            __m512 xv0 = _mm512_loadu_ps(x + k);
            __m512 xv1 = _mm512_loadu_ps(x + k + 16);
            __m256i r0a = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r0b = _mm256_loadu_si256((const __m256i *)(w0 + k + 16));
            __m256i r1a = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r1b = _mm256_loadu_si256((const __m256i *)(w1 + k + 16));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a), 16)), xv0, a0);
            a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b), 16)), xv1, a1);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a), 16)), xv0, a2);
            a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b), 16)), xv1, a3);
        }
        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(w1 + k));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0), 16)), xv, a0);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1), 16)), xv, a2);
        }

        float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[o]   : 0.0f);
        float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3)) + (bias ? bias[o+1] : 0.0f);

        for (; k < in_dim; k++) {
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            float v0,v1; memcpy(&v0,&b0,4); memcpy(&v1,&b1,4);
            s0+=v0*x[k]; s1+=v1*x[k];
        }
        y[o]=s0; y[o+1]=s1;
    }

    /* Single remaining row */
    // out_dim 为奇数时的最后一行：单累加器，16 元素/轮 + 标量尾巴
    for (; o < out_dim; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(w + k));
            acc = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16)),
                                  _mm512_loadu_ps(x + k), acc);
        }
        float sum = _mm512_reduce_add_ps(acc) + (bias ? bias[o] : 0.0f);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum += wv*x[k];
        }
        y[o] = sum;
    }
}

/* ds_argmax_bf16_range_avx - 行区间 [start,end) 流式 argmax 的 AVX-512 实现
 * 【算法/数学/参数】同 generic / NEON 版：区间内逐行点积、边算边比，
 *   无 bias；区间由 ds_argmax_worker 按线程切分（左闭右开）。
 * 【为什么】结构与上面的 matvec 同构（4 行 × 32 元素 × 8 累加器），
 *   intrinsic 逐条语义见 matvec 内注释，此处不重复。
 * 【参考】docs/03 §3.4；docs/05 §4.1；docs/09 §6.4。
 */
void ds_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 4 rows at a time */
    for (; o + 3 < end; o += 4) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;

        __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
        __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
        __m512 a4=_mm512_setzero_ps(), a5=_mm512_setzero_ps();
        __m512 a6=_mm512_setzero_ps(), a7=_mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            __m512 xv0 = _mm512_loadu_ps(x + k);
            __m512 xv1 = _mm512_loadu_ps(x + k + 16);
            __m256i r0a=_mm256_loadu_si256((const __m256i*)(w0+k));
            __m256i r0b=_mm256_loadu_si256((const __m256i*)(w0+k+16));
            a0=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a),16)),xv0,a0);
            a1=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b),16)),xv1,a1);
            __m256i r1a=_mm256_loadu_si256((const __m256i*)(w1+k));
            __m256i r1b=_mm256_loadu_si256((const __m256i*)(w1+k+16));
            a2=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a),16)),xv0,a2);
            a3=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b),16)),xv1,a3);
            __m256i r2a=_mm256_loadu_si256((const __m256i*)(w2+k));
            __m256i r2b=_mm256_loadu_si256((const __m256i*)(w2+k+16));
            a4=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2a),16)),xv0,a4);
            a5=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2b),16)),xv1,a5);
            __m256i r3a=_mm256_loadu_si256((const __m256i*)(w3+k));
            __m256i r3b=_mm256_loadu_si256((const __m256i*)(w3+k+16));
            a6=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3a),16)),xv0,a6);
            a7=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3b),16)),xv1,a7);
        }
        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0=_mm256_loadu_si256((const __m256i*)(w0+k));
            __m256i r1=_mm256_loadu_si256((const __m256i*)(w1+k));
            __m256i r2=_mm256_loadu_si256((const __m256i*)(w2+k));
            __m256i r3=_mm256_loadu_si256((const __m256i*)(w3+k));
            a0=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0),16)),xv,a0);
            a2=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1),16)),xv,a2);
            a4=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2),16)),xv,a4);
            a6=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3),16)),xv,a6);
        }

        float s0=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
        float s1=_mm512_reduce_add_ps(_mm512_add_ps(a2,a3));
        float s2=_mm512_reduce_add_ps(_mm512_add_ps(a4,a5));
        float s3=_mm512_reduce_add_ps(_mm512_add_ps(a6,a7));

        for (; k < in_dim; k++) {
            float xk=x[k];
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            uint32_t b2=((uint32_t)w2[k])<<16, b3=((uint32_t)w3[k])<<16;
            float v0,v1,v2,v3;
            memcpy(&v0,&b0,4); memcpy(&v1,&b1,4); memcpy(&v2,&b2,4); memcpy(&v3,&b3,4);
            s0+=v0*xk; s1+=v1*xk; s2+=v2*xk; s3+=v3*xk;
        }

        // 严格大于：并列时保留先出现的行号（与其他平台版本约定一致）
        if (s0>best_val){best_val=s0;best=o;}
        if (s1>best_val){best_val=s1;best=o+1;}
        if (s2>best_val){best_val=s2;best=o+2;}
        if (s3>best_val){best_val=s3;best=o+3;}
    }

    /* Remaining 1-3 rows */
    // 4 行主循环消化不了的 1-3 行：单行 16 元素/轮 + 标量尾巴
    for (; o < end; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(w + k));
            acc = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16)),
                                  _mm512_loadu_ps(x + k), acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum += wv*x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
}

#else /* AVX2 only (or AVX-512F without BW) */

/* ── AVX2-only 路径（256 位寄存器 = 32 字节 = 8 lane）───────────────────────
 * 没有 AVX-512（或只有 AVX-512F 而缺 AVX-512BW）时的回退：全部 _mm256_*
 * 指令。算法与 512 位版相同，宽度和行并行度减半（2 行 × 16 元素/轮）。 */

/* Helper: bf16→f32 for 8 elements */
/* ds_bf16x8_to_f32 - 8 个 BF16 → 8 个 F32
 * 【为什么】_mm256_cvtepu16_epi32 的输入只有 128 位（u16 零扩展 8 路），
 *   加宽后 _mm256_slli_epi32 左移 16 位得 F32 位模式，cast 位重解释——
 *   与 512 位版的三连完全同构，只是宽度 16→8。 */
static inline __m256 ds_bf16x8_to_f32(__m128i raw) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

/* ds_bf16x16_to_f32 - 16 个 BF16 → 两个 __m256（各 8 F32）
 * 【为什么】一次 256 位整数载入（32B = 16 BF16），拆高低两个 128 位半区
 *   （cast 取低、extracti128 取高）分别走 8 宽转换——AVX2 的跨 lane
 *   加宽没有 256 位输入版本，只能按半区处理。 */
static inline void ds_bf16x16_to_f32(const uint16_t *src, __m256 *lo, __m256 *hi) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)src);
    *lo = ds_bf16x8_to_f32(_mm256_castsi256_si128(raw));
    *hi = ds_bf16x8_to_f32(_mm256_extracti128_si256(raw, 1));
}

/* ds_bf16_matvec_fused_avx - BF16 matvec 的 AVX2+FMA 实现（256 位回退路径）
 * 【算法/数学/参数】同 generic；2 行 × 16 元素/轮，4 个 __m256 累加器。
 * 【为什么】相对 512 位版：宽度减半（8 lane）、行并行度减半（2 行）；
 *   尾部归约用 SSE 的 _mm_hadd_ps 两级折叠（AVX2 无跨 128 位 lane 的
 *   横向加法，必须拆半区，见下方注释）。
 */
void ds_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    for (; o + 1 < out_dim; o += 2) { // 2 行 × 16 元素/轮
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 xlo = _mm256_loadu_ps(x + k);    // x[k..k+7]（低 8 lane）
            __m256 xhi = _mm256_loadu_ps(x + k + 8); // x[k+8..15]
            __m256 wlo, whi;
            ds_bf16x16_to_f32(w0 + k, &wlo, &whi);   // 行0 的 16 个 BF16 → 2×8 F32
            a0 = _mm256_fmadd_ps(wlo, xlo, a0);      // 低半区：a0 += wlo*xlo
            a1 = _mm256_fmadd_ps(whi, xhi, a1);      // 高半区：a1 += whi*xhi
            ds_bf16x16_to_f32(w1 + k, &wlo, &whi);   // 行1 复用同一对 x 向量
            a2 = _mm256_fmadd_ps(wlo, xlo, a2);
            a3 = _mm256_fmadd_ps(whi, xhi, a3);
        }
        a0 = _mm256_add_ps(a0, a1); a2 = _mm256_add_ps(a2, a3);
        // 256 位水平归约第一步：AVX2 把 256 位视作 2 个独立 128 位 lane 组，
        // 没有跨组加法——cast 取低 128 位、extractf128 取高 128 位后相加，
        // 每条 lane = 元素 j 与 j+4 的部分和
        __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps(a2, 1));
        // 两级 _mm_hadd_ps 折叠 16 个部分和：第一次 hadd(r0,r1) 得 4 个
        // 四元素部分和（行0/行1 各占两个 lane），第二次 hadd(r0,r0) 收敛为
        // lane0=行0 总和、lane1=行1 总和
        r0 = _mm_hadd_ps(r0, r1); r0 = _mm_hadd_ps(r0, r0);
        float s0 = _mm_cvtss_f32(r0) + (bias ? bias[o] : 0.0f); // lane0 = 行0 和
        // _mm_shuffle_ps(r0,r0,1) 取 lane1 = 行1 和
        float s1 = _mm_cvtss_f32(_mm_shuffle_ps(r0,r0,1)) + (bias ? bias[o+1] : 0.0f);
        for (; k < in_dim; k++) { // 标量兜底：in_dim%16 尾巴
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            float v0,v1; memcpy(&v0,&b0,4); memcpy(&v1,&b1,4);
            s0+=v0*x[k]; s1+=v1*x[k];
        }
        y[o]=s0; y[o+1]=s1;
    }
    for (; o < out_dim; o++) { // 奇数个输出行时的最后一行：单累加器对
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 wlo, whi;
            ds_bf16x16_to_f32(w + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, _mm256_loadu_ps(x+k), a0);
            a1 = _mm256_fmadd_ps(whi, _mm256_loadu_ps(x+k+8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1)); // 拆半区相加
        r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r); // 两级 hadd 收敛：lane0 = 全行和
        float sum = _mm_cvtss_f32(r) + (bias ? bias[o] : 0.0f);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum+=wv*x[k];
        }
        y[o] = sum;
    }
}

/* ds_argmax_bf16_range_avx - 行区间流式 argmax 的 AVX2 实现
 * 【算法/数学/参数】同 generic；单行处理（16 元素/轮），无 bias。
 * 【为什么】AVX2 回退平台上 LM head 的扫描仍占每步流量约 29%
 *   （331 MB / 1.15 GB，docs/09 §1.1），值得 SIMD 化；归约方式与上面
 *   单行版相同（拆半区 + 两级 hadd）。
 */
void ds_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 wlo, whi;
            ds_bf16x16_to_f32(w + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, _mm256_loadu_ps(x+k), a0);
            a1 = _mm256_fmadd_ps(whi, _mm256_loadu_ps(x+k+8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
        float sum = _mm_cvtss_f32(r);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum+=wv*x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }
    *best_out = best;
    *best_val_out = best_val;
}

#endif /* AVX-512F+BW vs AVX2 for bf16 */

/* =====================================================================
 * f32 attention helpers - AVX2+FMA, with AVX-512F when available
 * (operates on L1-resident head vectors)
 * ===================================================================== */

/* ds_dot_f32_avx - F32 点积的 x86 实现（有 AVX-512F 用 16 lane，否则 8 lane）
 * 【算法/数学】Σ a[i]·b[i]。AVX-512：64 元素/轮（4×512 位）→ 16 → 标量；
 *   AVX2：32 元素/轮（4×256 位）→ 8 → 标量。
 * 【参数】a/b[n] 输入；n 元素数；返回点积。
 * 【为什么】4 个独立累加器打破 FMADD 依赖链（Intel FMADD 延迟约 4 拍、
 *   双 FMA 端口需要 ≥2 条独立链才能打满）；数据是 L1 常驻的注意力
 *   head 向量（head_dim=128），纯计算受限，指令形式即收益。
 */
float ds_dot_f32_avx(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) { // 64 元素/轮：4 条独立 FMADD 链各吃 16 元素
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)); // 加树归并
    for (; i + 16 <= n; i += 16) { // 次级循环：n%64 的 16 元素段
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc); // 16→1 加树归约（顺序固定可复现）
    for (; i < n; i++) sum += a[i] * b[i]; // 标量兜底
    return sum;
#else
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) { // 32 元素/轮：4 条独立 FMADD 链各吃 8 元素
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)); // 加树归并
    for (; i + 8 <= n; i += 8) { // 次级循环：n%32 的 8 元素段
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    }
    // 256 位水平归约：拆低/高 128 位半区相加，再两级 hadd 折叠到 lane0
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float sum = _mm_cvtss_f32(r);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

/* ds_vec_scale_inplace_avx - 原地缩放 dst[i] *= scale 的 x86 实现
 * 【参数】同 NEON 版：dst[n] 原地修改；scale 系数；n 元素数。
 * 【为什么】纯流式访存，64/16/1（AVX-512）或 32/8/1（AVX2）三级分块；
 *   此处没有第三个加数，用 _mm_mul_ps 而非 fmadd。 */
void ds_vec_scale_inplace_avx(float *dst, float scale, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 s = _mm512_set1_ps(scale); // 标量广播到 16 lane，循环外只播一次
    for (; i + 64 <= n; i += 64) {    // 64 元素/轮：4 条 512 位流水
        _mm512_storeu_ps(dst + i,      _mm512_mul_ps(_mm512_loadu_ps(dst + i),      s));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), s));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), s));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), s));
    }
    for (; i + 16 <= n; i += 16) {    // 次级：n%64 的 16 元素段
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#else
    int i = 0;
    __m256 s = _mm256_set1_ps(scale);
    for (; i + 32 <= n; i += 32) {    // AVX2 版：32 元素/轮
        _mm256_storeu_ps(dst+i,    _mm256_mul_ps(_mm256_loadu_ps(dst+i),    s));
        _mm256_storeu_ps(dst+i+8,  _mm256_mul_ps(_mm256_loadu_ps(dst+i+8),  s));
        _mm256_storeu_ps(dst+i+16, _mm256_mul_ps(_mm256_loadu_ps(dst+i+16), s));
        _mm256_storeu_ps(dst+i+24, _mm256_mul_ps(_mm256_loadu_ps(dst+i+24), s));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_mul_ps(_mm256_loadu_ps(dst+i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#endif
}

/* ds_vec_axpy_inplace_avx - 原地 axpy dst[i] += alpha·src[i] 的 x86 实现
 * 【参数】同 NEON 版：dst 原地修改；src 输入；alpha 系数；n 元素数。
 * 【为什么】fmadd(loadu(src), a, loadu(dst)) = src*alpha + dst，
 *   一条 FMADD 融合"乘 + 加"，读写与计算在一个微操作内完成。 */
void ds_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 a = _mm512_set1_ps(alpha);
    for (; i + 64 <= n; i += 64) { // 64 元素/轮
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), a, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), a, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), a, _mm512_loadu_ps(dst + i + 48)));
    }
    for (; i + 16 <= n; i += 16) { // 次级
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#else
    int i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    for (; i + 32 <= n; i += 32) { // AVX2 版：32 元素/轮
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(src+i),    a, _mm256_loadu_ps(dst+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(src+i+8),  a, _mm256_loadu_ps(dst+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+16), a, _mm256_loadu_ps(dst+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+24), a, _mm256_loadu_ps(dst+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(src+i), a, _mm256_loadu_ps(dst+i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#endif
}

/* ds_vec_scale_add_avx - 原地融合 dst[i] = dst[i]·correction + src[i] 的 x86 实现
 * 【参数】同 NEON 版；correction = 在线 softmax 的旧和重缩放因子
 *   exp(m_old - m_new)。
 * 【为什么】fmadd(loadu(dst), c, loadu(src)) = dst*correction + src，
 *   "缩放旧和 + 加新和"一条 FMADD 完成；分块同上。 */
void ds_vec_scale_add_avx(float *dst, const float *src, float correction, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 c = _mm512_set1_ps(correction);
    for (; i + 64 <= n; i += 64) { // 64 元素/轮：dst 缩放后加 src
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 16), c, _mm512_loadu_ps(src + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 32), c, _mm512_loadu_ps(src + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 48), c, _mm512_loadu_ps(src + i + 48)));
    }
    for (; i + 16 <= n; i += 16) { // 次级
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#else
    int i = 0;
    __m256 c = _mm256_set1_ps(correction);
    for (; i + 32 <= n; i += 32) { // AVX2 版：32 元素/轮
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(dst+i),    c, _mm256_loadu_ps(src+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+8),  c, _mm256_loadu_ps(src+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+16), c, _mm256_loadu_ps(src+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+24), c, _mm256_loadu_ps(src+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i), c, _mm256_loadu_ps(src+i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#endif
}

#endif /* __AVX2__ && __FMA__ */
