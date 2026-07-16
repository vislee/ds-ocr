/*
 * ds_quantize.c - Per-Row INT8 Weight Quantization for DeepSeek-OCR
 * ds_quantize.c — 逐行 INT8 权重量化实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】MoE 专家权重量化，减少显存/内存占用
 * ─────────────────────────────────────────────────────────────────────
 * MoE 解码器的专家权重占模型总参数的大头（64路由专家×2层=128组）
 * BF16 存储需要 ~2× 内存，INT8 压缩到 ~1×（50%节省），误差 < 0.01 RMS
 *
 * 【量化格式：逐行不对称 INT8】
 *   对每个输出行 r：
 *     min_val = min(W[r, :])
 *     max_val = max(W[r, :])
 *     scale   = (max_val - min_val) / 255.0
 *     idx[j]  = clamp(round((W[r,j] - min_val) / scale), 0, 255)
 *   反量化: W[r,j] = scale[r] * idx[r,j] + offset[r]
 *
 * 【优化技巧：分解计算】
 *   把 y[r] = scale[r] * dot(qrow+128, x) + offset[r] * sum(x)
 *   拆成两个独立的归约，分别用 SIMD 加速
 *
 * 【平台支持】
 *   Apple M2+ (NEON + FEAT_BF16): NEON 路径
 *   Apple M1 (NEON only): NEON 路径（无 dotprod, 手动 widen）
 *   x86 AVX2: generic C 路径
 *
 * 参考文档: 03-推理引擎实战篇.md — 量化与内核优化
 * ═══════════════════════════════════════════════════════════════════════
 */

#include "ds_quantize.h"
#include "ds_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ── Per-row asymmetric INT8 quantize ── */

/**
 * int8_quantize_row() — 对单个输出行做不对称 INT8 量化
 * @qrow:      输出的量化后 int8 值（减 128 偏移，便于 SIMD 加速）
 * @row_scale: 输出的 scale = range / 255
 * @row_offset: 输出的 offset = min_val
 * @values:    输入的原始 BF16→F32 权重行
 * @in_dim:    输入维度（该行的列数）
 *
 * 量化公式: qval = clamp(round((val - min) * 255 / range), 0, 255) - 128
 * 减 128 的原因: int8 范围 [-128,127]，加 128 后变成 [0,255]，适合 NEON vadd
 */
static void int8_quantize_row(int8_t *qrow, float *row_scale, float *row_offset,
                               const float *values, int in_dim) {
    /* Step 1: 扫描该行找到最小值/最大值 */
    float min_val = 1e30f, max_val = -1e30f;
    for (int c = 0; c < in_dim; c++) {
        if (values[c] < min_val) min_val = values[c];
        if (values[c] > max_val) max_val = values[c];
    }
    /* Step 2: 计算 scale 和 offset */
    float range = max_val - min_val;
    float scale = range / 255.0f;
    if (scale < 1e-10f) scale = 1e-10f;  /* 全零行的保护 */
    *row_scale = scale;
    *row_offset = min_val;
    /* Step 3: 逐元素量化（四舍五入、裁剪、减 128 偏移） */
    float inv_scale = 1.0f / scale;
    for (int c = 0; c < in_dim; c++) {
        int v = (int)roundf((values[c] - min_val) * inv_scale);
        if (v < 0) v = 0; if (v > 255) v = 255;
        qrow[c] = (int8_t)(v - 128);
    }
}

/**
 * ds_int4_quantize_bf16() — 将 BF16 权重量化为 INT8（"Int4" 只是历史命名）
 * @block:  输出量化的权重块
 * @W_bf16: 输入的 BF16 权重指针
 * @out_dim: 输出维度（行数）
 * @in_dim:  输入维度（列数）
 *
 * 流程: BF16 → F32（逐元素左移 16 位）→ 逐行量化
 */
int ds_int4_quantize_bf16(ds_int4_block_t *block,
                           const uint16_t *W_bf16,
                           int out_dim, int in_dim) {
    if (!block || !W_bf16 || out_dim <= 0 || in_dim <= 0) return -1;
    memset(block, 0, sizeof(*block));
    block->out_dim = out_dim;
    block->in_dim = in_dim;

    /* 分配存储: qweight + scale + offset */
    block->qweight = (int8_t *)malloc((size_t)out_dim * in_dim);
    block->scale   = (float *)malloc(out_dim * sizeof(float));
    block->offset  = (float *)malloc(out_dim * sizeof(float));
    if (!block->qweight || !block->scale || !block->offset) {
        ds_int4_block_free(block); return -1;
    }

    /* BF16 → F32 批量转换（中间 buffer，量化完成后释放） */
    size_t total = (size_t)out_dim * in_dim;
    float *W_f32 = (float *)malloc(total * sizeof(float));
    if (!W_f32) { ds_int4_block_free(block); return -1; }

    for (size_t i = 0; i < total; i++) {
        uint32_t u = ((uint32_t)W_bf16[i]) << 16;
        memcpy(&W_f32[i], &u, 4);
    }

    /* 逐行量化 */
    for (int r = 0; r < out_dim; r++)
        int8_quantize_row(block->qweight + (size_t)r * in_dim,
                          &block->scale[r], &block->offset[r],
                          W_f32 + (size_t)r * in_dim, in_dim);

    block->bytes = (size_t)out_dim * in_dim + (size_t)out_dim * sizeof(float) * 2;
    free(W_f32);
    return 0;
}

/**
 * ds_int4_quantize_f32() — 直接对 F32 权重做 INT8 量化
 * BF16→F32 转换在调用之前已经完成
 */
int ds_int4_quantize_f32(ds_int4_block_t *block,
                          const float *W_f32,
                          int out_dim, int in_dim) {
    if (!block || !W_f32 || out_dim <= 0 || in_dim <= 0) return -1;
    memset(block, 0, sizeof(*block));
    block->out_dim = out_dim;
    block->in_dim = in_dim;

    block->qweight = (int8_t *)malloc((size_t)out_dim * in_dim);
    block->scale   = (float *)malloc(out_dim * sizeof(float));
    block->offset  = (float *)malloc(out_dim * sizeof(float));
    if (!block->qweight || !block->scale || !block->offset) {
        ds_int4_block_free(block); return -1;
    }

    for (int r = 0; r < out_dim; r++)
        int8_quantize_row(block->qweight + (size_t)r * in_dim,
                          &block->scale[r], &block->offset[r],
                          W_f32 + (size_t)r * in_dim, in_dim);

    block->bytes = (size_t)out_dim * in_dim + (size_t)out_dim * sizeof(float) * 2;
    return 0;
}

/**
 * ds_int4_block_free() — 释放量化权重块内存
 */
void ds_int4_block_free(ds_int4_block_t *block) {
    if (!block) return;
    free(block->qweight); block->qweight = NULL;
    free(block->scale);   block->scale = NULL;
    free(block->offset);  block->offset = NULL;
    block->bytes = 0;
}

/**
 * ds_int4_dequantize_f32() — 将 INT8 权重反量化为 F32（调试/验证用）
 * 反量化公式: val = scale * (qval + 128) + offset
 */
void ds_int4_dequantize_f32(float *out, const ds_int4_block_t *block) {
    if (!out || !block) return;
    for (int r = 0; r < block->out_dim; r++) {
        float scale = block->scale[r];
        float off   = block->offset[r];
        const int8_t *qrow = block->qweight + (size_t)r * block->in_dim;
        float *orow = out + (size_t)r * block->in_dim;
        for (int c = 0; c < block->in_dim; c++)
            orow[c] = scale * ((int)qrow[c] + 128) + off;
    }
}

/**
 * ds_int4_quant_error_rms() — 计算量化误差的 RMS（相对原始值）
 * 用于评估量化精度，通常 < 0.01（对比 INT4 的 0.13-0.21）
 */
float ds_int4_quant_error_rms(const ds_int4_block_t *block,
                                const uint16_t *W_bf16) {
    if (!block || !W_bf16) return -1.0f;
    size_t total = (size_t)block->out_dim * block->in_dim;
    float *deq = (float *)malloc(total * sizeof(float));
    if (!deq) return -1.0f;
    ds_int4_dequantize_f32(deq, block);
    double ssq = 0, sor = 0;
    for (size_t i = 0; i < total; i++) {
        uint32_t u = ((uint32_t)W_bf16[i]) << 16;
        float orig; memcpy(&orig, &u, 4);
        float d = orig - deq[i];
        ssq += d * d;
        sor += orig * orig;
    }
    free(deq);
    return (sor > 0) ? sqrtf((float)(ssq / sor)) : 0.0f;
}

/* ========================================================================
 * INT8 matvec — optimized implementations
 *
 * 分解技巧：避免逐元素反量化的开销
 *   W[r,j] = scale[r] * (qweight[r,j] + 128) + offset[r]
 *   y[r]   = sum_j(W[r,j] * x[j])
 *          = scale[r] * sum_j((qweight[r,j]+128) * x[j]) + offset[r] * sum_j(x[j])
 *          = scale[r] * dot_j(qrow+128, x) + offset[r] * sum_j(x[j])
 *   dot 和 sum 两个归约都可以用 SIMD 并行加速
 * ======================================================================== */

/**
 * int8_matvec_generic() — INT8 matvec 纯 C 参考实现
 * 每个输出行独立计算 dot(sum((qweight+128)*x)) + offset*sum(x)
 * 适用于无 SIMD 的平台（x86 无 AVX2 时）
 */
static void int8_matvec_generic(float *y, const float *x,
                                  const ds_int4_block_t *block,
                                  int row_start, int nrows) {
    const int in_dim = block->in_dim;
    for (int r = row_start; r < row_start + nrows; r++) {
        float scale = block->scale[r];
        float off   = block->offset[r];
        const int8_t *qrow = block->qweight + (size_t)r * in_dim;
        float dot_sum = 0.0f, x_sum = 0.0f;
        for (int c = 0; c < in_dim; c++) {
            dot_sum += (float)((int)qrow[c] + 128) * x[c];
            x_sum += x[c];
        }
        y[r - row_start] = scale * dot_sum + off * x_sum;
    }
}

#ifdef __ARM_NEON

/**
 * int8_matvec_neon() — NEON SIMD 加速 INT8 matvec
 * 每 16 个元素一组处理（4×float32x4 = 16 条 NEON 指令）
 * dot 和 sum 两个归约并行进行，最后横向累加
 *
 * M2+ 上使用 dotprod 指令（SDOT）可进一步加速，
 * 目前使用通用 widen→f32→vmla 组合
 */
static void int8_matvec_neon(float *y, const float *x,
                               const ds_int4_block_t *block,
                               int row_start, int nrows) {
    const int in_dim = block->in_dim;

    for (int r = row_start; r < row_start + nrows; r++) {
        float scale_v = block->scale[r];
        float off_v   = block->offset[r];
        const int8_t *qrow = block->qweight + (size_t)r * in_dim;

        /* 4 路并行 dot 累加器 + 1 路 x_sum 累加器 */
        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0);
        float32x4_t xsum = vdupq_n_f32(0);

        int16x8_t bias = vdupq_n_s16(128);  /* 用于 (qval+128) 的常量 */

        int c = 0;
        /* 主循环：每次处理 16 个 int8，展开为 4 个 float32x4 */
        for (; c + 16 <= in_dim; c += 16) {
            /* 加载 16 个 int8 */
            int8x16_t qv = vld1q_s8(qrow + c);
            /* int8 → int16 (+128 偏移) */
            int16x8_t q0 = vaddq_s16(vmovl_s8(vget_low_s8(qv)), bias);
            int16x8_t q1 = vaddq_s16(vmovl_s8(vget_high_s8(qv)), bias);
            /* int16 → int32 → float32 */
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1)));

            /* 加载对应的 x 值 */
            float32x4_t xv0 = vld1q_f32(x + c);
            float32x4_t xv1 = vld1q_f32(x + c + 4);
            float32x4_t xv2 = vld1q_f32(x + c + 8);
            float32x4_t xv3 = vld1q_f32(x + c + 12);

            /* FMA: float += float * float */
            acc0 = vmlaq_f32(acc0, f0, xv0);
            acc1 = vmlaq_f32(acc1, f1, xv1);
            acc2 = vmlaq_f32(acc2, f2, xv2);
            acc3 = vmlaq_f32(acc3, f3, xv3);

            /* x_sum 累加 */
            xsum = vaddq_f32(xsum, vaddq_f32(vaddq_f32(xv0, xv1),
                                              vaddq_f32(xv2, xv3)));
        }

        /* 尾部元素（< 16 的余数）用纯 C 处理 */
        float dot_tail = 0.0f, x_tail = 0.0f;
        for (; c < in_dim; c++) {
            dot_tail += (float)((int)qrow[c] + 128) * x[c];
            x_tail += x[c];
        }

        /* 横向累加所有 NEON 寄存器的结果 */
        float dot_sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                              vaddq_f32(acc2, acc3))) + dot_tail;
        float x_sum  = vaddvq_f32(xsum) + x_tail;

        y[r - row_start] = scale_v * dot_sum + off_v * x_sum;
    }
}
#endif /* __ARM_NEON */

/* ── Threaded INT8 matvec ── */

/** 线程任务结构体，传递给 int8_matvec_worker */
typedef struct {
    float *y;
    const float *x;
    const ds_int4_block_t *block;
    int row_start;
    int nrows;
} int8_matvec_task_t;

/** int8_matvec_worker: 线程工作函数，按chunk分片计算INT8 matvec */
static void int8_matvec_worker(int tid, int n_threads, void *arg) {
    int8_matvec_task_t *t = (int8_matvec_task_t *)arg;
    int total = t->nrows;
    int chunk = (total + n_threads - 1) / n_threads;
    int my_off = tid * chunk;
    int my_count = chunk;
    if (my_off + my_count > total) my_count = total - my_off;
    if (my_count <= 0) return;
    int my_start = t->row_start + my_off;
    float *my_y = t->y + my_off;

#ifdef __ARM_NEON
    int8_matvec_neon(my_y, t->x, t->block, my_start, my_count);
#else
    int8_matvec_generic(my_y, t->x, t->block, my_start, my_count);
#endif
}

/** ds_int4_matvec: 完整INT8矩阵×向量乘法（自动选择单线程/多线程） */
void ds_int4_matvec(float *y, const float *x, const ds_int4_block_t *block) {
    if (!y || !x || !block || block->out_dim <= 0) return;
    ds_dispatch_ensure_init();

    if (ds_get_threads() <= 1) {
#ifdef __ARM_NEON
        int8_matvec_neon(y, x, block, 0, block->out_dim);
#else
        int8_matvec_generic(y, x, block, 0, block->out_dim);
#endif
        return;
    }
    int8_matvec_task_t task = { y, x, block, 0, block->out_dim };
    ds_parallel_for(int8_matvec_worker, &task);
}

/** ds_int4_matvec_rows: 部分行的INT8矩阵×向量乘法（从row_start开始计算nrows行） */
void ds_int4_matvec_rows(float *y, const float *x,
                           const ds_int4_block_t *block,
                           int row_start, int nrows) {
    if (!y || !x || !block || nrows <= 0) return;
    ds_dispatch_ensure_init();

    if (ds_get_threads() <= 1) {
#ifdef __ARM_NEON
        int8_matvec_neon(y, x, block, row_start, nrows);
#else
        int8_matvec_generic(y, x, block, row_start, nrows);
#endif
        return;
    }
    int8_matvec_task_t task = { y, x, block, row_start, nrows };
    ds_parallel_for(int8_matvec_worker, &task);
}

/* ── Expert forward with INT8 weights ── */

extern int ds_bf16_simulate_python;
void ds_bf16_truncate_array(float *a, int n);
void ds_swiglu_direct(float *out, const float *gate, const float *up,
                        int seq_len, int intermediate);

/** ds_expert_forward_fused_int4: INT8版本的专家前向（gate+up融合，INT8→F32反量化） */
void ds_expert_forward_fused_int4(
    float *out, const float *x,
    const ds_int4_block_t *gate_up_int4,
    const ds_int4_block_t *down_int4,
    int hidden, int intermediate,
    float *gate_up_buf, float *gate_buf,
    float *up_buf, float *hidden_buf) {
    /* Step 1: 融合 gate+up 投影 → INT8反量化 → F32 */
    ds_int4_matvec(gate_up_buf, x, gate_up_int4);
    memcpy(gate_buf, gate_up_buf, intermediate * sizeof(float));
    memcpy(up_buf, gate_up_buf + intermediate, intermediate * sizeof(float));

    /* Step 2: BF16精度截断（匹配Python autocast行为） */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(gate_buf, intermediate);
        ds_bf16_truncate_array(up_buf, intermediate);
    }

    /* Step 3: SwiGLU激活 */
    ds_swiglu_direct(hidden_buf, gate_buf, up_buf, 1, intermediate);
    if (ds_bf16_simulate_python)
        ds_bf16_truncate_array(hidden_buf, intermediate);

    /* Step 4: Down投影 → INT8反量化 → F32 */
    ds_int4_matvec(out, hidden_buf, down_int4);
    if (ds_bf16_simulate_python)
        ds_bf16_truncate_array(out, hidden);
}