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
 * 【量化格式：逐行不对称 INT8】(docs/09 §3.1-3.2)
 *   对每个输出行 r：
 *     min_val = min(W[r, :])
 *     max_val = max(W[r, :])
 *     scale   = (max_val - min_val) / 255.0
 *     idx[j]  = clamp(round((W[r,j] - min_val) / scale), 0, 255)
 *   反量化: W[r,j] = scale[r] * (idx[r,j] + 128) + offset[r]
 *   offset 存 min 而非 zero-point，节省一次减法；scale 预计算倒数，
 *   循环内只做乘法。手算算例见 docs/09 §3.2（最大误差 = scale/2）。
 *
 * 【量化粒度：为什么 per-row】(docs/09 §3.4)
 *   per-tensor 一个 scale 会被 outlier 行拉爆格距；per-group 是真 INT4
 *   的必需品（本工程未做）。同一输出通道的连接强度分布相近，一个
 *   scale 足够。元数据开销可忽略：gate_up_fused [1792,1280] 的
 *   scale+offset 仅 14 KB，占 4.59 MB 权重的 0.31%。
 *
 * 【命名澄清："--int4" 里没有任何 INT4】(docs/09 §3.5)
 *   Int4 是历史命名（flag 兼容性保留）。实际按 255 级量化（INT4 只有
 *   15 级）；覆盖 11 层 MoE 的 64 路由专家 + 2 共享专家的 gate_up_fused
 *   与 down_weight；Layer 0 稠密 FFN、注意力投影、路由 gate、LM head
 *   保持 BF16。验收线 RMS<0.01（ds_quantize.h），实测 0.0043-0.0044
 *   （test_int4 固定种子），真实权重口径 0.007-0.010（README）。
 *   真 INT4 需要 per-group + GPTQ/AWQ 校准，超出教学引擎边界 (§3.6)。
 *
 * 【优化技巧：分解计算】(docs/09 §3.3)
 *   把 y[r] = scale[r] * dot(qrow+128, x) + offset[r] * sum(x)
 *   拆成两个独立的归约（dot 与 x_sum），一遍扫描同时完成——
 *   避免朴素做法"先逐元素反量化再乘加"的两遍内存流量。
 *
 * 【平台支持】
 *   Apple M2+ (NEON + FEAT_BF16): NEON 路径
 *   Apple M1 (NEON only): NEON 路径（无 dotprod, 手动 widen）
 *   x86 AVX2: generic C 路径
 *
 * 参考文档: docs/09-量化与部署篇.md §3；docs/03-推理引擎实战篇.md
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
 *
 * 【说人话】INT8 量化是"把一把精确到毫米的尺子换成只刻 256 格的粗尺子"。
 *   一行权重里最小值和最大值之间是"要量的范围"，把它均匀切成 255 段，
 *   每个数落到最近的一格上——存的是"第几格"（1 字节），不是原值（4 字节），
 *   内存直接省 4 倍。用的时候按"第几格 × 格宽 + 起点"还原，误差最多半格。
 *   为什么每行单独算 scale？因为不同行的数值范围可能差几个数量级（有的行
 *   全是 0.001 量级，有的行到 10），用全局 scale 会让小范围的行全挤在
 *   同一格里，精度全丢。per-row 是"每行配一把自己的尺子"。
 * 【边界】量化误差是**有损且不可逆**的——还原出来的值不等于原值，只是
 *   落在半格误差内。对权重来说通常可接受（权重本身带训练噪声），但对
 *   激活值要小心：激活的动态范围随输入变化，per-row 静态 scale 可能
 *   在极端输入下溢出（所以工业实现常用 per-token 动态量化）。
 *   另外"Int4"只是历史命名，实际是 INT8——见 ds_int4_quantize_bf16 的注释。 */
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
    float inv_scale = 1.0f / scale;      /* 预计算倒数: 循环内除法变乘法 */
    for (int c = 0; c < in_dim; c++) {
        int v = (int)roundf((values[c] - min_val) * inv_scale);  /* 归一到 [0,255] 格点 */
        if (v < 0) v = 0; if (v > 255) v = 255;  /* 裁剪: 防浮点误差越出格点范围 */
        qrow[c] = (int8_t)(v - 128);     /* 平移到 int8 表示域 [-128,127] */
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
                      /* 内存账: qweight 1B/值 + scale/offset 各 4B/行（BF16 版路径） */
    free(W_f32);      /* 中间 F32 副本完成使命，释放（qweight 已是最终形态） */
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
        int8_quantize_row(block->qweight + (size_t)r * in_dim,   /* 第 r 行的量化输出 */
                          &block->scale[r], &block->offset[r],   /* 第 r 行的元数据 */
                          W_f32 + (size_t)r * in_dim, in_dim);   /* 第 r 行的输入 */

    block->bytes = (size_t)out_dim * in_dim + (size_t)out_dim * sizeof(float) * 2;
                      /* 内存账: qweight 1B/值 + scale/offset 各 4B/行（F32 直入路径） */
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
 * @out:     输出缓冲，需 out_dim*in_dim 个 float（行主序，同原矩阵布局）
 * @block:   量化权重块
 *
 * 反量化公式: val = scale * (qval + 128) + offset
 * （加回 128 恢复 [0,255] 无符号格点，乘 scale 还原跨度，加 min 还原平移）
 * 这是 ds_int4_quant_error_rms 的内部依赖，也供单元测试对拍。
 */
void ds_int4_dequantize_f32(float *out, const ds_int4_block_t *block) {
    if (!out || !block) return;
    for (int r = 0; r < block->out_dim; r++) {       /* 逐行：scale/offset 是 per-row 的 */
        float scale = block->scale[r];               /* 取本行的 scale/offset，内层循环不再查表 */
        float off   = block->offset[r];
        const int8_t *qrow = block->qweight + (size_t)r * block->in_dim;  /* 本行量化值 */
        float *orow = out + (size_t)r * block->in_dim;                    /* 本行输出 */
        for (int c = 0; c < block->in_dim; c++)
            orow[c] = scale * ((int)qrow[c] + 128) + off;  /* 逐元素套反量化公式 */
    }
}

/**
 * ds_int4_quant_error_rms() — 计算量化误差的 RMS（相对原始值）
 * 用于评估量化精度，通常 < 0.01（对比 INT4 的 0.13-0.21）
 */
/* 误差度量 = sqrt(Σ(orig-deq)² / Σorig²)，即相对 RMS。
 * docs/09 §2.3 的理论值：INT8 均匀格点 255 级 → 理想 0.39%，
 * 实测固定种子 0.0043-0.0044、真实权重 0.007-0.010，全部落在
 * 设计验收线 0.01 之内。test_int4 用固定种子合成权重钉住这条线。 */
float ds_int4_quant_error_rms(const ds_int4_block_t *block,
                                const uint16_t *W_bf16) {
    if (!block || !W_bf16) return -1.0f;
    size_t total = (size_t)block->out_dim * block->in_dim;
    float *deq = (float *)malloc(total * sizeof(float));   /* 反量化结果的临时缓冲 */
    if (!deq) return -1.0f;
    ds_int4_dequantize_f32(deq, block);                    /* 得到"量化再还原"的权重 */
    double ssq = 0, sor = 0;                               /* double 累加防大矩阵求和漂移 */
    for (size_t i = 0; i < total; i++) {
        uint32_t u = ((uint32_t)W_bf16[i]) << 16;          /* BF16 → F32：尾数补零左移 16 位 */
        float orig; memcpy(&orig, &u, 4);                  /* 位型重解释成 float（无类型双关） */
        float d = orig - deq[i];                           /* 单元素量化误差 */
        ssq += d * d;                                      /* 误差平方和 Σ(orig-deq)² */
        sor += orig * orig;                                /* 原值平方和 Σorig²（归一化基准） */
    }
    free(deq);
    return (sor > 0) ? sqrtf((float)(ssq / sor)) : 0.0f;   /* 相对 RMS；全零矩阵定义为 0 */
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
static void __attribute__((unused)) int8_matvec_generic(float *y, const float *x,
                                  const ds_int4_block_t *block,
                                  int row_start, int nrows) {
    const int in_dim = block->in_dim;
    for (int r = row_start; r < row_start + nrows; r++) {  /* 只算 [row_start, row_start+nrows) 区间 */
        float scale = block->scale[r];                     /* 本行的反量化参数 */
        float off   = block->offset[r];
        const int8_t *qrow = block->qweight + (size_t)r * in_dim;
        float dot_sum = 0.0f, x_sum = 0.0f;                /* 两个独立归约（分解技巧，见上方推导） */
        for (int c = 0; c < in_dim; c++) {
            dot_sum += (float)((int)qrow[c] + 128) * x[c]; /* 归约一: Σ(q+128)·x —— 带偏移点积 */
            x_sum += x[c];                                 /* 归约二: Σx —— 供 offset 项使用 */
        }
        y[r - row_start] = scale * dot_sum + off * x_sum;  /* 合成: y = scale·dot + offset·sum(x) */
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
    /* 行分片: 总行数按线程数切成等大 chunk，线程 tid 取第 tid 块。
     * chunk = ceil(total/n_threads) 保证恰好覆盖全部行；末线程裁掉越界部分 */
    int total = t->nrows;
    int chunk = (total + n_threads - 1) / n_threads;   /* 向上取整的块大小 */
    int my_off = tid * chunk;                          /* 本线程负责的行在任务内的起点 */
    int my_count = chunk;
    if (my_off + my_count > total) my_count = total - my_off;  /* 末块收尾: 裁到剩余行数 */
    if (my_count <= 0) return;                         /* 行数少于线程数: 多余线程直接返回 */
    int my_start = t->row_start + my_off;              /* 换算成权重矩阵里的绝对行号 */
    float *my_y = t->y + my_off;                       /* 输出 y 同样按块偏移，互不重叠 */

#ifdef __ARM_NEON
    int8_matvec_neon(my_y, t->x, t->block, my_start, my_count);   /* NEON 路径 */
#else
    int8_matvec_generic(my_y, t->x, t->block, my_start, my_count);/* 纯 C 路径 */
#endif
}

/** ds_int4_matvec: 完整INT8矩阵×向量乘法（自动选择单线程/多线程）
 * y[out_dim] = W_quant·x[in_dim]；decode 的专家投影都走这里。
 * 行间无依赖、天然可并行；线程池按行分片（见 int8_matvec_worker）。 */
void ds_int4_matvec(float *y, const float *x, const ds_int4_block_t *block) {
    if (!y || !x || !block || block->out_dim <= 0) return;
    ds_dispatch_ensure_init();                     /* 惰性初始化线程池（重复调用无害） */

    if (ds_get_threads() <= 1) {                   /* 单线程: 省掉 fork-join 开销直算 */
#ifdef __ARM_NEON
        int8_matvec_neon(y, x, block, 0, block->out_dim);       /* 全部行一次算完 */
#else
        int8_matvec_generic(y, x, block, 0, block->out_dim);
#endif
        return;
    }
    int8_matvec_task_t task = { y, x, block, 0, block->out_dim };  /* 任务 = 整个矩阵 */
    ds_parallel_for(int8_matvec_worker, &task);    /* fork-join: 每线程领一块行区间 */
}

/** ds_int4_matvec_rows: 部分行的INT8矩阵×向量乘法（从row_start开始计算nrows行）
 * 与 ds_int4_matvec 同一套分片逻辑，只是行区间缩小——供调用方自己做
 * 粗粒度并行（如按专家分层）时复用；y 长度只需 nrows。 */
void ds_int4_matvec_rows(float *y, const float *x,
                           const ds_int4_block_t *block,
                           int row_start, int nrows) {
    if (!y || !x || !block || nrows <= 0) return;
    ds_dispatch_ensure_init();

    if (ds_get_threads() <= 1) {                   /* 单线程直算指定行区间 */
#ifdef __ARM_NEON
        int8_matvec_neon(y, x, block, row_start, nrows);
#else
        int8_matvec_generic(y, x, block, row_start, nrows);
#endif
        return;
    }
    int8_matvec_task_t task = { y, x, block, row_start, nrows };  /* 区间起点透传给分片器 */
    ds_parallel_for(int8_matvec_worker, &task);
}

/* ── Expert forward with INT8 weights ── */

extern int ds_bf16_simulate_python;
void ds_bf16_truncate_array(float *a, int n);
void ds_swiglu_direct(float *out, const float *gate, const float *up,
                        int seq_len, int intermediate);

/** ds_expert_forward_fused_int4: INT8版本的专家前向（gate+up融合，INT8→F32反量化）
 * 数学: out = Down · SwiGLU(GateUp·x)，与 BF16 版 ds_expert_forward_fused
 * 逐算子对应，ds_moe_decoder.c 按 int4_enabled 二选一。
 * @param out           输出 [hidden]（bf16_simulate 开启时会做 BF16 截断）
 * @param x             输入 [hidden]
 * @param gate_up_int4  gate+up 融合权重量化块 [2*inter, hidden]
 * @param down_int4     down 投影权重量化块 [hidden, inter]
 * @param hidden        隐层维度（1280）
 * @param intermediate  专家中间维度（896）
 * @param gate_up_buf/gate_buf/up_buf/hidden_buf
 *                      调用方预分配的复用缓冲（每步零 malloc，形状见函数体）
 * 量化误差只发生在权重读取，激活与累加全程 F32（docs/09 §3.5）。 */
void ds_expert_forward_fused_int4(
    float *out, const float *x,
    const ds_int4_block_t *gate_up_int4,
    const ds_int4_block_t *down_int4,
    int hidden, int intermediate,
    float *gate_up_buf, float *gate_buf,
    float *up_buf, float *hidden_buf) {
    /* Step 1: 融合 gate+up 投影 → INT8反量化 → F32
     * gate_up_buf = [gate | up] 两段拼接（2*inter），两次 memcpy 拆开 */
    ds_int4_matvec(gate_up_buf, x, gate_up_int4);   /* 反量化融在 matvec 里，不落中间矩阵 */
    memcpy(gate_buf, gate_up_buf, intermediate * sizeof(float));              /* 前段 = gate */
    memcpy(up_buf, gate_up_buf + intermediate, intermediate * sizeof(float)); /* 后段 = up */

    /* Step 2: BF16精度截断（匹配Python autocast行为） */
    if (ds_bf16_simulate_python) {
        ds_bf16_truncate_array(gate_buf, intermediate);
        ds_bf16_truncate_array(up_buf, intermediate);
    }

    /* Step 3: SwiGLU激活 */
    ds_swiglu_direct(hidden_buf, gate_buf, up_buf, 1, intermediate);  /* SiLU(gate)*up，逐元素 */
    if (ds_bf16_simulate_python)
        ds_bf16_truncate_array(hidden_buf, intermediate);   /* 激活输出同样截断回 BF16 */

    /* Step 4: Down投影 → INT8反量化 → F32 */
    ds_int4_matvec(out, hidden_buf, down_int4);   /* 第二次量化 matvec: [hidden,inter]·buf */
    if (ds_bf16_simulate_python)
        ds_bf16_truncate_array(out, hidden);      /* 专家最终输出截断，对齐 PyTorch autocast */
}