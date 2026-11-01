/*
 * ds_quantize.h - Per-Row INT8 Weight Quantization for DeepSeek-OCR
 *
 * Per-row asymmetric INT8 quantization for MoE expert weights.
 * Each row gets its own scale and offset, ensuring high accuracy
 * on real weights (RMS < 0.01, vs INT4's 0.13-0.21).
 *
 * Format: per-row asymmetric int8 + scale + offset
 *   For each row r:
 *     min_val = min(W[r, :])
 *     max_val = max(W[r, :])
 *     scale   = (max_val - min_val) / 255.0
 *     idx[j]  = clamp(round((W[r,j] - min_val) / scale), 0, 255)
 *   Dequantize: W[r,j] = scale[r] * (idx + 128) + offset[r]
 *
 * Storage: 1 byte/value + 4 bytes/row (scale) + 4 bytes/row (offset)
 *          = out_dim * (in_dim + 8) bytes
 *          ~2x smaller than BF16 for typical shapes
 *
 * Example: [1792, 1280] → 2295 KB INT8 vs 4480 KB BF16 = 1.95x
 *
 * 【docs/09 §3.5 命名澄清】"Int4" 是历史命名——实际按 255 级量化
 * （INT4 只有 15 级），flag 名 --int4 仅为兼容保留。量化对象与范围：
 * 11 层 MoE 的 64 路由专家 + 2 共享专家的 gate_up_fused/down_weight，
 * Layer 0 稠密 FFN 与所有注意力投影保持 BF16。
 * 【docs/09 §3.4 粒度选择】per-row 而非 per-tensor：同一输出通道的
 * 权重分布相近，一个 scale 够用；元数据 8B/行仅占权重 0.31%。
 */

#ifndef DS_QUANTIZE_H
#define DS_QUANTIZE_H

#include <stdint.h>
#include <stddef.h>

/* INT8 quantized weight block (per-row quantization) */
typedef struct {
    int8_t   *qweight;      /* Quantized indices: [out_dim, in_dim] row-major */
    float    *scale;        /* Per-row scale [out_dim] */
    float    *offset;       /* Per-row min value [out_dim] */
    int       out_dim;      /* Number of output rows */
    int       in_dim;       /* Number of input columns */
    size_t    bytes;        /* Total allocated bytes */
} ds_int4_block_t;         /* Keep name for compatibility with existing code */

/* Quantize a BF16 weight matrix to per-row INT8.
 * W_bf16: [out_dim, in_dim] row-major BF16 weights
 * Returns: 0 on success, -1 on error. */
int ds_int4_quantize_bf16(ds_int4_block_t *block,
                           const uint16_t *W_bf16,
                           int out_dim, int in_dim);

/* Quantize an F32 weight matrix to per-row INT8. */
int ds_int4_quantize_f32(ds_int4_block_t *block,
                          const float *W_f32,
                          int out_dim, int in_dim);

/* Free quantized block resources. */
void ds_int4_block_free(ds_int4_block_t *block);

/* Dequantize INT8 → F32 (full matrix, for verification) */
void ds_int4_dequantize_f32(float *out, const ds_int4_block_t *block);

/* INT8 matvec: y[out_dim] = INT8(W) * x[in_dim]
 * Per-row dequantize-on-the-fly, NEON-optimized for decode (seq_len=1).
 * 【docs/09 §3.3 分解技巧】y[r] = scale·dot(q+128, x) + offset·sum(x)：
 * 两个独立归约一遍扫描完成，反量化零额外内存流量。 */
void ds_int4_matvec(float *y, const float *x,
                     const ds_int4_block_t *block);

/* INT8 matvec for a subset of rows（线程池按行分片时用）. */
void ds_int4_matvec_rows(float *y, const float *x,
                           const ds_int4_block_t *block,
                           int row_start, int nrows);

/* Expert forward with INT8 weights:
 * gate_up_fused + SwiGLU + down — same interface as ds_expert_forward_fused.
 * 量化版专家前向：与 BF16 版逐字节同接口，ds_moe_decoder.c 按
 * int4_enabled 选择走哪条路径。 */
void ds_expert_forward_fused_int4(
    float *out, const float *x,
    const ds_int4_block_t *gate_up_int4,
    const ds_int4_block_t *down_int4,
    int hidden, int intermediate,
    float *gate_up_buf, float *gate_buf,
    float *up_buf, float *hidden_buf);

/* Compute quantization error: RMS error between original and dequantized.
 * 验收线 <0.01（docs/09 §3.5：理想 0.39%，实测 0.43-1.0%）。 */
float ds_int4_quant_error_rms(const ds_int4_block_t *block,
                                const uint16_t *W_bf16);

#endif /* DS_QUANTIZE_H */
