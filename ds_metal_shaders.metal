/*
 * ds_metal_shaders.metal - Metal Compute Shaders for DeepSeek-OCR
 *
 * GPU kernels for MoE decode acceleration.
 * BF16->F32: (uint16)val << 16 -> reinterpret as float32.
 * One threadgroup per output row, SIMD_W=32 threads per TG.
 * SIMD-group reduction via simd_shuffle_down.
 */

#include <metal_stdlib>
using namespace metal;

#define SIMD_W 32

/* ========================================================================
 * bf16_matvec — Core BF16×F32 matrix-vector multiply
 * y[row] = sum_col BF16(W[row,col]) * x[col]
 *
 * Dispatch: 1 threadgroup per row, SIMD_W threads per threadgroup.
 * Supports W offset via setBuffer offset parameter.
 * ======================================================================== */

kernel void bf16_matvec(
    device float* y             [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device const ushort* W      [[buffer(2)]],   /* BF16 weights, may use offset */
    constant int& in_dim        [[buffer(3)]],
    constant int& out_dim       [[buffer(4)]],
    constant int& y_offset      [[buffer(5)]],   /* write offset into y (in floats) */
    uint simd_id  [[thread_index_in_simdgroup]],
    uint simd_gi  [[threadgroup_position_in_grid]]
) {
    int row = (int)simd_gi;
    if (row >= out_dim) return;

    /* W row pointer — W already includes the base offset from setBuffer */
    device const ushort* w_row = W + (size_t)row * in_dim;
    float sum = 0.0f;
    for (int col = (int)simd_id; col < in_dim; col += SIMD_W) {
        float w_f32 = as_type<float>((uint32_t)w_row[col] << 16);
        sum += w_f32 * x[col];
    }
    for (int o = 16; o > 0; o >>= 1)
        sum += simd_shuffle_down(sum, o);

    if (simd_id == 0)
        y[y_offset + row] = sum;
}

/* ========================================================================
 * bf16_matvec_bias — same as bf16_matvec with bias addition
 * ======================================================================== */

kernel void bf16_matvec_bias(
    device float* y             [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device const ushort* W      [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant int& in_dim        [[buffer(4)]],
    constant int& out_dim       [[buffer(5)]],
    constant int& y_offset      [[buffer(6)]],
    uint simd_id  [[thread_index_in_simdgroup]],
    uint simd_gi  [[threadgroup_position_in_grid]]
) {
    int row = (int)simd_gi;
    if (row >= out_dim) return;
    device const ushort* w_row = W + (size_t)row * in_dim;
    float sum = 0.0f;
    for (int col = (int)simd_id; col < in_dim; col += SIMD_W) {
        float w_f32 = as_type<float>((uint32_t)w_row[col] << 16);
        sum += w_f32 * x[col];
    }
    for (int o = 16; o > 0; o >>= 1) sum += simd_shuffle_down(sum, o);
    if (simd_id == 0) y[y_offset + row] = sum + bias[y_offset + row];
}

/* ========================================================================
 * bf16_matvec_qkv — Fused QKV: q=Wq@x, k=Wk@x, v=Wv@x in one dispatch
 * ======================================================================== */

kernel void bf16_matvec_qkv(
    device float* q_out         [[buffer(0)]],
    device float* k_out         [[buffer(1)]],
    device float* v_out         [[buffer(2)]],
    device const float* x       [[buffer(3)]],
    device const ushort* Wq     [[buffer(4)]],
    device const ushort* Wk     [[buffer(5)]],
    device const ushort* Wv     [[buffer(6)]],
    constant int& in_dim        [[buffer(7)]],
    constant int& q_dim         [[buffer(8)]],
    constant int& kv_dim        [[buffer(9)]],
    uint simd_id  [[thread_index_in_simdgroup]],
    uint simd_gi  [[threadgroup_position_in_grid]]
) {
    int total = q_dim + kv_dim * 2;
    int row = (int)simd_gi;
    if (row >= total) return;

    device const ushort* w_row;
    device float* out_buf;
    int out_idx;

    if (row < q_dim) {
        w_row = Wq + (size_t)row * in_dim;
        out_buf = q_out; out_idx = row;
    } else if (row < q_dim + kv_dim) {
        int kr = row - q_dim;
        w_row = Wk + (size_t)kr * in_dim;
        out_buf = k_out; out_idx = kr;
    } else {
        int vr = row - q_dim - kv_dim;
        w_row = Wv + (size_t)vr * in_dim;
        out_buf = v_out; out_idx = vr;
    }

    float sum = 0.0f;
    for (int col = (int)simd_id; col < in_dim; col += SIMD_W) {
        float w_f32 = as_type<float>((uint32_t)w_row[col] << 16);
        sum += w_f32 * x[col];
    }
    for (int o = 16; o > 0; o >>= 1) sum += simd_shuffle_down(sum, o);
    if (simd_id == 0) out_buf[out_idx] = sum;
}

/* ========================================================================
 * swiglu — SiLU(gate) * up, split layout: gate_up[0..n-1]=gate, [n..2n-1]=up
 * ======================================================================== */

kernel void swiglu(
    device float* out           [[buffer(0)]],
    device const float* gate_up [[buffer(1)]],
    constant int& inter         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)inter) return;
    float g = gate_up[gid];
    float u = gate_up[inter + gid];
    out[gid] = g * (1.0f / (1.0f + exp(-g))) * u;
}

/* ========================================================================
 * swiglu_offset — SwiGLU with output and input offsets
 * For batched expert processing in a single command buffer:
 *   out[out_offset + i] = SiLU(gate_up[in_offset + i]) * gate_up[in_offset + inter + i]
 * ======================================================================== */

kernel void swiglu_offset(
    device float* out           [[buffer(0)]],
    device const float* gate_up [[buffer(1)]],
    constant int& inter         [[buffer(2)]],
    constant int& out_offset    [[buffer(3)]],
    constant int& in_offset     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)inter) return;
    float g = gate_up[in_offset + gid];
    float u = gate_up[in_offset + inter + gid];
    out[out_offset + gid] = g * (1.0f / (1.0f + exp(-g))) * u;
}

/* ========================================================================
 * expert_combine — output[i] = sum_k weights[k] * expert_outs[k * hidden + i]
 * ======================================================================== */

kernel void expert_combine(
    device float* output        [[buffer(0)]],
    device const float* expert_outs [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant int& top_k         [[buffer(3)]],
    constant int& hidden        [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)hidden) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++)
        sum += weights[k] * expert_outs[(size_t)k * hidden + gid];
    output[gid] = sum;
}

kernel void add_inplace(
    device float* output        [[buffer(0)]],
    device const float* addend  [[buffer(1)]],
    constant int& n             [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)n) return;
    output[gid] += addend[gid];
}

/* ========================================================================
 * argmax_partial — per-TG chunk argmax for LM head
 * ======================================================================== */

struct argmax_pair { float value; int index; };

kernel void argmax_partial(
    device struct argmax_pair* results [[buffer(0)]],
    device const float* x             [[buffer(1)]],
    device const ushort* W            [[buffer(2)]],
    constant int& in_dim              [[buffer(3)]],
    constant int& vocab               [[buffer(4)]],
    constant int& rows_per_chunk      [[buffer(5)]],
    uint gid    [[thread_position_in_grid]],
    uint simd_id [[thread_index_in_simdgroup]],
    uint simd_n  [[simdgroups_per_threadgroup]],
    uint simd_gi [[threadgroup_position_in_grid]]
) {
    int chunk_start = (int)simd_gi * rows_per_chunk;
    int chunk_end = min(chunk_start + rows_per_chunk, vocab);
    int row = chunk_start + (int)(gid / SIMD_W);
    float row_sum = -1e30f;
    int row_idx = -1;

    if (row < chunk_end) {
        device const ushort* w_row = W + (size_t)row * in_dim;
        row_sum = 0.0f;
        for (int col = (int)simd_id; col < in_dim; col += SIMD_W) {
            float w_f32 = as_type<float>((uint32_t)w_row[col] << 16);
            row_sum += w_f32 * x[col];
        }
        for (int o = 16; o > 0; o >>= 1)
            row_sum += simd_shuffle_down(row_sum, o);
        row_idx = row;
    }

    threadgroup float tg_vals[64];
    threadgroup int tg_idxs[64];
    int sg = (int)(gid / SIMD_W);
    if (simd_id == 0 && sg < 64) {
        tg_vals[sg] = row_sum;
        tg_idxs[sg] = row_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid == 0) {
        float bv = -1e30f; int bi = -1;
        for (int r = 0; r < (int)simd_n && r < 64; r++)
            if (tg_idxs[r] >= 0 && tg_vals[r] > bv) { bv = tg_vals[r]; bi = tg_idxs[r]; }
        results[simd_gi].value = bv;
        results[simd_gi].index = bi;
    }
}

kernel void zero_fill(
    device float* buf [[buffer(0)]],
    constant int& n   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)n) return;
    buf[gid] = 0.0f;
}

kernel void scale_add(
    device float* output  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)n) return;
    output[gid] += scale * input[gid];
}

/* ========================================================================
 * scale_add_offset — scale_add with dst/src offsets
 * output[dst_off + i] += scale * input[src_off + i]
 * For accumulating each expert's output into combined buffer.
 * ======================================================================== */

kernel void scale_add_offset(
    device float* output  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& n       [[buffer(3)]],
    constant int& dst_off [[buffer(4)]],
    constant int& src_off [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)n) return;
    output[dst_off + gid] += scale * input[src_off + gid];
}
