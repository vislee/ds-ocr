/*
 * ds_metal.h - Metal GPU acceleration for DeepSeek-OCR
 *
 * Provides GPU-accelerated MoE matvec, shared expert forward, and LM head
 * argmax using Apple Metal Compute Shaders. Falls back to CPU when Metal
 * is unavailable (non-Apple platforms).
 *
 * Key design: zero-copy weight sharing via unified memory.
 * BF16 weights loaded via mmap are directly wrapped as Metal buffers
 * (MTLStorageModeShared), so CPU and GPU share the same physical memory
 * with no data transfer overhead.
 */

#ifndef DS_METAL_H
#define DS_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __APPLE__
#define DS_METAL_AVAILABLE 1
#else
#define DS_METAL_AVAILABLE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Metal context — contains device, queue, pipelines, buffers */
typedef struct ds_metal_ctx ds_metal_ctx_t;

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

/* Initialize Metal context. Returns NULL if Metal is unavailable.
 * On success, holds a reference to the default Metal device and
 * a command queue ready for compute work. */
ds_metal_ctx_t *ds_metal_init(void);

/* Release all Metal resources (pipelines, buffers, device). */
void ds_metal_free(ds_metal_ctx_t *ctx);

/* Check if Metal context is valid and ready for compute. */
int ds_metal_is_available(const ds_metal_ctx_t *ctx);

/* ========================================================================
 * Weight Registration (zero-copy)
 * ======================================================================== */

/* Wrap an existing BF16 weight buffer (mmap'd or malloc'd) as a Metal buffer.
 * Uses MTLStorageModeShared on unified memory systems (Apple Silicon),
 * so no copy occurs — GPU reads directly from the same physical pages.
 * Returns 0 on success, -1 on failure. */
int ds_metal_register_bf16(ds_metal_ctx_t *ctx, const uint16_t *data, size_t n);

/* Register a F32 weight buffer as a Metal buffer. */
int ds_metal_register_f32(ds_metal_ctx_t *ctx, const float *data, size_t n);

/* Register the contiguous expert weight block for zero-copy offset access.
 * When expert weights are stored contiguously (expert_block_bf16), a single
 * Metal buffer is created and each expert's weights are accessed via byte offset.
 * This avoids creating 64+ separate Metal buffers for individual expert weights. */
void ds_metal_register_expert_block(ds_metal_ctx_t *ctx, const void *block_ptr,
                                     size_t block_bytes);

/* ========================================================================
 * MoE Expert Batch Forward (decode hot path)
 * ======================================================================== */

/* Execute a batch of MoE expert forwards on GPU.
 *
 * For each expert k in [0..top_k-1]:
 *   1. gate_up_fused matvec: x[hidden] × W_gate_up^T → gate_up[2*inter]
 *   2. Split + SwiGLU: SiLU(gate) * up → hidden_buf[inter]
 *   3. down matvec: hidden_buf[inter] × W_down^T → out_k[hidden]
 * All intermediate results stay on GPU — no CPU↔GPU sync between steps.
 *
 * After all experts:
 *   4. Combine: output[i] = Σ_k weight[k] * expert_out_k[i]
 *
 * Parameters:
 *   ctx         - Metal context
 *   x           - Input vector [hidden] (F32, CPU accessible)
 *   gate_up_ptrs - Array of gate_up_fused_bf16 pointers for each selected expert
 *   down_ptrs   - Array of down_bf16 pointers for each selected expert
 *   top_weights  - Softmax weights for combining expert outputs
 *   top_k       - Number of selected experts (typically 6)
 *   hidden      - Hidden dimension (1280)
 *   inter       - MoE intermediate dimension (896)
 *   output      - Output vector [hidden] (F32, CPU accessible)
 *
 * This is the PRIMARY GPU offload target — MoE MLP is 74-77% of decode time.
 */
void ds_metal_moe_experts(ds_metal_ctx_t *ctx,
                           const float *x,
                           const uint16_t **gate_up_ptrs,
                           const uint16_t **down_ptrs,
                           const float *top_weights,
                           int top_k,
                           int hidden,
                           int inter,
                           float *output);

/* ========================================================================
 * Shared Expert Forward (decode hot path)
 * ======================================================================== */

/* Execute shared expert forward on GPU.
 * Same structure as a single routed expert but with shared_inter = n_shared * inter.
 *
 * Parameters:
 *   ctx         - Metal context
 *   x           - Input [hidden] (F32)
 *   gate_up_bf16 - Fused gate+up weights [2*shared_inter, hidden] (BF16)
 *   down_bf16   - Down projection weights [hidden, shared_inter] (BF16)
 *   hidden      - Hidden dimension (1280)
 *   shared_inter - Shared intermediate dimension (2*896 = 1792)
 *   output      - Output [hidden] (F32), ADDED to existing content (accumulates with routed)
 */
void ds_metal_shared_experts(ds_metal_ctx_t *ctx,
                              const float *x,
                              const uint16_t *gate_up_bf16,
                              const uint16_t *down_bf16,
                              int hidden,
                              int shared_inter,
                              float *output);

/* ========================================================================
 * LM Head Argmax (decode hot path — ~7% of decode time)
 * ======================================================================== */

/* Compute logits = x @ W^T for LM head and return argmax token ID.
 * Instead of materializing all vocab_size logits, tracks only the
 * maximum value and index during computation (streaming argmax).
 *
 * Parameters:
 *   ctx    - Metal context
 *   x      - Input [hidden] (F32)
 *   W_bf16 - Weight [vocab_size, hidden] (BF16)
 *   hidden - Hidden dimension
 *   vocab  - Vocabulary size
 * Returns: token ID with highest logit, or -1 on failure.
 */
int ds_metal_lm_head_argmax(ds_metal_ctx_t *ctx,
                              const float *x,
                              const uint16_t *W_bf16,
                              int hidden,
                              int vocab);

/* ========================================================================
 * Generic BF16 Matvec (for QKV, output projection, etc.)
 * ======================================================================== */

/* Single BF16 matvec on GPU: y = x @ W^T (+ bias if non-NULL)
 * For seq_len=1 (decode), this is equivalent to ds_linear_nobias_bf16.
 *
 * Parameters:
 *   ctx    - Metal context
 *   y      - Output [out_dim] (F32)
 *   x      - Input [hidden] (F32)
 *   W_bf16 - Weight [out_dim, hidden] (BF16)
 *   bias   - Optional bias [out_dim] (F32, may be NULL)
 *   in_dim  - Input dimension
 *   out_dim - Output dimension
 */
void ds_metal_matvec_bf16(ds_metal_ctx_t *ctx,
                            float *y,
                            const float *x,
                            const uint16_t *W_bf16,
                            const float *bias,
                            int in_dim,
                            int out_dim);

/* Fused QKV matvec on GPU: computes q=Wq@x, k=Wk@x, v=Wv@x in one dispatch.
 * Reduces kernel launch overhead vs three separate matvec calls. */
void ds_metal_matvec_bf16_qkv(ds_metal_ctx_t *ctx,
                                float *q, float *k, float *v,
                                const float *x,
                                const uint16_t *Wq_bf16,
                                const uint16_t *Wk_bf16,
                                const uint16_t *Wv_bf16,
                                int in_dim, int q_dim, int kv_dim);

#ifdef __cplusplus
}
#endif

#endif /* DS_METAL_H */
