/*
 * ds_moe_decoder.c - DeepSeek-V2 MoE Decoder for DeepSeek-OCR
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
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include "ds_metal.h"
#include "ds_quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

/* Forward declaration */
static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_metal_ctx_t *metal_ctx,
                         float *expert_gate_buf, float *expert_up_buf,
                         float *expert_gate_up_buf, float *expert_hidden_buf,
                         float *shared_gate_buf, float *shared_up_buf,
                         float *shared_gate_up_buf, float *shared_swiglu_buf,
                         float *shared_out_buf,
                         float *expert_outputs_buf);

/* ========================================================================
 * BF16 ↔ F32 Conversion Helpers
 * ======================================================================== */

/* ========================================================================
 * KV Cache Access Helpers — F32 direct storage (no BF16 conversion)
 * ======================================================================== */

/* Get pointer to K cache row for a given layer and position.
 * Cache layout: [layers][max_seq][kv_row_stride], kv_row_stride >= kv_dim. */
static inline float *ds_kv_k_row(ds_ctx_t *ctx, int layer, int pos) {
    int stride = ctx->_kv_row_stride;
    return ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * stride + pos * stride;
}

static inline float *ds_kv_v_row(ds_ctx_t *ctx, int layer, int pos) {
    int stride = ctx->_kv_row_stride;
    return ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * stride + pos * stride;
}

/* Get pointer to start of K/V cache for a layer (row 0) */
static inline float *ds_kv_k_layer(ds_ctx_t *ctx, int layer) {
    int stride = ctx->_kv_row_stride;
    return ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * stride;
}

static inline float *ds_kv_v_layer(ds_ctx_t *ctx, int layer) {
    int stride = ctx->_kv_row_stride;
    return ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * stride;
}

/* Store F32 K/V directly into the aligned F32 KV cache */
static inline void ds_kv_store_f32(float *dst, const float *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* ========================================================================
 * Dense FFN Forward Pass (SwiGLU, used for layer 0)
 * ======================================================================== */

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

    /* SwiGLU: SiLU(gate) * up — direct computation without interleaving.
     * Previously: gate_up[2*i]=gate[i], gate_up[2*i+1]=up[i], then ds_swiglu_multiply.
     * That interleaving caused non-sequential writes (cache-unfriendly).
     * ds_swiglu_direct computes SiLU(gate[i]) * up[i] directly. */
    ds_swiglu_direct(swiglu_buf, gate_buf, up_buf, 1, intermediate);

    /* Down projection */
    ds_linear_nobias_bf16(output, swiglu_buf, layer->dense_down_weight_bf16, 1, intermediate, hidden);

    if (own_gate) free(gate_buf);
    if (own_up) free(up_buf);
    if (own_swiglu) free(swiglu_buf);
}

static void mlp_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_metal_ctx_t *metal_ctx, int layer_idx,
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
        moe_forward(output, x, layer, cfg, metal_ctx,
                    expert_gate_buf, expert_up_buf,
                    expert_gate_up_buf, expert_hidden_buf,
                    shared_gate_buf, shared_up_buf,
                    shared_gate_up_buf, shared_swiglu_buf,
                    shared_out_buf,
                    expert_outputs_buf);
    }
}

static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, ds_metal_ctx_t *metal_ctx,
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

    /* Step 1: Router gate scores */
    float scores_buf[DS_MAX_EXPERTS];
    float *scores = (n_experts <= DS_MAX_EXPERTS) ? scores_buf :
                    (float *)malloc(n_experts * sizeof(float));
    /* Use BF16 gate weight to match Python's BF16 precision.
     * F32 gate weight can select different experts than Python BF16,
     * causing accuracy degradation especially for long sequences. */
    ds_moe_router_bf16(scores, x, layer->gate_weight_bf16, hidden, n_experts);

    /* Step 2: Softmax over all experts first, then select top-K (matching Python).
     * Previously ds_moe_top_k did softmax over only top-K, which was wrong —
     * the softmax values should be computed over all 64 experts before selecting. */
    {
        float max_s = -1e30f;
        for (int e = 0; e < n_experts; e++)
            if (scores[e] > max_s) max_s = scores[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < n_experts; e++) {
            scores[e] = ds_fast_expf(scores[e] - max_s);
            sum_exp += scores[e];
        }
        float inv = 1.0f / sum_exp;
        for (int e = 0; e < n_experts; e++)
            scores[e] *= inv;
    }
    int top_indices[DS_MAX_EXPERTS];
    float top_weights[DS_MAX_EXPERTS];
    for (int k = 0; k < top_k; k++) {
        int best = -1; float best_score = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            int already = 0;
            for (int j = 0; j < k; j++)
                if (top_indices[j] == e) { already = 1; break; }
            if (already) continue;
            if (scores[e] > best_score) { best_score = scores[e]; best = e; }
        }
        top_indices[k] = best;
        top_weights[k] = best_score;
    }
    if (scores != scores_buf) free(scores);

    /* Step 3: Forward through each selected routed expert.
     * Metal GPU path: batch all experts on GPU for massive parallelism.
     * CPU path: sequential per-expert matvec with prefetch optimization. */
    float *expert_outputs = expert_outputs_buf ? expert_outputs_buf :
                            (float *)calloc(top_k * hidden, sizeof(float));

    if (metal_ctx && ds_metal_is_available(metal_ctx) && getenv("DS_METAL_MOE")) {
        /* ── Metal GPU path: MoE expert forward ── */
        const uint16_t *gate_up_ptrs[DS_MAX_EXPERTS];
        const uint16_t *down_ptrs[DS_MAX_EXPERTS];
        for (int k = 0; k < top_k; k++) {
            gate_up_ptrs[k] = layer->experts[top_indices[k]].gate_up_fused_bf16;
            down_ptrs[k] = layer->experts[top_indices[k]].down_weight_bf16;
        }
        /* GPU computes combined expert output directly into 'output' */
        ds_metal_moe_experts(metal_ctx, x, gate_up_ptrs, down_ptrs,
                              top_weights, top_k, hidden, moe_inter, output);
        /* Skip CPU combine since GPU already combines */
        if (!expert_outputs_buf) free(expert_outputs);
        goto shared_experts;
    }

    /* ── CPU path ── */
    /* Prefetch the entire range of selected experts' gate_up_fused weights
     * experts. We also prefetch the down weight (still in mmap'd region). */
    /* Prefetch the entire range of selected experts' gate_up_fused weights
     * within the contiguous block. This covers all top-k experts' gate_up
     * in one system call, letting the kernel async-read-ahead. */
    if (layer->expert_block_bf16 && top_k > 1) {
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
        int expert_id = top_indices[k];
        /* Prefetch next expert's down weight (in mmap'd region, not in block) */
        if (k + 1 < top_k) {
            int next_id = top_indices[k + 1];
            size_t down_bytes = (size_t)hidden * moe_inter * 2;
            madvise((void *)layer->experts[next_id].down_weight_bf16, down_bytes, MADV_WILLNEED);
        }
        /* Use INT4 path if quantized weights available, otherwise BF16 */
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

    /* Step 4: Combine routed expert outputs (CPU path) */
    ds_expert_combine(output, expert_outputs, top_indices, top_weights, top_k, hidden);
    if (!expert_outputs_buf) free(expert_outputs);
    else memset(expert_outputs_buf, 0, top_k * hidden * sizeof(float));

shared_experts:
    /* Step 5: Add shared expert outputs (always active) */
    if (layer->int4_enabled && layer->shared_gate_up_int4.qweight) {
        /* ── INT4 path for shared experts ── */
        int shared_inter = n_shared * moe_inter;
        ds_expert_forward_fused_int4(shared_out_buf, x,
                                      &layer->shared_gate_up_int4,
                                      &layer->shared_down_int4,
                                      hidden, shared_inter,
                                      shared_gate_up_buf, shared_gate_buf,
                                      shared_up_buf, shared_swiglu_buf);
        ds_vec_add(output, output, shared_out_buf, hidden);
    } else if (metal_ctx && ds_metal_is_available(metal_ctx) && getenv("DS_METAL_MOE") &&
        layer->shared_gate_up_fused_bf16 && layer->shared_down_weight_bf16) {
        /* ── Metal GPU path: shared experts ── */
        int shared_inter = n_shared * moe_inter;
        ds_metal_shared_experts(metal_ctx, x,
                                 layer->shared_gate_up_fused_bf16,
                                 layer->shared_down_weight_bf16,
                                 hidden, shared_inter, output);
    } else if (layer->shared_gate_up_fused_bf16) {
        /* Fused gate+up path: single matvec [hidden → 2*shared_inter] */
        int shared_inter = n_shared * moe_inter;
        ds_linear_nobias_bf16(shared_gate_up_buf, x, layer->shared_gate_up_fused_bf16,
                               1, hidden, 2 * shared_inter);
        /* Split into gate and up */
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
        /* Fallback: separate gate + up (for prefill or if fused not available) */
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
    float scale = 1.0f / sqrtf((float)head_dim);

    double t_layer_start = 0, t0 = 0;
    if (ctx->profile_enabled) t_layer_start = ds_time_sec();

    /* Input RMSNorm */
    float *x_norm = ctx->dec_x_norm;
    ds_rms_norm(x_norm, x, layer->input_norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* QKV projections (fused single-token matvec) */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;

    ds_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                layer->wq_weight_bf16,
                                layer->wk_weight_bf16,
                                layer->wv_weight_bf16,
                                hidden, q_dim, kv_dim);

    /* Per-head Q/K RMSNorm (DeepSeek-V2 V1 style; V2/OCR-2 does not use these) */
    if (layer->q_norm_weight)
        ds_rms_norm_per_head(q, layer->q_norm_weight, 1, n_heads, head_dim, cfg->dec_rms_norm_eps);
    if (layer->k_norm_weight)
        ds_rms_norm_per_head(k, layer->k_norm_weight, 1, n_kv_heads, head_dim, cfg->dec_rms_norm_eps);

    /* RoPE (applied to both Q and K) */
    float *cos_vals = ctx->rope_cache_cos + pos * head_dim;
    float *sin_vals = ctx->rope_cache_sin + pos * head_dim;
    ds_apply_rope_neox(q, cos_vals, sin_vals, 1, n_heads, head_dim);
    ds_apply_rope_neox(k, cos_vals, sin_vals, 1, n_kv_heads, head_dim);

    if (ctx->profile_enabled) ctx->perf_layer_qkv_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;

    /* Store K, V directly into F32 KV cache (no BF16 conversion needed).
     * This eliminates the per-step batch BF16→F32 conversion that previously
     * dominated attention time (O(seq_len) conversion per decode step). */
    int cache_offset = ctx->kv_cache_len;
    float *cache_k_row = ds_kv_k_row(ctx, layer_idx, cache_offset);
    float *cache_v_row = ds_kv_v_row(ctx, layer_idx, cache_offset);
    ds_kv_store_f32(cache_k_row, k, kv_dim);
    ds_kv_store_f32(cache_v_row, v, kv_dim);

    /* Causal attention — read directly from F32 KV cache (zero conversion).
     * The cache is aligned and contiguous for optimal sequential read.
     * K/V rows are at stride _kv_row_stride (≥ kv_dim, aligned to 64 bytes).
     *
     * R-SWA (Reference Sliding Window Attention, Unlimited-OCR):
     * During decode (cache_offset >= prefill_token_count), only attend to:
     *   1. Reference tokens: positions [0..prefill_token_count-1] (visual + prompt tokens)
     *   2. Recent text tokens: positions [window_start..kv_cache_len]
     * where window_start = max(prefill_token_count, kv_seq_len - sliding_window_size)
     * During prefill (cache_offset < prefill_token_count), use full causal attention. */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *attn_out = ctx->dec_attn_out;
    float *k_base = ds_kv_k_layer(ctx, layer_idx);
    float *v_base = ds_kv_v_layer(ctx, layer_idx);
    int kv_seq_len = cache_offset + 1;
    int kv_stride = ctx->_kv_row_stride;

    if (cfg->sliding_window_size > 0 && cache_offset >= ctx->prefill_token_count) {
        /* R-SWA decode path: attend only to reference + sliding window */
        int prefill_len = ctx->prefill_token_count;
        int W = cfg->sliding_window_size;
        int window_start = prefill_len > (kv_seq_len - W) ? prefill_len : (kv_seq_len - W);

        /* Two non-contiguous ranges: [0..prefill_len-1] and [window_start..kv_seq_len-1]
         * For efficiency, we compute attention manually with these two ranges.
         * If window_start <= prefill_len, the ranges overlap and we just use
         * [0..kv_seq_len-1] (which happens during warmup when kv_seq_len < prefill_len + W). */
        if (window_start <= prefill_len) {
            /* Warmup phase: no gap between reference and window, use full attention */
            ds_causal_attention_aligned(attn_out, q, k_base, v_base,
                                        1, kv_seq_len, n_heads, n_kv_heads, head_dim,
                                        scale, cache_offset, kv_stride);
        } else {
            /* Steady state: gap between reference and window.
             * Compute Q@K^T for reference range and window range separately,
             * then combine into a single softmax. */
            ds_rswa_attention_aligned(attn_out, q, k_base, v_base,
                                       1, n_heads, n_kv_heads, head_dim,
                                       scale, kv_stride,
                                       prefill_len, window_start, kv_seq_len);
        }
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
    if (ctx->profile_enabled) ctx->perf_layer_proj_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;

    /* Post-attention: fused residual add + RMS norm.
     * This combines out = x + proj_out and norm_out = rms_norm(out, weight)
     * into a single operation, eliminating one pass over hidden=1280 elements. */
    ds_residual_rms_norm(out, x_norm, x, proj_out, layer->post_attn_norm, hidden, cfg->dec_rms_norm_eps);

    /* MLP (dense FFN for layer 0, MoE for layers 1-11) */
    if (ctx->profile_enabled) t0 = ds_time_sec();
    float *mlp_out = ctx->dec_expert_out;  /* reuse preallocated buffer */
    mlp_forward(mlp_out, x_norm, layer, cfg, ctx->metal_ctx, layer_idx,
                ctx->dec_dense_gate, ctx->dec_dense_up, ctx->dec_dense_swiglu,
                ctx->moe_expert_gate_buf, ctx->moe_expert_up_buf,
                ctx->moe_expert_gate_up_buf, ctx->moe_expert_hidden_buf,
                ctx->moe_shared_gate_buf, ctx->moe_shared_up_buf,
                ctx->moe_shared_gate_up_buf, ctx->moe_shared_swiglu_buf,
                ctx->moe_shared_out_buf,
                ctx->moe_expert_outputs);

    /* Add MLP output + residual (SIMD vector add) */
    ds_vec_add(out, out, mlp_out, hidden);

    if (ctx->profile_enabled) {
        ctx->perf_layer_mlp_ms[layer_idx] += (ds_time_sec() - t0) * 1000.0;
        ctx->perf_layer_total_ms[layer_idx] += (ds_time_sec() - t_layer_start) * 1000.0;
    }
}

/* ========================================================================
 * Decoder Prefill (multiple tokens)
 * ======================================================================== */

void ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Allocate prefill buffers */
    float *x = (float *)malloc(seq_len * hidden * sizeof(float));
    float *x_norm = (float *)malloc(seq_len * hidden * sizeof(float));
    float *Q = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *K = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *V = (float *)malloc(seq_len * kv_dim * sizeof(float));

    memcpy(x, input_embeds, seq_len * hidden * sizeof(float));

    for (int l = 0; l < cfg->dec_layers; l++) {
        ds_dec_layer_t *layer = &dec->layers[l];

        /* Input RMSNorm */
        ds_rms_norm(x_norm, x, layer->input_norm, seq_len, hidden, cfg->dec_rms_norm_eps);

        /* QKV */
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

        /* Apply RoPE using precomputed cache */
        {
            int base = ctx->kv_cache_len;
            ds_apply_rope_neox(Q, ctx->rope_cache_cos + base * head_dim,
                               ctx->rope_cache_sin + base * head_dim,
                               seq_len, n_heads, head_dim);
            ds_apply_rope_neox(K, ctx->rope_cache_cos + base * head_dim,
                               ctx->rope_cache_sin + base * head_dim,
                               seq_len, n_kv_heads, head_dim);
        }

        /* Store K, V directly into F32 KV cache (no BF16 conversion) */
        for (int s = 0; s < seq_len; s++) {
            int pos = ctx->kv_cache_len + s;
            float *cache_k_row = ds_kv_k_row(ctx, l, pos);
            float *cache_v_row = ds_kv_v_row(ctx, l, pos);
            ds_kv_store_f32(cache_k_row, K + s * kv_dim, kv_dim);
            ds_kv_store_f32(cache_v_row, V + s * kv_dim, kv_dim);
        }

        /* Causal attention (prefill all tokens) — read directly from F32 KV cache */
        {
            int total_kv_len = ctx->kv_cache_len + seq_len;
            float *k_base = ds_kv_k_layer(ctx, l);
            float *v_base = ds_kv_v_layer(ctx, l);
            int kv_stride = ctx->_kv_row_stride;

            float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
            ds_causal_attention_aligned(attn_out, Q, k_base, v_base,
                                seq_len, total_kv_len,
                                n_heads, n_kv_heads, head_dim, scale, ctx->kv_cache_len, kv_stride);

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
             * Uses F32 weights from BF16→F32 conversion (cached in safetensors).
             * Gate precision is F32 instead of BF16, but top-K expert selection
             * is robust to small numerical differences — the same experts are
             * selected in practice. */
            float *gate_scores = (float *)malloc(seq_len * n_experts * sizeof(float));
            ds_linear_nobias_bf16(gate_scores, x_norm, layer->gate_weight_bf16,
                                   seq_len, hidden, n_experts);

            /* --- Batched softmax + top-K for all tokens --- */
            /* Store top-K indices and weights for all tokens */
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

                /* Scatter-add expert outputs to mlp_out with routing weights */
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

    ctx->kv_cache_len += seq_len;

    /* Compute logits from last token position (needed for first generated token) */
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
}

/* ========================================================================
 * Decoder Forward (single token)
 * ======================================================================== */

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
        decoder_layer_forward(ctx, x, out, l, ctx->kv_cache_len);
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
    ctx->kv_cache_len++;

    /* Final RMSNorm */
    ds_rms_norm(x, x, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* Debug: dump norm output for first decode steps */
    if (getenv("DS_DUMP_DECODE_STEPS") && _step_before < 865) {
        char path[256];
        snprintf(path, sizeof(path), "dump/c_dec_step%d_norm_last.bin", _step_before);
        FILE *df = fopen(path, "wb");
        if (df) { fwrite(x, sizeof(float), hidden, df); fclose(df); }
    }

    /* BF16 intermediate truncation: match Python's autocast("cuda", dtype=bfloat16).
     * Python computes the entire decoder in BF16, including attention Q/K/V,
     * attention scores, and the final RMSNorm output before LM head.
     * C's F32 path produces higher-precision logits where EOS tends to dominate.
     * Truncating the final hidden state to BF16 before LM head makes logits
     * closer to Python's distribution, reducing the EOS-vs-content logit gap. */
    extern int ds_bf16_simulate_python;
    if (ds_bf16_simulate_python && cfg->model_version == 3) {
        ds_bf16_truncate_array(x, hidden);
    }

    /* LM head: compute logits then sample with optional repetition penalty */
    int vocab = cfg->vocab_size;
    float *logits = ctx->dec_logits;  /* [vocab_size] buffer */
    const uint16_t *lm_w = dec->lm_head_bf16 ? dec->lm_head_bf16 : dec->tok_embeddings_bf16;
    float rp = ctx->repeat_penalty;  /* Repetition penalty (used in both fast and full paths) */

    /* ──── Fast argmax path (no temperature sampling) ────
     * Instead of computing all 129280 logits via sgemm (631MB BF16→F32),
     * use ds_argmax_matvec_bf16 which computes dot products on-the-fly
     * and only tracks the best value. For repetition penalty, we
     * selectively recompute only the ~hundred history tokens' logits
     * instead of all 129280. This saves ~55ms/step on LM head. */
    if (ctx->temperature <= 0.0f && lm_w) {
        double t_lm = ctx->profile_enabled ? ds_time_sec() : 0;

        /* Step 1: Argmax over all rows — GPU path if DS_METAL_LM env var set */
        int best_token;
        if (ctx->metal_ctx && ds_metal_is_available(ctx->metal_ctx) && getenv("DS_METAL_LM")) {
            best_token = ds_metal_lm_head_argmax(ctx->metal_ctx, x, lm_w, hidden, vocab);
        } else {
            best_token = ds_argmax_matvec_bf16(x, lm_w, hidden, vocab);
        }
        float best_val = ds_bf16_dot_row(x, lm_w, hidden, best_token);
        if (rp > 1.0f && ctx->token_history && ctx->token_history_len > 0) {
            /* First: if the argmax winner is in history, apply penalty */
            for (int i = 0; i < ctx->token_history_len; i++) {
                if (ctx->token_history[i] == best_token) {
                    best_val = (best_val > 0) ? best_val / rp : best_val * rp;
                    break;
                }
            }
            /* Then: compute logits for other history tokens that might
             * surpass the penalized best. Only check unique tokens. */
            for (int i = 0; i < ctx->token_history_len; i++) {
                int tid = ctx->token_history[i];
                if (tid >= 0 && tid < vocab && tid != best_token) {
                    float val = ds_bf16_dot_row(x, lm_w, hidden, tid);
                    val = (val > 0) ? val / rp : val * rp;
                    if (val > best_val) {
                        best_val = val;
                        best_token = tid;
                    }
                }
            }
        }

        /* Step 3: N-gram blocking — if best token is banned, find next best.
         * For ngram blocking we need the full argmax scan again with banned
         * tokens excluded — but typically ngram only bans 0-1 tokens, so
         * just check if best is banned and if so, rescan. */
        int ngram_n = ctx->no_repeat_ngram_size;
        if (ngram_n > 0 && ctx->token_history_len >= ngram_n - 1) {
            int prefix_len = ngram_n - 1;
            int *hist = ctx->token_history;
            int hist_len = ctx->token_history_len;
            for (int i = 0; i <= hist_len - prefix_len - 1; i++) {
                int match = 1;
                for (int j = 0; j < prefix_len; j++) {
                    if (hist[i + j] != hist[hist_len - prefix_len + j]) {
                        match = 0; break;
                    }
                }
                if (match) {
                    int banned = hist[i + prefix_len];
                    if (banned == best_token) {
                        /* Best token is banned — need to find next best.
                         * Fall back to full sgemm logits for this step. */
                        goto full_logits_path;
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

    /* N-gram blocking: prevent repeating any n-gram of given size.
     * If the last (n-1) tokens match a previously seen n-gram prefix,
     * ban the token that would complete it.
     * This matches HuggingFace no_repeat_ngram_size behavior. */
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

    /* Find argmax from logits */
    int token = 0;
    float best = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > best) {
            best = logits[i];
            token = i;
        }
    }

    /* Temperature sampling (if enabled) */
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
            ds_kv_store_f32(ds_kv_k_row(ctx, l, pos), K + s * kv_dim, kv_dim);
            ds_kv_store_f32(ds_kv_v_row(ctx, l, pos), V + s * kv_dim, kv_dim);
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
                mlp_forward(mlp_out, x_norm + s * hidden, layer, cfg, ctx->metal_ctx, l,
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

    /* Final RMSNorm + LM head for each token */
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

    /* Record all tokens in history */
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
    /* Weight loading is done in ds_ocr.c during ds_load() */
    return 0;
}
