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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declaration */
static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg);

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

    /* SwiGLU: SiLU(gate) * up */
    float *gate_up = (float *)malloc(2 * intermediate * sizeof(float));
    for (int i = 0; i < intermediate; i++) {
        gate_up[2 * i] = gate_buf[i];
        gate_up[2 * i + 1] = up_buf[i];
    }

    ds_swiglu_multiply(swiglu_buf, gate_up, 1, intermediate);
    free(gate_up);

    /* Down projection */
    ds_linear_nobias_bf16(output, swiglu_buf, layer->dense_down_weight_bf16, 1, intermediate, hidden);

    if (own_gate) free(gate_buf);
    if (own_up) free(up_buf);
    if (own_swiglu) free(swiglu_buf);
}

static void mlp_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg, int layer_idx,
                         float *dense_gate_buf, float *dense_up_buf, float *dense_swiglu_buf) {
    if (layer_idx < cfg->dec_first_k_dense) {
        /* Dense FFN (SwiGLU) */
        dense_ffn_forward(output, x, layer, cfg, dense_gate_buf, dense_up_buf, dense_swiglu_buf);
    } else {
        /* MoE */
        moe_forward(output, x, layer, cfg);
    }
}

static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg) {
    int hidden = cfg->dec_hidden;
    int moe_inter = cfg->dec_moe_inter;
    int n_experts = cfg->dec_n_routed_experts;
    int top_k = cfg->dec_top_k;
    int n_shared = cfg->dec_n_shared_experts;

    /* Step 1: Router gate scores */
    float *scores = (float *)malloc(n_experts * sizeof(float));
    ds_moe_router(scores, x, layer->gate_weight, hidden, n_experts);

    /* Step 2: Top-K expert selection */
    int top_indices[DS_MAX_EXPERTS];
    float top_weights[DS_MAX_EXPERTS];
    ds_moe_top_k(top_indices, top_weights, scores, n_experts, top_k);
    free(scores);

    /* Step 3: Forward through each selected routed expert */
    float *expert_outputs = (float *)calloc(top_k * hidden, sizeof(float));
    for (int k = 0; k < top_k; k++) {
        int expert_id = top_indices[k];
        ds_expert_forward(expert_outputs + k * hidden, x,
                          layer->experts[expert_id].gate_weight_bf16,
                          layer->experts[expert_id].up_weight_bf16,
                          layer->experts[expert_id].down_weight_bf16,
                          hidden, moe_inter);
    }

    /* Step 4: Combine routed expert outputs */
    ds_expert_combine(output, expert_outputs, top_indices, top_weights, top_k, hidden);
    free(expert_outputs);

    /* Step 5: Add shared expert outputs (always active) */
    if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
        int shared_inter = n_shared * moe_inter;

        /* Shared experts: fused gate+up → SwiGLU → down */
        float *shared_gate = (float *)malloc(shared_inter * sizeof(float));
        float *shared_up = (float *)malloc(shared_inter * sizeof(float));

        ds_linear_nobias_bf16(shared_gate, x, layer->shared_gate_weight_bf16, 1, hidden, shared_inter);
        ds_linear_nobias_bf16(shared_up, x, layer->shared_up_weight_bf16, 1, hidden, shared_inter);

        /* Interleave for SwiGLU */
        float *shared_gate_up = (float *)malloc(2 * shared_inter * sizeof(float));
        for (int i = 0; i < shared_inter; i++) {
            shared_gate_up[2 * i] = shared_gate[i];
            shared_gate_up[2 * i + 1] = shared_up[i];
        }
        free(shared_gate); free(shared_up);

        float *shared_swiglu = (float *)malloc(shared_inter * sizeof(float));
        ds_swiglu_multiply(shared_swiglu, shared_gate_up, 1, shared_inter);
        free(shared_gate_up);

        float *shared_out = (float *)malloc(hidden * sizeof(float));
        ds_linear_nobias_bf16(shared_out, shared_swiglu, layer->shared_down_weight_bf16,
                               1, shared_inter, hidden);
        free(shared_swiglu);

        /* Add shared expert output */
        for (int i = 0; i < hidden; i++) {
            output[i] += shared_out[i];
        }
        free(shared_out);
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

    /* Input RMSNorm */
    float *x_norm = ctx->dec_x_norm;
    ds_rms_norm(x_norm, x, layer->input_norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* QKV projections (fused single-token matvec) */
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

    /* Store K, V in KV cache */
    int cache_offset = ctx->kv_cache_len;
    float *cache_k = ctx->kv_cache_k + (size_t)layer_idx * ctx->kv_cache_max * kv_dim
                      + cache_offset * kv_dim;
    float *cache_v = ctx->kv_cache_v + (size_t)layer_idx * ctx->kv_cache_max * kv_dim
                      + cache_offset * kv_dim;
    memcpy(cache_k, k, kv_dim * sizeof(float));
    memcpy(cache_v, v, kv_dim * sizeof(float));

    /* Causal attention with GQA */
    float *attn_out = ctx->dec_attn_out;
    ds_causal_attention(attn_out, q, ctx->kv_cache_k + (size_t)layer_idx * ctx->kv_cache_max * kv_dim,
                        ctx->kv_cache_v + (size_t)layer_idx * ctx->kv_cache_max * kv_dim,
                        1, cache_offset + 1, n_heads, n_kv_heads, head_dim, scale, cache_offset);

    /* Output projection + residual */
    float *proj_out = ctx->dec_proj_out;
    ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, 1, q_dim, hidden);

    /* Post-attention RMSNorm + residual */
    for (int i = 0; i < hidden; i++) out[i] = x[i] + proj_out[i];
    ds_rms_norm(x_norm, out, layer->post_attn_norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* MLP (dense FFN for layer 0, MoE for layers 1-11) */
    float *mlp_out = (float *)malloc(hidden * sizeof(float));
    mlp_forward(mlp_out, x_norm, layer, cfg, layer_idx,
                ctx->dec_dense_gate, ctx->dec_dense_up, ctx->dec_dense_swiglu);

    /* Add MLP output + residual */
    for (int i = 0; i < hidden; i++) out[i] += mlp_out[i];
    free(mlp_out);
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

        /* Per-head RMSNorm (optional: V2/OCR-2 does not use these) */
        if (layer->q_norm_weight)
            ds_rms_norm_per_head(Q, layer->q_norm_weight, seq_len, n_heads, head_dim, cfg->dec_rms_norm_eps);
        if (layer->k_norm_weight)
            ds_rms_norm_per_head(K, layer->k_norm_weight, seq_len, n_kv_heads, head_dim, cfg->dec_rms_norm_eps);

        /* Compute RoPE for each position */
        int *positions = (int *)malloc(seq_len * sizeof(int));
        for (int i = 0; i < seq_len; i++) positions[i] = ctx->kv_cache_len + i;

        float *cos_buf = (float *)malloc(seq_len * head_dim * sizeof(float));
        float *sin_buf = (float *)malloc(seq_len * head_dim * sizeof(float));
        ds_compute_rope_neox(cos_buf, sin_buf, positions, seq_len, head_dim, cfg->dec_rope_theta);
        ds_apply_rope_neox(Q, cos_buf, sin_buf, seq_len, n_heads, head_dim);
        ds_apply_rope_neox(K, cos_buf, sin_buf, seq_len, n_kv_heads, head_dim);
        free(positions); free(cos_buf); free(sin_buf);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            int pos = ctx->kv_cache_len + s;
            float *cache_k = ctx->kv_cache_k + (size_t)l * ctx->kv_cache_max * kv_dim + pos * kv_dim;
            float *cache_v = ctx->kv_cache_v + (size_t)l * ctx->kv_cache_max * kv_dim + pos * kv_dim;
            memcpy(cache_k, K + s * kv_dim, kv_dim * sizeof(float));
            memcpy(cache_v, V + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention (prefill all tokens) */
        float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
        ds_causal_attention(attn_out, Q,
                            ctx->kv_cache_k + (size_t)l * ctx->kv_cache_max * kv_dim,
                            ctx->kv_cache_v + (size_t)l * ctx->kv_cache_max * kv_dim,
                            seq_len, ctx->kv_cache_len + seq_len,
                            n_heads, n_kv_heads, head_dim, scale, ctx->kv_cache_len);

        /* Output projection + residual */
        float *proj_out = (float *)malloc(seq_len * hidden * sizeof(float));
        ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, seq_len, q_dim, hidden);
        free(attn_out);

        for (int i = 0; i < seq_len * hidden; i++) x[i] += proj_out[i];
        free(proj_out);

        /* Post-attention RMSNorm */
        ds_rms_norm(x_norm, x, layer->post_attn_norm, seq_len, hidden, cfg->dec_rms_norm_eps);

        /* MLP: for prefill, process each token individually */
        float *mlp_out = (float *)malloc(seq_len * hidden * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            mlp_forward(mlp_out + s * hidden, x_norm + s * hidden, layer, cfg, l,
                        NULL, NULL, NULL); /* prefill uses separate allocs for dense */
        }

        for (int i = 0; i < seq_len * hidden; i++) x[i] += mlp_out[i];
        free(mlp_out);

        if (ds_verbose >= 2)
            fprintf(stderr, "Decoder prefill layer %d/%d done\n", l + 1, cfg->dec_layers);
    }

    ctx->kv_cache_len += seq_len;
    free(x); free(x_norm); free(Q); free(K); free(V);
}

/* ========================================================================
 * Decoder Forward (single token)
 * ======================================================================== */

int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed) {
    ds_moe_decoder_t *dec = &ctx->decoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;

    /* Copy input to working buffer */
    float *x = ctx->dec_x;
    memcpy(x, input_embed, hidden * sizeof(float));

    /* Process through all layers */
    float *out = (float *)malloc(hidden * sizeof(float));
    for (int l = 0; l < cfg->dec_layers; l++) {
        decoder_layer_forward(ctx, x, out, l, ctx->kv_cache_len);
        memcpy(x, out, hidden * sizeof(float));
    }
    free(out);

    /* Advance KV cache position */
    ctx->kv_cache_len++;

    /* Final RMSNorm */
    ds_rms_norm(x, x, dec->norm, 1, hidden, cfg->dec_rms_norm_eps);

    /* LM head: argmax over vocab (streaming, no full logits) */
    int token;
    if (dec->lm_head_bf16) {
        token = ds_argmax_matvec_bf16(x, dec->lm_head_bf16, hidden, cfg->vocab_size);
    } else {
        /* Tied embeddings: use tok_embeddings as lm_head */
        token = ds_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, hidden, cfg->vocab_size);
    }

    return token;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int ds_decoder_load(ds_ctx_t *ctx) {
    /* Weight loading is done in ds_ocr.c during ds_load() */
    return 0;
}
