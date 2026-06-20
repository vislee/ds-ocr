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
                         ds_config_t *cfg,
                         float *expert_gate_buf, float *expert_up_buf,
                         float *expert_gate_up_buf, float *expert_hidden_buf,
                         float *shared_gate_buf, float *shared_up_buf,
                         float *shared_gate_up_buf, float *shared_swiglu_buf,
                         float *shared_out_buf,
                         float *expert_outputs_buf);

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
        moe_forward(output, x, layer, cfg,
                    expert_gate_buf, expert_up_buf,
                    expert_gate_up_buf, expert_hidden_buf,
                    shared_gate_buf, shared_up_buf,
                    shared_gate_up_buf, shared_swiglu_buf,
                    shared_out_buf,
                    expert_outputs_buf);
    }
}

static void moe_forward(float *output, const float *x, ds_dec_layer_t *layer,
                         ds_config_t *cfg,
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
            scores[e] = expf(scores[e] - max_s);
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

    /* Step 3: Forward through each selected routed expert */
    float *expert_outputs = expert_outputs_buf ? expert_outputs_buf :
                            (float *)calloc(top_k * hidden, sizeof(float));
    for (int k = 0; k < top_k; k++) {
        int expert_id = top_indices[k];
        ds_expert_forward(expert_outputs + k * hidden, x,
                          layer->experts[expert_id].gate_weight_bf16,
                          layer->experts[expert_id].up_weight_bf16,
                          layer->experts[expert_id].down_weight_bf16,
                          hidden, moe_inter,
                          expert_gate_buf, expert_up_buf,
                          expert_gate_up_buf, expert_hidden_buf);
    }

    /* Step 4: Combine routed expert outputs */
    ds_expert_combine(output, expert_outputs, top_indices, top_weights, top_k, hidden);
    if (!expert_outputs_buf) free(expert_outputs);
    else memset(expert_outputs_buf, 0, top_k * hidden * sizeof(float));

    /* Step 5: Add shared expert outputs (always active) */
    if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
        int shared_inter = n_shared * moe_inter;

        ds_linear_nobias_bf16(shared_gate_buf, x, layer->shared_gate_weight_bf16, 1, hidden, shared_inter);
        ds_linear_nobias_bf16(shared_up_buf, x, layer->shared_up_weight_bf16, 1, hidden, shared_inter);

        for (int i = 0; i < shared_inter; i++) {
            shared_gate_up_buf[2 * i] = shared_gate_buf[i];
            shared_gate_up_buf[2 * i + 1] = shared_up_buf[i];
        }

        ds_swiglu_multiply(shared_swiglu_buf, shared_gate_up_buf, 1, shared_inter);

        ds_linear_nobias_bf16(shared_out_buf, shared_swiglu_buf, layer->shared_down_weight_bf16,
                               1, shared_inter, hidden);

        for (int i = 0; i < hidden; i++) {
            output[i] += shared_out_buf[i];
        }
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
    float *mlp_out = ctx->dec_expert_out;  /* reuse preallocated buffer */
    mlp_forward(mlp_out, x_norm, layer, cfg, layer_idx,
                ctx->dec_dense_gate, ctx->dec_dense_up, ctx->dec_dense_swiglu,
                ctx->moe_expert_gate_buf, ctx->moe_expert_up_buf,
                ctx->moe_expert_gate_up_buf, ctx->moe_expert_hidden_buf,
                ctx->moe_shared_gate_buf, ctx->moe_shared_up_buf,
                ctx->moe_shared_gate_up_buf, ctx->moe_shared_swiglu_buf,
                ctx->moe_shared_out_buf,
                ctx->moe_expert_outputs);

    /* Add MLP output + residual */
    for (int i = 0; i < hidden; i++) out[i] += mlp_out[i];
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

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            int pos = ctx->kv_cache_len + s;
            float *cache_k = ctx->kv_cache_k + (size_t)l * ctx->kv_cache_max * kv_dim + pos * kv_dim;
            float *cache_v = ctx->kv_cache_v + (size_t)l * ctx->kv_cache_max * kv_dim + pos * kv_dim;
            memcpy(cache_k, K + s * kv_dim, kv_dim * sizeof(float));
            memcpy(cache_v, V + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Debug: dump cached K for layer 0 to verify cache layout */
        if (l == 0 && getenv("DS_DUMP_DECODER")) {
            FILE *df = fopen("dump/c_layer0_cached_K.bin", "wb");
            if (df) {
                /* Dump from cache base, seq_len positions */
                float *cache_base = ctx->kv_cache_k + (size_t)l * ctx->kv_cache_max * kv_dim;
                fwrite(cache_base, sizeof(float), seq_len * kv_dim, df);
                fclose(df);
            }
        }

        /* Causal attention (prefill all tokens) */
        float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
        ds_causal_attention(attn_out, Q,
                            ctx->kv_cache_k + (size_t)l * ctx->kv_cache_max * kv_dim,
                            ctx->kv_cache_v + (size_t)l * ctx->kv_cache_max * kv_dim,
                            seq_len, ctx->kv_cache_len + seq_len,
                            n_heads, n_kv_heads, head_dim, scale, ctx->kv_cache_len);

        /* Debug: dump layer 0 attn_out */
        if (l == 0 && getenv("DS_DUMP_DECODER")) {
            FILE *df = fopen("dump/c_layer0_attn_out.bin", "wb");
            if (df) { fwrite(attn_out, sizeof(float), seq_len * q_dim, df); fclose(df); }
            fprintf(stderr, "Dumped C layer 0 attn_out (%d x %d)\n", seq_len, q_dim);
        }

        /* Output projection + residual */
        float *proj_out = (float *)malloc(seq_len * hidden * sizeof(float));
        ds_linear_nobias_bf16(proj_out, attn_out, layer->wo_weight_bf16, seq_len, q_dim, hidden);
        free(attn_out);

        for (int i = 0; i < seq_len * hidden; i++) x[i] += proj_out[i];

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

        free(proj_out);

        /* Post-attention RMSNorm */
        ds_rms_norm(x_norm, x, layer->post_attn_norm, seq_len, hidden, cfg->dec_rms_norm_eps);

        /* MLP: dense FFN for layer 0 (batched sgemm), MoE for layers 1-11 (per-token) */
        if (l < cfg->dec_first_k_dense) {
            /* Dense FFN (layer 0): batched via BLAS sgemm — much faster than per-token */
            float *gate_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *up_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *gate_up_buf = (float *)malloc(seq_len * 2 * cfg->dec_intermediate * sizeof(float));
            float *swiglu_buf = (float *)malloc(seq_len * cfg->dec_intermediate * sizeof(float));
            float *mlp_out = (float *)malloc(seq_len * hidden * sizeof(float));

            ds_linear_nobias_bf16(gate_buf, x_norm, layer->dense_gate_weight_bf16,
                                   seq_len, hidden, cfg->dec_intermediate);
            ds_linear_nobias_bf16(up_buf, x_norm, layer->dense_up_weight_bf16,
                                   seq_len, hidden, cfg->dec_intermediate);

            for (int s = 0; s < seq_len; s++) {
                for (int i = 0; i < cfg->dec_intermediate; i++) {
                    gate_up_buf[(size_t)s * 2 * cfg->dec_intermediate + 2 * i] =
                        gate_buf[(size_t)s * cfg->dec_intermediate + i];
                    gate_up_buf[(size_t)s * 2 * cfg->dec_intermediate + 2 * i + 1] =
                        up_buf[(size_t)s * cfg->dec_intermediate + i];
                }
            }
            ds_swiglu_multiply(swiglu_buf, gate_up_buf, seq_len, cfg->dec_intermediate);
            ds_linear_nobias_bf16(mlp_out, swiglu_buf, layer->dense_down_weight_bf16,
                                   seq_len, cfg->dec_intermediate, hidden);

            for (int i = 0; i < seq_len * hidden; i++) x[i] += mlp_out[i];
            free(gate_buf); free(up_buf); free(gate_up_buf); free(swiglu_buf); free(mlp_out);
        } else {
            /* MoE MLP (layers 1-11): batched gate scoring + batched shared experts
             * + per-token routed expert forward. */
            int n_experts = cfg->dec_n_routed_experts;
            int top_k = cfg->dec_top_k;

            /* Batched BF16 gate scoring: per-token BF16 matvec to match decode path precision.
             * Using ds_linear_nobias_bf16(sg==1) → BF16 matvec (matches Python BF16 gate).
             * Batched sgemm uses F32 weights from BF16 conversion → different precision →
             * can select different experts, causing output divergence. */
            float *gate_scores = (float *)malloc(seq_len * n_experts * sizeof(float));
            for (int s = 0; s < seq_len; s++) {
                ds_moe_router_bf16(gate_scores + s * n_experts,
                                   x_norm + s * hidden,
                                   layer->gate_weight_bf16, hidden, n_experts);
            }

            /* Debug: dump gate scores for last token */
            if (getenv("DS_DUMP_GATE") && l <= 6) {
                char path[256];
                snprintf(path, sizeof(path), "dump/c_gate_layer%d_last.bin", l);
                FILE *df = fopen(path, "wb");
                if (df) { fwrite(gate_scores + (seq_len-1) * n_experts, sizeof(float), n_experts, df); fclose(df); }
            }

            float *mlp_out = (float *)malloc(seq_len * hidden * sizeof(float));
            for (int s = 0; s < seq_len; s++) {
                /* Gate scoring: softmax over ALL 64 experts first (matching Python),
                 * then select top-K and use softmax values as weights.
                 * Previously ds_moe_top_k did softmax over only top-K, which was wrong. */
                float *s_scores = gate_scores + s * n_experts;
                /* Softmax over all n_experts */
                {
                    float max_s = -1e30f;
                    for (int e = 0; e < n_experts; e++)
                        if (s_scores[e] > max_s) max_s = s_scores[e];
                    float sum_exp = 0.0f;
                    for (int e = 0; e < n_experts; e++) {
                        s_scores[e] = expf(s_scores[e] - max_s);
                        sum_exp += s_scores[e];
                    }
                    float inv = 1.0f / sum_exp;
                    for (int e = 0; e < n_experts; e++)
                        s_scores[e] *= inv;
                }

                /* Select top-K from softmax'd scores */
                int top_indices[DS_MAX_EXPERTS];
                float top_weights[DS_MAX_EXPERTS];
                for (int k = 0; k < top_k; k++) {
                    int best = -1;
                    float best_score = -1e30f;
                    for (int e = 0; e < n_experts; e++) {
                        int already = 0;
                        for (int j = 0; j < k; j++)
                            if (top_indices[j] == e) { already = 1; break; }
                        if (already) continue;
                        if (s_scores[e] > best_score) {
                            best_score = s_scores[e];
                            best = e;
                        }
                    }
                    top_indices[k] = best;
                    top_weights[k] = best_score;
                }

                /* Routed experts (per-token, same as decode path) */
                float *expert_outputs = ctx->moe_expert_outputs;
                for (int k = 0; k < top_k; k++) {
                    int expert_id = top_indices[k];
                    ds_expert_forward(expert_outputs + k * hidden,
                                      x_norm + s * hidden,
                                      layer->experts[expert_id].gate_weight_bf16,
                                      layer->experts[expert_id].up_weight_bf16,
                                      layer->experts[expert_id].down_weight_bf16,
                                      hidden, cfg->dec_moe_inter,
                                      ctx->moe_expert_gate_buf, ctx->moe_expert_up_buf,
                                      ctx->moe_expert_gate_up_buf, ctx->moe_expert_hidden_buf);
                }
                ds_expert_combine(mlp_out + s * hidden, expert_outputs,
                                  top_indices, top_weights, top_k, hidden);
                memset(ctx->moe_expert_outputs, 0, top_k * hidden * sizeof(float));

                /* Shared experts (per-token, same as decode path) */
                if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
                    int shared_inter = cfg->dec_n_shared_experts * cfg->dec_moe_inter;
                    ds_linear_nobias_bf16(ctx->moe_shared_gate_buf, x_norm + s * hidden,
                                           layer->shared_gate_weight_bf16, 1, hidden, shared_inter);
                    ds_linear_nobias_bf16(ctx->moe_shared_up_buf, x_norm + s * hidden,
                                           layer->shared_up_weight_bf16, 1, hidden, shared_inter);
                    for (int i = 0; i < shared_inter; i++) {
                        ctx->moe_shared_gate_up_buf[2 * i] = ctx->moe_shared_gate_buf[i];
                        ctx->moe_shared_gate_up_buf[2 * i + 1] = ctx->moe_shared_up_buf[i];
                    }
                    ds_swiglu_multiply(ctx->moe_shared_swiglu_buf,
                                       ctx->moe_shared_gate_up_buf, 1, shared_inter);
                    ds_linear_nobias_bf16(ctx->moe_shared_out_buf,
                                           ctx->moe_shared_swiglu_buf,
                                           layer->shared_down_weight_bf16,
                                           1, shared_inter, hidden);
                    for (int i = 0; i < hidden; i++) {
                        mlp_out[s * hidden + i] += ctx->moe_shared_out_buf[i];
                    }
                }
            }
            for (int i = 0; i < seq_len * hidden; i++) x[i] += mlp_out[i];
            free(gate_scores);
            free(mlp_out);
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

    /* Copy input to working buffer */
    float *x = ctx->dec_x;
    memcpy(x, input_embed, hidden * sizeof(float));

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

    /* LM head: compute logits then sample with optional repetition penalty */
    int vocab = cfg->vocab_size;
    float *logits = ctx->dec_logits;  /* [vocab_size] buffer */
    if (!logits) {
        /* Fallback: direct argmax (no penalty support) */
        int token;
        if (dec->lm_head_bf16) {
            token = ds_argmax_matvec_bf16(x, dec->lm_head_bf16, hidden, vocab);
        } else {
            token = ds_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, hidden, vocab);
        }
        return token;
    }

    /* Compute full logits vector */
    if (dec->lm_head_bf16) {
        ds_bf16_matvec_pub(logits, x, dec->lm_head_bf16, NULL, hidden, vocab);
    } else {
        ds_bf16_matvec_pub(logits, x, dec->tok_embeddings_bf16, NULL, hidden, vocab);
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

    /* Apply repetition penalty */
    float rp = ctx->repeat_penalty;
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
 * Weight Loading
 * ======================================================================== */

int ds_decoder_load(ds_ctx_t *ctx) {
    /* Weight loading is done in ds_ocr.c during ds_load() */
    return 0;
}
