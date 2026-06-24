/*
 * ds_deep_encoder.c - Encoders for DeepSeek-OCR
 *
 * V1: CLIP ViT-L/14 → takes SAM patch_embeds → CLIP output + SAM concat → projector
 * V2: Qwen2-0.5B based encoder with causal flow queries
 */

#include "ds_deep_encoder.h"
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include "ds_dump.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ========================================================================
 * CLIP ViT-L/14 Forward Pass (V1)
 * ======================================================================== */

/* CLIP layer forward: pre-LN + fused QKV attention + residual + pre-LN + MLP + residual */
static void clip_layer_forward(float *out, const float *x,
                                ds_clip_encoder_t *clip, int layer_idx,
                                int seq_len) {
    int dim = DS_CLIP_HIDDEN;      /* 1024 */
    int n_heads = DS_CLIP_HEADS;    /* 16 */
    int head_dim = DS_CLIP_HEAD_DIM; /* 64 */
    int mlp_dim = DS_CLIP_MLP_DIM;  /* 4096 */
    float scale = 1.0f / sqrtf((float)head_dim);

    float *ln1_w = clip->layers[layer_idx].layer_norm1_weight;
    float *ln1_b = clip->layers[layer_idx].layer_norm1_bias;
    float *qkv_w = clip->layers[layer_idx].qkv_proj_weight;
    float *qkv_b = clip->layers[layer_idx].qkv_proj_bias;
    float *out_w = clip->layers[layer_idx].out_proj_weight;
    float *out_b = clip->layers[layer_idx].out_proj_bias;
    float *ln2_w = clip->layers[layer_idx].layer_norm2_weight;
    float *ln2_b = clip->layers[layer_idx].layer_norm2_bias;
    float *fc1_w = clip->layers[layer_idx].mlp_fc1_weight;
    float *fc1_b = clip->layers[layer_idx].mlp_fc1_bias;
    float *fc2_w = clip->layers[layer_idx].mlp_fc2_weight;
    float *fc2_b = clip->layers[layer_idx].mlp_fc2_bias;

    /* Pre-LayerNorm1 */
    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    ds_layer_norm(x_norm, x, ln1_w, ln1_b, seq_len, dim, 1e-6f);

    /* Fused QKV: [seq, dim] @ [dim, 3*dim] */
    float *qkv = (float *)malloc(seq_len * 3 * dim * sizeof(float));
    ds_linear(qkv, x_norm, qkv_w, qkv_b, seq_len, dim, 3 * dim);

    float *Q = (float *)malloc(seq_len * dim * sizeof(float));
    float *K = (float *)malloc(seq_len * dim * sizeof(float));
    float *V = (float *)malloc(seq_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        memcpy(Q + s * dim, qkv + s * 3 * dim, dim * sizeof(float));
        memcpy(K + s * dim, qkv + s * 3 * dim + dim, dim * sizeof(float));
        memcpy(V + s * dim, qkv + s * 3 * dim + 2 * dim, dim * sizeof(float));
    }
    free(qkv);

    /* Bidirectional attention (CLIP uses full attention) */
    float *attn_out = (float *)malloc(seq_len * dim * sizeof(float));
    ds_bidirectional_attention(attn_out, Q, K, V, seq_len, n_heads, head_dim, scale);
    free(Q); free(K); free(V);

    /* Output projection + residual */
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(proj_out, attn_out, out_w, out_b, seq_len, dim, dim);
    free(attn_out);

    for (int i = 0; i < seq_len * dim; i++) out[i] = x[i] + proj_out[i];
    free(proj_out);

    /* Pre-LayerNorm2 + MLP (GELU) */
    ds_layer_norm(x_norm, out, ln2_w, ln2_b, seq_len, dim, 1e-6f);

    float *fc1 = (float *)malloc(seq_len * mlp_dim * sizeof(float));
    ds_linear(fc1, x_norm, fc1_w, fc1_b, seq_len, dim, mlp_dim);
    ds_gelu(fc1, seq_len * mlp_dim);

    float *fc2 = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(fc2, fc1, fc2_w, fc2_b, seq_len, mlp_dim, dim);

    for (int i = 0; i < seq_len * dim; i++) out[i] += fc2[i];

    free(x_norm); free(fc1); free(fc2);
}

/* Full CLIP encoder forward pass */
float *ds_clip_encoder_forward(ds_ctx_t *ctx, const float *patch_embeds,
                                int n_patches, const float *sam_features,
                                int n_sam_tokens, int *out_seq_len) {
    ds_clip_encoder_t *clip = &ctx->clip_encoder;
    ds_config_t *cfg = &ctx->config;
    int clip_dim = DS_CLIP_HIDDEN; /* 1024 */

    if (ds_verbose >= 1)
        fprintf(stderr, "CLIP encoder: %d patch_embeds from SAM\n", n_patches);

    /* Step 1: Build CLIP input embeddings
     *
     * For DeepSeek-OCR V1:
     *   CLIP takes SAM's [768, H', W'] spatial features, applies its own
     *   patch_embedding Conv2d(768→1024, k14, s14) to project and subsample.
     *
     * For Unlimited-OCR (V3):
     *   CLIP receives SAM's [1024, H', W'] downsampled features directly
     *   as patch_embeds, bypassing its own patch_embedding Conv2d.
     *   In Python: patch_embeds = local_features_1 (SAM output [B, 1024, 16, 16])
     *   Then: patch_embeds = patch_embeds.flatten(2).transpose(1, 2) → [B, 256, 1024]
     *
     * We detect V3 by checking if patch_embeds dim matches clip_dim (1024).
     * For V1, n_patches is 4096 (64×64 SAM patches) and data is 768-dim.
     * For V3, n_patches is 256 (16×16 SAM downsampled) and data is 1024-dim.
     */

    int clip_n_patches;
    float *clip_tokens;  /* [clip_n_patches, clip_dim] — CLIP patch tokens */

    if (cfg->model_version == 3) {
        /* Unlimited-OCR: SAM downsampled features used directly as CLIP patch_embeds.
         * sam_features is [n_sam_tokens, ds2_dim] = [256, 1024] in token format.
         * We need to reshape to [1024, 16, 16] spatial, then flatten to tokens.
         * Actually, sam_features is already [256, 1024] token format — just use it directly! */
        clip_n_patches = n_sam_tokens;  /* 256 for 1024×1024 input */
        clip_tokens = (float *)malloc(clip_n_patches * clip_dim * sizeof(float));
        /* Copy sam_features as-is: [n_sam_tokens, 1024] → [clip_n_patches, clip_dim] */
        memcpy(clip_tokens, sam_features, clip_n_patches * clip_dim * sizeof(float));

        if (ds_verbose >= 1)
            fprintf(stderr, "CLIP V3: using SAM features directly as patch_embeds (%d tokens x %d dim)\n",
                    clip_n_patches, clip_dim);
    } else {
        /* V1: Apply CLIP's patch_embedding Conv2d to SAM's spatial patch_embeds */
        int grid_h = (int)sqrtf((float)n_patches);
        int grid_w = n_patches / grid_h;

        float *patch_spatial = (float *)malloc(DS_SAM_EMBED_DIM * n_patches * sizeof(float));
        for (int p = 0; p < n_patches; p++) {
            for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
                patch_spatial[d * n_patches + p] = patch_embeds[p * DS_SAM_EMBED_DIM + d];
            }
        }

        int clip_h, clip_w;
        clip_tokens = (float *)malloc(clip_dim * n_patches * sizeof(float)); /* over-allocate */

        ds_conv2d(clip_tokens, patch_spatial, clip->patch_embedding_weight, NULL,
                  DS_SAM_EMBED_DIM, clip_dim, grid_h, grid_w, 14, 14, 14, 0);
        free(patch_spatial);

        clip_h = (grid_h - 14) / 14 + 1;
        clip_w = (grid_w - 14) / 14 + 1;
        clip_n_patches = clip_h * clip_w;

        /* Convert [1024, clip_n_patches] → [clip_n_patches, 1024] */
        float *clip_tokens_t = (float *)malloc(clip_n_patches * clip_dim * sizeof(float));
        for (int p = 0; p < clip_n_patches; p++) {
            for (int d = 0; d < clip_dim; d++) {
                clip_tokens_t[p * clip_dim + d] = clip_tokens[d * clip_n_patches + p];
            }
        }
        free(clip_tokens);
        clip_tokens = clip_tokens_t;
    }

    /* Build full input: CLS token + patch tokens */
    int total_len = 1 + clip_n_patches;
    float *x = (float *)calloc(total_len * clip_dim, sizeof(float));

    /* CLS token at position 0 */
    if (clip->class_embedding) {
        memcpy(x, clip->class_embedding, clip_dim * sizeof(float));
    }

    /* Patch tokens */
    memcpy(x + clip_dim, clip_tokens, clip_n_patches * clip_dim * sizeof(float));
    free(clip_tokens);

    /* Add position embedding */
    if (clip->position_embedding) {
        for (int i = 0; i < total_len; i++) {
            for (int d = 0; d < clip_dim; d++) {
                x[i * clip_dim + d] += clip->position_embedding[i * clip_dim + d];
            }
        }
    }

    /* Pre-LayerNorm */
    if (clip->pre_layernorm_weight) {
        float *tmp = (float *)malloc(total_len * clip_dim * sizeof(float));
        ds_layer_norm(tmp, x, clip->pre_layernorm_weight, clip->pre_layernorm_bias,
                      total_len, clip_dim, 1e-6f);
        free(x);
        x = tmp;
    }

    /* CLIP transformer layers (24 layers) */
    for (int l = 0; l < DS_CLIP_LAYERS; l++) {
        float *next = (float *)malloc(total_len * clip_dim * sizeof(float));
        clip_layer_forward(next, x, clip, l, total_len);
        free(x);
        x = next;
        if (ds_verbose >= 2)
            fprintf(stderr, "CLIP layer %d/%d done\n", l + 1, DS_CLIP_LAYERS);
    }

    /* Post-LayerNorm (not always present, but typically is) */
    if (clip->final_norm_weight) {
        float *tmp = (float *)malloc(total_len * clip_dim * sizeof(float));
        ds_layer_norm(tmp, x, clip->final_norm_weight, clip->final_norm_bias,
                      total_len, clip_dim, 1e-6f);
        free(x);
        x = tmp;
    }

    /* Extract CLIP output: remove CLS token (take [:, 1:]) */
    int clip_output_len = clip_n_patches; /* without CLS */
    float *clip_output = (float *)malloc(clip_output_len * clip_dim * sizeof(float));
    memcpy(clip_output, x + clip_dim, clip_output_len * clip_dim * sizeof(float));
    free(x); /* Free CLIP transformer output */

    /* =====================================================================
     * Concatenate CLIP output + SAM flattened features → projector input
     * =====================================================================
     * Python modeling code:
     *   local_features = cat(clip_output[:, 1:], sam_output.flatten(2).permute(0,2,1), dim=-1)
     *
     * For Unlimited-OCR (V3):
     *   clip_output[:, 1:] = [B, 256, 1024]  (256 CLIP patch tokens)
     *   sam_output.flatten(2).permute(0,2,1) = [B, 256, 1024]  (256 SAM tokens)
     *   concat → [B, 256, 2048]
     *
     * For DeepSeek-OCR V1:
     *   clip_output[:, 1:] = [B, clip_n_patches, 1024]
     *   sam_output = [B, n_sam_tokens, 1024] (visual_tokens from SAM neck)
     *   For V1: clip_n_patches may differ from n_sam_tokens — this is known
     *   to be broken for V1. V3 is the primary target.
     * ===================================================================== */

    float *concat_features = NULL;
    int concat_len = 0;
    int proj_in_dim = 0;

    if (cfg->model_version == 3 && clip_output_len == n_sam_tokens) {
        /* Unlimited-OCR: CLIP and SAM have same token count (256 each for 1024×1024 input)
         * Concat: [256, 1024] + [256, 1024] → [256, 2048] along feature dim */
        concat_len = clip_output_len;
        proj_in_dim = clip_dim + cfg->sam_ds2_dim; /* 1024 + 1024 = 2048 */
        concat_features = (float *)malloc(concat_len * proj_in_dim * sizeof(float));
        for (int t = 0; t < concat_len; t++) {
            memcpy(concat_features + t * proj_in_dim,
                   clip_output + t * clip_dim, clip_dim * sizeof(float));
            memcpy(concat_features + t * proj_in_dim + clip_dim,
                   sam_features + t * cfg->sam_ds2_dim, cfg->sam_ds2_dim * sizeof(float));
        }
        free(clip_output);

        if (ds_verbose >= 1)
            fprintf(stderr, "CLIP+SAM concat: [%d, %d] + [%d, %d] → [%d, %d]\n",
                    clip_output_len, clip_dim, n_sam_tokens, cfg->sam_ds2_dim,
                    concat_len, proj_in_dim);
    } else {
        /* Fallback: use CLIP output only (broken for V1, but won't crash) */
        concat_features = clip_output;
        concat_len = clip_output_len;
        proj_in_dim = clip_dim;
        if (ds_verbose >= 1)
            fprintf(stderr, "Warning: CLIP+SAM concat skipped (clip=%d, sam=%d), using CLIP only\n",
                    clip_output_len, n_sam_tokens);
    }

    /* Project to decoder dimension via projector: [concat_len, proj_in_dim] → [concat_len, 1280] */
    int dec_hidden = cfg->dec_hidden; /* 1280 */
    float *projected = (float *)malloc(concat_len * dec_hidden * sizeof(float));
    if (ctx->projector.weight) {
        ds_linear(projected, concat_features, ctx->projector.weight, ctx->projector.bias,
                  concat_len, proj_in_dim, dec_hidden);
    } else {
        memset(projected, 0, concat_len * dec_hidden * sizeof(float));
    }
    free(concat_features);

    if (cfg->model_version == 3) {
        /* =====================================================================
         * Insert image_newline between grid rows + append view_seperator
         * =====================================================================
         * Python code:
         *   global_features = proj_output.view(h, w, n_dim)
         *   global_features = cat([global_features, image_newline[None,None,:].expand(h,1,n_dim)], dim=1)
         *   global_features = global_features.view(-1, n_dim)  # [272, 1280]
         *   global_local_features = cat([global_features, view_seperator[None,:]], dim=0)  # [273, 1280]
         *
         * Layout: 16 rows × (16 patches + 1 newline) + 1 view_seperator = 273 tokens
         * ===================================================================== */
        int grid_h = (int)sqrtf((float)concat_len); /* 16 */
        int grid_w = concat_len / grid_h;           /* 16 */
        float *image_newline = ctx->vis_tokenizer.image_newline;   /* [1280] */
        float *view_sep = ctx->vis_tokenizer.view_seperator;       /* [1280] */

        if (image_newline && view_sep && grid_h * grid_w == concat_len) {
            /* Build: [row0_patches, newline, row1_patches, newline, ..., view_sep]
             * Each row has grid_w patches + 1 newline = grid_w+1 tokens
             * Total = grid_h * (grid_w + 1) + 1 */
            int final_len = grid_h * (grid_w + 1) + 1; /* 16*17+1 = 273 */
            float *final_out = (float *)malloc(final_len * dec_hidden * sizeof(float));
            int dst = 0;
            for (int row = 0; row < grid_h; row++) {
                /* Copy grid_w patches from this row */
                memcpy(final_out + dst * dec_hidden,
                       projected + row * grid_w * dec_hidden,
                       grid_w * dec_hidden * sizeof(float));
                dst += grid_w;
                /* Insert image_newline */
                memcpy(final_out + dst * dec_hidden, image_newline, dec_hidden * sizeof(float));
                dst += 1;
            }
            /* Append view_seperator */
            memcpy(final_out + dst * dec_hidden, view_sep, dec_hidden * sizeof(float));
            dst += 1;

            free(projected);
            projected = final_out;
            *out_seq_len = final_len;

            if (ds_verbose >= 1)
                fprintf(stderr, "CLIP+SAM+Projector: %d patches → %d tokens (with newlines + view_sep)\n",
                        concat_len, final_len);
        } else {
            *out_seq_len = concat_len;
            if (ds_verbose >= 1)
                fprintf(stderr, "Warning: image_newline/view_seperator not loaded or grid mismatch (%d=%d*%d)\n",
                        concat_len, grid_h, grid_w);
        }
    } else {
        *out_seq_len = concat_len;
        if (ds_verbose >= 1)
            fprintf(stderr, "CLIP encoder: %d patches in -> %d tokens out (projected to %d)\n",
                    n_patches, concat_len, dec_hidden);
    }

    return projected;
}

/* ========================================================================
 * DeepEncoder V2 Forward Pass (Qwen2-0.5B based)
 * ======================================================================== */

/* Encoder-specific linear layers with float64 accumulation.
 * The 24-layer DeepEncoder amplifies float32 rounding errors dramatically.
 * Accumulating dot products in float64 while keeping weights/inputs as float32
 * costs ~2x in compute but dramatically reduces precision drift. */
static void enc_linear_f64acc(float *y, const float *x, const float *W, const float *b,
                               int seq_len, int in_dim, int out_dim) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            double sum = (b != NULL) ? (double)b[o] : 0.0;
            for (int i = 0; i < in_dim; i++) {
                sum += (double)x_row[i] * (double)w_row[i];
            }
            y_row[o] = (float)sum;
        }
    }
}

static void enc_linear_nobias_f64acc(float *y, const float *x, const float *W,
                                       int seq_len, int in_dim, int out_dim) {
    enc_linear_f64acc(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Apply RoPE to Q/K vectors for encoder (Qwen2 style)
 * Uses float64 for frequency, angle, cos/sin computation to avoid
 * trig precision drift accumulating over 24 layers. */
static void enc_apply_rope(float *q, float *k, int seq_len, int n_q_heads,
                            int n_kv_heads, int head_dim, float rope_theta) {
    /* Apply RoPE in split-half (NeoX/Llama) style:
     * x1 = x[:half], x2 = x[half:]
     * rotated = cat(-x2, x1)
     * result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    double theta_d = (double)rope_theta;
    for (int pos = 0; pos < seq_len; pos++) {
        /* Q heads */
        for (int h = 0; h < n_q_heads; h++) {
            float *qh = q + pos * n_q_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / pow(theta_d, (double)(2 * i) / (double)head_dim);
                double angle = (double)pos * freq;
                double cos_t = cos(angle);
                double sin_t = sin(angle);
                double x1 = (double)qh[i];
                double x2 = (double)qh[half + i];
                qh[i]        = (float)(x1 * cos_t - x2 * sin_t);
                qh[half + i] = (float)(x2 * cos_t + x1 * sin_t);
            }
        }
        /* K heads */
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + pos * n_kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / pow(theta_d, (double)(2 * i) / (double)head_dim);
                double angle = (double)pos * freq;
                double cos_t = cos(angle);
                double sin_t = sin(angle);
                double x1 = (double)kh[i];
                double x2 = (double)kh[half + i];
                kh[i]        = (float)(x1 * cos_t - x2 * sin_t);
                kh[half + i] = (float)(x2 * cos_t + x1 * sin_t);
            }
        }
    }
}

/* GQA attention with mixed mask (bidirectional for visual, causal for flow)
 * Uses float64 for QK^T dot product, softmax, and V accumulation to mitigate
 * precision drift in the 24-layer DeepEncoder pipeline. */
static void enc_gqa_mixed_attention(float *out, const float *Q, const float *K,
                                     const float *V, int visual_len, int total_len,
                                     int n_q_heads, int n_kv_heads, int head_dim) {
    double scale = 1.0 / sqrt((double)head_dim);
    int kv_group_size = n_q_heads / n_kv_heads;

    /* Allocate double buffer for V accumulation */
    double *o_row_d = (double *)malloc(head_dim * sizeof(double));

    for (int h = 0; h < n_q_heads; h++) {
        int kv_h = h / kv_group_size;

        for (int qi = 0; qi < total_len; qi++) {
            const float *q_vec = Q + qi * n_q_heads * head_dim + h * head_dim;

            int vis_end;
            if (qi < visual_len) {
                vis_end = visual_len;
            } else {
                vis_end = qi + 1;
            }

            /* Float64 online softmax with QK^T and V accumulation */
            double max_score = -1e30;
            double sum_exp = 0.0;
            for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

            for (int ki = 0; ki < vis_end; ki++) {
                const float *k_vec = K + ki * n_kv_heads * head_dim + kv_h * head_dim;
                const float *v_vec = V + ki * n_kv_heads * head_dim + kv_h * head_dim;

                /* Float64 QK dot product */
                double s = 0.0;
                for (int d = 0; d < head_dim; d++)
                    s += (double)q_vec[d] * (double)k_vec[d];
                s *= scale;

                if (s > max_score) {
                    double correction = exp(max_score - s);
                    sum_exp = sum_exp * correction + 1.0;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] = o_row_d[d] * correction + (double)v_vec[d];
                    max_score = s;
                } else {
                    double wt = exp(s - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] += wt * (double)v_vec[d];
                }
            }

            float *out_vec = out + qi * n_q_heads * head_dim + h * head_dim;
            if (sum_exp > 0.0) {
                double inv = 1.0 / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    out_vec[d] = (float)(o_row_d[d] * inv);
            } else {
                for (int d = 0; d < head_dim; d++) out_vec[d] = 0.0f;
            }
        }
    }

    free(o_row_d);
}

static void enc_v2_layer_forward(float *out, const float *x,
                                  ds_ctx_t *ctx, int layer_idx,
                                  int visual_len, int total_len) {
    ds_deep_encoder_t *enc = &ctx->encoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->enc_hidden;
    int n_q_heads = cfg->enc_heads;
    int n_kv_heads = cfg->enc_kv_heads;
    int head_dim = cfg->enc_head_dim;
    int intermediate = cfg->enc_intermediate;
    int kv_dim = n_kv_heads * head_dim;

    float *ln1_w = enc->layers[layer_idx].layer_norm1_weight;
    float *wq_w = enc->layers[layer_idx].wq_weight;
    float *wk_w = enc->layers[layer_idx].wk_weight;
    float *wv_w = enc->layers[layer_idx].wv_weight;
    float *wo_w = enc->layers[layer_idx].wo_weight;
    float *wq_b = enc->layers[layer_idx].wq_bias;
    float *wk_b = enc->layers[layer_idx].wk_bias;
    float *wv_b = enc->layers[layer_idx].wv_bias;
    float *ln2_w = enc->layers[layer_idx].layer_norm2_weight;
    float *gate_w = enc->layers[layer_idx].gate_weight;
    float *up_w = enc->layers[layer_idx].up_weight;
    float *down_w = enc->layers[layer_idx].down_weight;

    /* Pre-attention RMSNorm */
    float *x_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(x_norm, x, ln1_w, total_len, hidden, cfg->dec_rms_norm_eps);

    /* QKV projections with bias (GQA: Q is full, K/V are smaller) */
    float *Q = (float *)malloc(total_len * hidden * sizeof(float));
    float *K = (float *)malloc(total_len * kv_dim * sizeof(float));
    float *V = (float *)malloc(total_len * kv_dim * sizeof(float));

    ds_linear(Q, x_norm, wq_w, wq_b, total_len, hidden, hidden);
    ds_linear(K, x_norm, wk_w, wk_b, total_len, hidden, kv_dim);
    ds_linear(V, x_norm, wv_w, wv_b, total_len, hidden, kv_dim);

    /* Debug: dump QKV before RoPE for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_Q_before_rope", Q, total_len * hidden, "[288, 896]");
        ds_dump_tensor("enc0_K_before_rope", K, total_len * kv_dim, "[288, 128]");
        ds_dump_tensor("enc0_V_before_rope", V, total_len * kv_dim, "[288, 128]");
    }

    /* Apply RoPE to Q and K */
    enc_apply_rope(Q, K, total_len, n_q_heads, n_kv_heads, head_dim, cfg->enc_rope_theta);

    /* Debug: dump QKV after RoPE for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_Q_after_rope", Q, total_len * hidden, "[288, 896]");
        ds_dump_tensor("enc0_K_after_rope", K, total_len * kv_dim, "[288, 128]");
    }

    /* GQA mixed attention (bidirectional for visual, causal for flow) */
    float *attn_out = (float *)malloc(total_len * hidden * sizeof(float));
    enc_gqa_mixed_attention(attn_out, Q, K, V, visual_len, total_len,
                            n_q_heads, n_kv_heads, head_dim);

    /* Debug: dump attention output for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_attn_out", attn_out, total_len * hidden, "[288, 896]");
    }

    /* Output projection + residual */
    float *proj_out = (float *)malloc(total_len * hidden * sizeof(float));
    ds_linear_nobias(proj_out, attn_out, wo_w, total_len, hidden, hidden);
    for (int i = 0; i < total_len * hidden; i++) out[i] = x[i] + proj_out[i];

    /* Pre-FFN RMSNorm + SwiGLU MLP */
    float *ffn_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(ffn_norm, out, ln2_w, total_len, hidden, cfg->dec_rms_norm_eps);

    /* Gate + Up */
    float *gate_buf = (float *)malloc(total_len * intermediate * sizeof(float));
    float *up_buf = (float *)malloc(total_len * intermediate * sizeof(float));
    ds_linear_nobias(gate_buf, ffn_norm, gate_w, total_len, hidden, intermediate);
    ds_linear_nobias(up_buf, ffn_norm, up_w, total_len, hidden, intermediate);

    /* SwiGLU: silu(gate) * up (float64 for numerical stability) */
    for (int i = 0; i < total_len * intermediate; i++) {
        double g = (double)gate_buf[i];
        double sigmoid_g = 1.0 / (1.0 + exp(-g));
        gate_buf[i] = (float)(g * sigmoid_g * (double)up_buf[i]);
    }
    free(up_buf);

    /* Down projection + residual */
    float *down_out = (float *)malloc(total_len * hidden * sizeof(float));
    ds_linear_nobias(down_out, gate_buf, down_w, total_len, intermediate, hidden);
    for (int i = 0; i < total_len * hidden; i++) out[i] += down_out[i];

    free(x_norm); free(Q); free(K); free(V);
    free(attn_out); free(proj_out); free(ffn_norm);
    free(gate_buf); free(down_out);
}

float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len,
                               int n_causal_queries, const float *causal_queries) {
    ds_deep_encoder_t *enc = &ctx->encoder;
    ds_config_t *cfg = &ctx->config;

    int visual_len = n_tokens;
    int total_len = visual_len + n_causal_queries;
    int hidden = cfg->enc_hidden;

    if (ds_verbose >= 1)
        fprintf(stderr, "DeepEncoder V2: %d visual + %d causal flow = %d total\n",
                visual_len, n_causal_queries, total_len);

    /* Concatenate visual tokens and causal flow queries */
    float *x = (float *)calloc(total_len * hidden, sizeof(float));
    memcpy(x, visual_tokens, visual_len * hidden * sizeof(float));

    if (n_causal_queries > 0 && causal_queries) {
        memcpy(x + visual_len * hidden,
               causal_queries,
               n_causal_queries * hidden * sizeof(float));
    }

    /* DUMP: encoder input (visual + causal queries concatenated) */
    ds_dump_tensor("enc_input", x, total_len * hidden, "[512, 896]");

    /* Optional: override encoder input with Python reference for debugging.
     * This isolates encoder precision from SAM/resize differences.
     * Usage: DS_LOAD_ENC_INPUT=dump/py_enc_input_crop0.bin */
    {
        const char *load_enc_input = getenv("DS_LOAD_ENC_INPUT");
        if (load_enc_input) {
            FILE *ef = fopen(load_enc_input, "rb");
            if (ef) {
                fread(x, sizeof(float), total_len * hidden, ef);
                fclose(ef);
                fprintf(stderr, "Loaded encoder input from %s (%d x %d)\n",
                        load_enc_input, total_len, hidden);
                /* Re-dump to confirm */
                ds_dump_tensor("enc_input_loaded", x, total_len * hidden, "[512, 896]");
            } else {
                fprintf(stderr, "Warning: DS_LOAD_ENC_INPUT=%s not found\n", load_enc_input);
            }
        }
    }

    /* Transformer layers */
    for (int l = 0; l < cfg->enc_layers; l++) {
        float *next = (float *)malloc(total_len * hidden * sizeof(float));
        enc_v2_layer_forward(next, x, ctx, l, visual_len, total_len);
        free(x);
        x = next;

        /* DUMP: each encoder layer output */
        {
            char dump_name[64];
            snprintf(dump_name, sizeof(dump_name), "enc_layer%d", l);
            ds_dump_tensor(dump_name, x, total_len * hidden, "[512, 896]");
        }
    }

    /* Final norm */
    float *x_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(x_norm, x, enc->final_norm_weight, total_len, hidden, cfg->dec_rms_norm_eps);
    free(x);

    /* DUMP: after final norm (full [512, 896] including visual + causal) */
    ds_dump_tensor("enc_final_norm", x_norm, total_len * hidden, "[512, 896]");

    /* Extract only causal flow query tokens, project to decoder dim */
    int output_len = n_causal_queries;
    float *encoder_output = (float *)malloc(output_len * cfg->dec_hidden * sizeof(float));

    if (ctx->projector.weight) {
        ds_linear(encoder_output, x_norm + visual_len * hidden,
                  ctx->projector.weight, ctx->projector.bias,
                  output_len, hidden, cfg->dec_hidden);
    } else {
        memcpy(encoder_output, x_norm + visual_len * hidden,
               output_len * hidden * sizeof(float));
    }

    free(x_norm);

    /* DUMP: projector output [256, 1280] */
    ds_dump_tensor("proj_output", encoder_output, output_len * cfg->dec_hidden, "[256, 1280]");
    *out_seq_len = output_len;

    if (ds_verbose >= 1) {
        fprintf(stderr, "DeepEncoder V2: %d in -> %d out\n", total_len, output_len);
        /* Debug: print first few values of encoder output */
        float sum = 0, max_val = 0;
        for (int i = 0; i < output_len * cfg->dec_hidden; i++) {
            float v = encoder_output[i] < 0 ? -encoder_output[i] : encoder_output[i];
            sum += v;
            if (v > max_val) max_val = v;
        }
        fprintf(stderr, "  encoder output: mean_abs=%.4f max_abs=%.4f first=[%.4f, %.4f, %.4f, %.4f]\n",
                sum / (output_len * cfg->dec_hidden), max_val,
                encoder_output[0], encoder_output[1], encoder_output[2], encoder_output[3]);
    }

    return encoder_output;
}

/* ========================================================================
 * Unified Encoder Forward (dispatches to V1 or V2)
 * ======================================================================== */

float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len,
                           int n_causal_queries, const float *causal_queries) {
    if (ctx->config.enc_type == 1) {
        /* V1: CLIP path — visual_tokens here are SAM features
         * CLIP needs patch_embeds separately, but for the unified API
         * we pass SAM features directly */
        return ds_clip_encoder_forward(ctx, visual_tokens, n_tokens,
                                       visual_tokens, n_tokens, out_seq_len);
    } else {
        return ds_encoder_forward_v2(ctx, visual_tokens, n_tokens, out_seq_len,
                                      n_causal_queries, causal_queries);
    }
}

/* Weight Loading (done in ds_ocr.c during ds_load()) */
int ds_encoder_load(ds_ctx_t *ctx) {
    return 0;
}
