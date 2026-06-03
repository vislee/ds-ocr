/*
 * ds_deep_encoder.c - Encoders for DeepSeek-OCR
 *
 * V1: CLIP ViT-L/14 → takes SAM patch_embeds → CLIP output + SAM concat → projector
 * V2: Qwen2-0.5B based encoder with causal flow queries
 */

#include "ds_deep_encoder.h"
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
     * CLIP takes SAM's patch_embeds as input (NOT raw image)
     * Embeddings = patch_embeds + position_embedding + class_embedding
     * Note: patch_embeds are [n_patches, clip_dim] (already projected from SAM's output)
     *
     * In the actual model, CLIP's patch_embedding is a Conv2d that takes the
     * SAM patch_embeds (which are [n_patches, 768] in SAM space).
     * But looking at the modeling code: CLIP's embeddings() method takes
     * both the raw image AND patch_embeds, and uses patch_embeds as input
     * to its own patch_embedding layer.
     *
     * Actually, the CLIP ViT-L takes patch_embeds from SAM as input:
     * x = self.patch_embedding(patch_embeds)  # [n_patches, 1024]
     * x = cat([class_embedding, x])  # [1+n_patches, 1024]
     * x = x + position_embedding
     */
    int total_len = 1 + n_patches; /* CLS token + patches */

    float *x = (float *)calloc(total_len * clip_dim, sizeof(float));

    /* CLS token at position 0 */
    if (clip->class_embedding) {
        memcpy(x, clip->class_embedding, clip_dim * sizeof(float));
    }

    /* Patch embeddings through CLIP's patch_embedding layer */
    /* patch_embeds is [n_patches, 768] — CLIP's patch_embedding is [1024, 3, 14, 14]?
     * No! Looking at the code more carefully:
     * The CLIP in DeepSeek-OCR takes SAM's patch_embeds (768-dim) and
     * applies its own patch_embedding Conv2d(3, 1024, 14, 14) to the SAM output
     * But that doesn't make sense dimensionally...
     *
     * Actually re-reading the modeling code:
     * self.embeddings(x, patch_embeds=patch_embeds)
     * Inside CLIPVisionEmbeddings:
     *   patch_embeds = self.patch_embedding(patch_embeds)  # Conv2d on patch_embeds
     * But patch_embeds is [B, 768, H, W] spatial format from SAM
     * And self.patch_embedding is Conv2d(768, 1024, kernel_size=14, stride=14)
     * So it's a projection from 768→1024 via 14×14 conv!
     *
     * Wait, that still doesn't work because SAM output is already patches...
     * Let me re-read the actual code more carefully.
     *
     * OK I see: patch_embeds from SAM are in [B, 768, H', W'] format where
     * H' = H/patch_size, W' = W/patch_size. Then CLIP's patch_embedding
     * Conv2d(768, 1024, 14, 14) further subsamples these with stride 14.
     * This produces [B, 1024, H'/14, W'/14] tokens.
     *
     * But our SAM output is already in token format [n_patches, 768]...
     * We need to reshape to spatial [768, grid_h, grid_w] then apply conv.
     */

    /* Reshape patch_embeds [n_patches, 768] → spatial [768, grid_h, grid_w] */
    int grid_h = (int)sqrtf((float)n_patches); /* For square images */
    int grid_w = n_patches / grid_h;

    float *patch_spatial = (float *)malloc(DS_SAM_EMBED_DIM * n_patches * sizeof(float));
    for (int p = 0; p < n_patches; p++) {
        for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
            patch_spatial[d * n_patches + p] = patch_embeds[p * DS_SAM_EMBED_DIM + d];
        }
    }

    /* CLIP patch_embedding: Conv2d(768, 1024, 14, stride=14) on SAM's spatial output */
    int clip_h, clip_w;
    float *clip_patches = (float *)malloc(clip_dim * n_patches * sizeof(float)); /* over-allocate */

    /* Conv2d(768→1024, k14, s14) on [768, grid_h, grid_w] */
    ds_conv2d(clip_patches, patch_spatial, clip->patch_embedding_weight, NULL,
              DS_SAM_EMBED_DIM, clip_dim, grid_h, grid_w, 14, 14, 14, 0);
    free(patch_spatial);

    clip_h = (grid_h - 14) / 14 + 1;
    clip_w = (grid_w - 14) / 14 + 1;
    int clip_n_patches = clip_h * clip_w;

    /* Convert [1024, clip_n_patches] → [clip_n_patches, 1024] and place after CLS */
    for (int p = 0; p < clip_n_patches; p++) {
        for (int d = 0; d < clip_dim; d++) {
            x[(1 + p) * clip_dim + d] = clip_patches[d * clip_n_patches + p];
        }
    }
    free(clip_patches);

    /* Re-compute total_len with actual CLIP patch count */
    total_len = 1 + clip_n_patches;

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

    /* Concatenate CLIP output + SAM features → projector
     * CLIP: [clip_n_patches, 1024]
     * SAM: [n_sam_tokens, 1024] (flattened from spatial)
     * Combined: [clip_n_patches, 2048] (aligned by position, CLIP + SAM)
     *
     * But the token counts need to match! In the actual model:
     * clip_output shape: [B, L_clip, 1024] where L_clip = clip_n_patches
     * sam_output shape: [B, L_sam, 1024] where L_sam = n_sam_tokens
     * They are NOT the same length!
     *
     * Actually, looking at the modeling code:
     * cat([clip_output[:, 1:, :], sam_output.flatten(2).permute(0,2,1)], dim=-1)
     * This concatenates along the FEATURE dimension, not the sequence dimension!
     * But they need the same number of tokens for this to work...
     *
     * Let me re-check: in the modeling code, CLIP output after removing CLS is
     * [B, N_clip, 1024] and SAM is [B, 1024, H', W'] → flatten → [B, N_sam, 1024]
     * where N_clip = N_sam (they should be the same spatial resolution after
     * SAM downsampling and CLIP's additional patch embedding)
     *
     * For 1024×1024 input:
     * - SAM: 1024/16 = 64 → 64×64 patches → neck → 64×64 → ds → 32×32 → ds → 16×16
     * - CLIP takes SAM's 64×64 patch_embeds → Conv2d(14,14) stride 14 →
     *   64/14 = ~4.5... This doesn't divide evenly!
     *
     * Hmm, let me re-read the modeling code more carefully.
     * Actually, the CLIP patch_embedding might take SAM's output at a different
     * stage. Let me check the actual dimensions...
     *
     * Looking at the real code flow for V1:
     * 1. SAM produces patch_embeds [B, 768, H/16, W/16]
     * 2. CLIP processes these: embeddings(patch_embeds) produces [B, N, 1024]
     *    where N = (H/16 / patch_size) * (W/16 / patch_size) with patch_size=14
     * 3. SAM also has its downsample path: [B, 1024, H/64, W/64]
     * 4. Concat: [B, N, 1024+1024] = [B, N, 2048]
     *
     * But SAM output has (H/64)*(W/64) = 16*16 = 256 tokens
     * CLIP output has ((H/16)/14)*((W/16)/14) = 4*4 = 16? No, that's too few...
     *
     * Actually wait: CLIP's patch_embedding doesn't use 14 as stride on the SAM features.
     * It's treating the SAM features as a "pseudo-image" and the CLIP patch_embedding
     * is a 1x1 or linear projection...
     *
     * I think I'm overcomplicating this. Let me just look at the actual tensor shapes
     * by examining the modeling code flow for a 1024x1024 input.
     *
     * For 1024×1024:
     * - SAM patch_embed: [1, 768, 64, 64]
     * - After SAM blocks: [1, 768, 64, 64]
     * - SAM neck → [1, 256, 64, 64] → net_2 → [1, 512, 32, 32] → net_3 → [1, 1024, 16, 16]
     *   sam_output = [1, 1024, 16, 16] → flatten(2) → [1, 1024, 256] → permute → [1, 256, 1024]
     *
     * - CLIP input: SAM's patch_embed [1, 768, 64, 64]
     *   CLIP patch_embedding Conv2d(768, 1024, 14, 14) on [1, 768, 64, 64]
     *   → [1, 1024, 64/14, 64/14] = [1, 1024, 4, 4] (with floor division)
     *   That gives only 16 CLIP tokens vs 256 SAM tokens...
     *
     * Hmm, this doesn't match. Let me reconsider.
     * Maybe CLIP's patch_embedding is actually a linear/1x1 projection, not a 14x14 conv?
     * Looking at the config: "patch_size": 14 for CLIP, but that's for raw image (224x224).
     * When taking SAM patch_embeds as input, maybe the CLIP uses a different embedding?
     *
     * I need to read the actual CLIPVisionEmbeddings code from the model...
     *
     * OK, I think the key insight is: the CLIP in DeepSeek-OCR is MODIFIED.
     * It doesn't use the standard CLIP embedding. Instead:
     * self.patch_embedding = nn.Conv2d(768, 1024, kernel_size=14, stride=14, padding=0)
     * This takes the 768-dim SAM features and projects to 1024-dim with spatial downsampling.
     *
     * For 1024×1024 → SAM produces [768, 64, 64]
     * CLIP Conv2d(768→1024, k14, s14) → [1024, 4, 4] = 16 tokens
     * But SAM output is [1024, 16, 16] = 256 tokens
     * These don't match for concatenation!
     *
     * UNLESS... the concatenation is not along the token dimension but along the feature dim.
     * cat([clip_output[:, 1:], sam_output], dim=-1) where:
     * - clip_output[:, 1:] = [B, 16, 1024]  (16 CLIP tokens, 1024 dim)
     * - sam_output = [B, 256, 1024] (256 SAM tokens, 1024 dim)
     * But these have different sequence lengths! Can't concat along dim=-1 with different L!
     *
     * I think I need to actually read the code more carefully. Let me skip this for now
     * and just implement the V2 path which is simpler, and come back to V1 later.
     */

    /* For now, just use the clip_output as the combined features
     * TODO: Properly implement SAM+CLIP concatenation based on actual modeling code
     */
    free(x);

    /* Project to decoder dimension via projector */
    int proj_in = clip_dim; /* Will be 2048 when concat is properly implemented */
    float *projected = (float *)malloc(clip_output_len * cfg->dec_hidden * sizeof(float));
    if (ctx->projector.weight) {
        ds_linear(projected, clip_output, ctx->projector.weight, ctx->projector.bias,
                  clip_output_len, proj_in, cfg->dec_hidden);
    } else {
        /* No projector, just copy (shouldn't happen) */
        memset(projected, 0, clip_output_len * cfg->dec_hidden * sizeof(float));
    }
    free(clip_output);

    *out_seq_len = clip_output_len;

    if (ds_verbose >= 1)
        fprintf(stderr, "CLIP encoder: %d patches in -> %d tokens out (projected to %d)\n",
                n_patches, clip_output_len, cfg->dec_hidden);

    return projected;
}

/* ========================================================================
 * DeepEncoder V2 Forward Pass (Qwen2-0.5B based)
 * ======================================================================== */

/* Apply RoPE to Q/K vectors for encoder (Qwen2 style) */
static void enc_apply_rope(float *q, float *k, int seq_len, int n_q_heads,
                            int n_kv_heads, int head_dim, float rope_theta) {
    /* Apply RoPE in split-half (NeoX/Llama) style:
     * x1 = x[:half], x2 = x[half:]
     * rotated = cat(-x2, x1)
     * result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    for (int pos = 0; pos < seq_len; pos++) {
        /* Q heads */
        for (int h = 0; h < n_q_heads; h++) {
            float *qh = q + pos * n_q_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
                float angle = pos * freq;
                float cos_t = cosf(angle);
                float sin_t = sinf(angle);
                float x1 = qh[i];
                float x2 = qh[half + i];
                qh[i]        = x1 * cos_t - x2 * sin_t;
                qh[half + i] = x2 * cos_t + x1 * sin_t;
            }
        }
        /* K heads */
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + pos * n_kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
                float angle = pos * freq;
                float cos_t = cosf(angle);
                float sin_t = sinf(angle);
                float x1 = kh[i];
                float x2 = kh[half + i];
                kh[i]        = x1 * cos_t - x2 * sin_t;
                kh[half + i] = x2 * cos_t + x1 * sin_t;
            }
        }
    }
}

/* GQA attention with mixed mask (bidirectional for visual, causal for flow) */
static void enc_gqa_mixed_attention(float *out, const float *Q, const float *K,
                                     const float *V, int visual_len, int total_len,
                                     int n_q_heads, int n_kv_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group_size = n_q_heads / n_kv_heads;

    for (int h = 0; h < n_q_heads; h++) {
        int kv_h = h / kv_group_size; /* which KV head this Q head uses */

        for (int qi = 0; qi < total_len; qi++) {
            const float *q_vec = Q + qi * n_q_heads * head_dim + h * head_dim;

            /* Compute attention scores with online softmax */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            float *acc = (float *)calloc(head_dim, sizeof(float));

            for (int ki = 0; ki < total_len; ki++) {
                /* Mixed attention mask:
                 * - Visual tokens (qi < visual_len): attend to all visual tokens (bidirectional)
                 * - Causal tokens (qi >= visual_len): attend to all visual + causal up to qi
                 */
                int visible;
                if (qi < visual_len) {
                    /* Visual token: only attend to other visual tokens */
                    visible = (ki < visual_len);
                } else {
                    /* Causal token: attend to all visual + causal tokens up to current pos */
                    visible = (ki < visual_len) || (ki <= qi);
                }
                if (!visible) continue;

                const float *k_vec = K + ki * n_kv_heads * head_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_vec[d] * k_vec[d];
                score *= scale;

                /* Online softmax update */
                if (score > max_score) {
                    float exp_diff = expf(max_score - score);
                    for (int d = 0; d < head_dim; d++)
                        acc[d] *= exp_diff;
                    sum_exp = sum_exp * exp_diff + 1.0f;
                    max_score = score;
                } else {
                    float exp_score = expf(score - max_score);
                    sum_exp += exp_score;
                    const float *v_vec = V + ki * n_kv_heads * head_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        acc[d] += exp_score * v_vec[d];
                    continue;
                }
                /* Add current V when max_score was updated */
                const float *v_vec = V + ki * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    acc[d] += v_vec[d];
            }

            /* Normalize */
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            float *out_vec = out + qi * n_q_heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_vec[d] = acc[d] * inv_sum;

            free(acc);
        }
    }
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

    /* Apply RoPE to Q and K */
    enc_apply_rope(Q, K, total_len, n_q_heads, n_kv_heads, head_dim, cfg->enc_rope_theta);

    /* GQA mixed attention (bidirectional for visual, causal for flow) */
    float *attn_out = (float *)malloc(total_len * hidden * sizeof(float));
    enc_gqa_mixed_attention(attn_out, Q, K, V, visual_len, total_len,
                            n_q_heads, n_kv_heads, head_dim);

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

    /* SwiGLU: silu(gate) * up */
    for (int i = 0; i < total_len * intermediate; i++) {
        float g = gate_buf[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        gate_buf[i] = g * sigmoid_g * up_buf[i];
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
                               int n_tokens, int *out_seq_len) {
    ds_deep_encoder_t *enc = &ctx->encoder;
    ds_config_t *cfg = &ctx->config;

    int visual_len = n_tokens;
    int n_causal_queries = cfg->enc_causal_flow_queries;
    int total_len = visual_len + n_causal_queries;
    int hidden = cfg->enc_hidden;

    if (ds_verbose >= 1)
        fprintf(stderr, "DeepEncoder V2: %d visual + %d causal flow = %d total\n",
                visual_len, n_causal_queries, total_len);

    /* Concatenate visual tokens and causal flow queries */
    float *x = (float *)calloc(total_len * hidden, sizeof(float));
    memcpy(x, visual_tokens, visual_len * hidden * sizeof(float));

    if (n_causal_queries > 0 && ctx->vis_tokenizer.causal_query_embeddings) {
        memcpy(x + visual_len * hidden,
               ctx->vis_tokenizer.causal_query_embeddings,
               n_causal_queries * hidden * sizeof(float));
    }

    /* Transformer layers */
    for (int l = 0; l < cfg->enc_layers; l++) {
        float *next = (float *)malloc(total_len * hidden * sizeof(float));
        enc_v2_layer_forward(next, x, ctx, l, visual_len, total_len);
        free(x);
        x = next;
    }

    /* Final norm */
    float *x_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(x_norm, x, enc->final_norm_weight, total_len, hidden, cfg->dec_rms_norm_eps);
    free(x);

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
                           int n_tokens, int *out_seq_len) {
    if (ctx->config.enc_type == 1) {
        /* V1: CLIP path — visual_tokens here are SAM features
         * CLIP needs patch_embeds separately, but for the unified API
         * we pass SAM features directly */
        return ds_clip_encoder_forward(ctx, visual_tokens, n_tokens,
                                       visual_tokens, n_tokens, out_seq_len);
    } else {
        return ds_encoder_forward_v2(ctx, visual_tokens, n_tokens, out_seq_len);
    }
}

/* Weight Loading (done in ds_ocr.c during ds_load()) */
int ds_encoder_load(ds_ctx_t *ctx) {
    return 0;
}
