/*
 * ds_visual_tokenizer.c - SAM Vision Tokenizer for DeepSeek-OCR
 *
 * Architecture: SAM ViT-B encoder → neck → downsample → spatial features
 * - SAM ViT-B: 12 layers, 768 dim, 12 heads, patch_size=16
 * - Window attention with relative position embeddings
 * - Global attention at layers [2, 5, 8, 11]
 * - SAM neck: 2×(Conv2d + LayerNorm2d): 768→256→256
 * - Downsample: net_2 Conv2d(256→512, k3, s2, p1) + net_3 Conv2d(512→1024, k3, s2, p1)
 */

#include "ds_visual_tokenizer.h"
#include "ds_kernels.h"
#include "ds_image.h"
#include "ds_safetensors.h"
#include "ds_ocr.h"
#include "ds_dump.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* Set by ds_visual_tokenizer_forward based on model version */
static int g_sam_is_v2 = 0;

/* ========================================================================
 * SAM LayerNorm2d (Channel-wise LayerNorm for spatial features)
 * ======================================================================== */

static void layer_norm_2d(float *out, const float *in,
                           const float *weight, const float *bias,
                           int channels, int spatial) {
    /* in/out: [channels, spatial], normalize across channels for each spatial position */
    for (int s = 0; s < spatial; s++) {
        float mean = 0.0f;
        for (int c = 0; c < channels; c++) mean += in[c * spatial + s];
        mean /= channels;

        float var = 0.0f;
        for (int c = 0; c < channels; c++) {
            float d = in[c * spatial + s] - mean;
            var += d * d;
        }
        var /= channels;
        float inv_std = 1.0f / sqrtf(var + 1e-6f);

        for (int c = 0; c < channels; c++) {
            float val = (in[c * spatial + s] - mean) * inv_std;
            out[c * spatial + s] = val * weight[c] + (bias ? bias[c] : 0.0f);
        }
    }
}

/* ========================================================================
 * SAM Relative Position Embedding
 * ======================================================================== */

/* Get relative position bias from rel_pos embeddings
 * V1: rel_pos shape [n_heads, head_dim, 2*window_size-1]
 * V2: rel_pos shape [2*window_size-1, head_dim] (shared across heads)
 * q_coords, k_coords: coordinates within the window
 * Returns bias value for this head and (q,k) pair
 */
static float get_rel_pos_v2(const float *rel_pos, int head_dim,
                             int q_coord, int k_coord, int rel_size) {
    int rel_offset = q_coord - k_coord + rel_size - 1;
    if (rel_offset < 0 || rel_offset >= 2 * rel_size - 1) return 0.0f;
    /* V2: shape [2*rel_size-1, head_dim] */
    const float *rp = rel_pos + rel_offset * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) sum += rp[d];
    return sum;
}

static float get_rel_pos_v1(const float *rel_pos, int n_heads, int head_dim,
                             int q_coord, int k_coord, int rel_size, int head_idx) {
    int rel_offset = q_coord - k_coord + rel_size - 1;
    if (rel_offset < 0 || rel_offset >= 2 * rel_size - 1) return 0.0f;
    /* V1: shape [n_heads, head_dim, 2*rel_size-1] */
    const float *rp = rel_pos + (size_t)head_idx * head_dim * (2 * rel_size - 1) + rel_offset;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) sum += rp[d * (2 * rel_size - 1)];
    return sum;
}

/* Add relative position embedding to attention scores
 * V1: rel_pos_h/w shape [n_heads, head_dim, 2*window_h/w-1]
 * V2: rel_pos_h/w shape [2*window_h/w-1, head_dim] (shared across heads)
 */
static void add_rel_pos_bias(float *attn_scores, int n_heads,
                              int q_h, int q_w, int k_h, int k_w,
                              const float *rel_pos_h, const float *rel_pos_w,
                              int head_dim, int is_v2) {
    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < q_h * q_w; qi++) {
            int qy = qi / q_w;
            int qx = qi % q_w;
            for (int ki = 0; ki < k_h * k_w; ki++) {
                int ky = ki / k_w;
                int kx = ki % k_w;
                float bias;
                if (is_v2) {
                    bias = get_rel_pos_v2(rel_pos_h, head_dim, qy, ky, q_h)
                         + get_rel_pos_v2(rel_pos_w, head_dim, qx, kx, q_w);
                } else {
                    bias = get_rel_pos_v1(rel_pos_h, n_heads, head_dim, qy, ky, q_h, h)
                         + get_rel_pos_v1(rel_pos_w, n_heads, head_dim, qx, kx, q_w, h);
                }
                attn_scores[(size_t)h * q_h * q_w * k_h * k_w + qi * k_h * k_w + ki] += bias;
            }
        }
    }
}

/* ========================================================================
 * SAM Window Attention
 * ======================================================================== */

/* Window partition with padding: [seq_h, seq_w, dim] -> [n_windows, window_size*window_size, dim]
 * Pads if seq_h/seq_w not divisible by win_size.
 * Returns: [n_windows, win_size*win_size, dim] (caller must free)
 * Also sets *n_windows_out and *pad_h_out, *pad_w_out
 */
static float *window_partition_pad(const float *x, int seq_h, int seq_w,
                                    int win_size, int dim, int *n_windows_out,
                                    int *padded_h, int *padded_w) {
    int pad_h = (win_size - seq_h % win_size) % win_size;
    int pad_w = (win_size - seq_w % win_size) % win_size;
    int Hp = seq_h + pad_h;
    int Wp = seq_w + pad_w;
    *padded_h = Hp;
    *padded_w = Wp;

    /* Pad input if needed (zero padding) */
    float *padded = NULL;
    if (pad_h > 0 || pad_w > 0) {
        padded = (float *)calloc((size_t)Hp * Wp * dim, sizeof(float));
        for (int y = 0; y < seq_h; y++) {
            memcpy(padded + y * Wp * dim, x + y * seq_w * dim, seq_w * dim * sizeof(float));
        }
    }
    const float *src = padded ? padded : x;
    int src_w = padded ? Wp : seq_w;

    int n_h = Hp / win_size;
    int n_w = Wp / win_size;
    int n_windows = n_h * n_w;
    int win_tokens = win_size * win_size;

    float *out = (float *)malloc((size_t)n_windows * win_tokens * dim * sizeof(float));

    for (int nh = 0; nh < n_h; nh++) {
        for (int nw = 0; nw < n_w; nw++) {
            int win_idx = nh * n_w + nw;
            for (int wh = 0; wh < win_size; wh++) {
                for (int ww = 0; ww < win_size; ww++) {
                    int src_row = (nh * win_size + wh) * src_w + (nw * win_size + ww);
                    int dst_row = win_idx * win_tokens + wh * win_size + ww;
                    memcpy(out + dst_row * dim, src + src_row * dim, dim * sizeof(float));
                }
            }
        }
    }

    if (padded) free(padded);
    *n_windows_out = n_windows;
    return out;
}

/* Window unpartition with unpadding: reverse of window_partition_pad */
static void window_unpartition_pad(float *x, int seq_h, int seq_w,
                                    int win_size, int dim,
                                    int padded_h, int padded_w,
                                    const float *windowed) {
    int win_tokens = win_size * win_size;
    int n_h = padded_h / win_size;
    int n_w = padded_w / win_size;

    /* Unpartition to padded size, then copy only valid region */
    if (padded_h == seq_h && padded_w == seq_w) {
        /* No padding needed */
        for (int nh = 0; nh < n_h; nh++) {
            for (int nw = 0; nw < n_w; nw++) {
                int win_idx = nh * n_w + nw;
                for (int wh = 0; wh < win_size; wh++) {
                    for (int ww = 0; ww < win_size; ww++) {
                        int dst_row = (nh * win_size + wh) * seq_w + (nw * win_size + ww);
                        int src_row = win_idx * win_tokens + wh * win_size + ww;
                        memcpy(x + dst_row * dim, windowed + src_row * dim, dim * sizeof(float));
                    }
                }
            }
        }
    } else {
        /* Unpartition to padded buffer, then extract valid region */
        float *padded_out = (float *)malloc((size_t)padded_h * padded_w * dim * sizeof(float));
        for (int nh = 0; nh < n_h; nh++) {
            for (int nw = 0; nw < n_w; nw++) {
                int win_idx = nh * n_w + nw;
                for (int wh = 0; wh < win_size; wh++) {
                    for (int ww = 0; ww < win_size; ww++) {
                        int dst_row = (nh * win_size + wh) * padded_w + (nw * win_size + ww);
                        int src_row = win_idx * win_tokens + wh * win_size + ww;
                        memcpy(padded_out + dst_row * dim, windowed + src_row * dim, dim * sizeof(float));
                    }
                }
            }
        }
        /* Extract original region [seq_h, seq_w] from padded [padded_h, padded_w] */
        for (int y = 0; y < seq_h; y++) {
            memcpy(x + y * seq_w * dim, padded_out + y * padded_w * dim, seq_w * dim * sizeof(float));
        }
        free(padded_out);
    }
}

/* Forward attention within a single window (bidirectional, with rel pos) */
static void window_attn_forward(float *out, const float *Q, const float *K, const float *V,
                                 int n_tokens, int n_heads, int head_dim,
                                 const float *rel_pos_h, const float *rel_pos_w,
                                 int win_h, int win_w) {
    int hidden = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < n_tokens; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            int qy = i / win_w;
            int qx = i % win_w;

            for (int j = 0; j < n_tokens; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_row[d] * k_row[d];
                score *= scale;

                /* Add relative position bias: dot(q, Rh[offset]) + dot(q, Rw[offset]) */
                int ky = j / win_w;
                int kx = j % win_w;

                int rh_offset = qy - ky + win_h - 1;
                int rw_offset = qx - kx + win_w - 1;

                if (rh_offset >= 0 && rh_offset < 2 * win_h - 1) {
                    const float *rh_vec = rel_pos_h + rh_offset * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        score += q_row[d] * rh_vec[d];
                }
                if (rw_offset >= 0 && rw_offset < 2 * win_w - 1) {
                    const float *rw_vec = rel_pos_w + rw_offset * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        score += q_row[d] * rw_vec[d];
                }

                /* Online softmax */
                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] = o_row[d] * correction + v_row[d];
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] += wt * v_row[d];
                }
            }

            if (sum_exp > 0.0f) {
                float inv = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) o_row[d] *= inv;
            }
        }
    }
}

/* ========================================================================
 * SAM ViT-B Layer Forward Pass
 * ======================================================================== */

static void sam_layer_forward(float *out, const float *x,
                               const ds_visual_tokenizer_t *vt,
                               int layer_idx, int seq_h, int seq_w) {
    int dim = DS_SAM_EMBED_DIM;     /* 768 */
    int n_heads = DS_SAM_HEADS;     /* 12 */
    int head_dim = DS_SAM_HEAD_DIM; /* 64 */
    int mlp_dim = DS_SAM_MLP_DIM;   /* 3072 */
    int seq_len = seq_h * seq_w;
    int win_size = DS_SAM_WINDOW_SIZE; /* 14 */

    /* Global attention at layers 2, 5, 8, 11 */
    int use_global_attn = (layer_idx == 2 || layer_idx == 5 || layer_idx == 8 || layer_idx == 11);

    const float *norm1_w = vt->sam_layers[layer_idx].norm1_weight;
    const float *norm1_b = vt->sam_layers[layer_idx].norm1_bias;
    const float *qkv_w = vt->sam_layers[layer_idx].attn_qkv_weight;
    const float *qkv_b = vt->sam_layers[layer_idx].attn_qkv_bias;
    const float *proj_w = vt->sam_layers[layer_idx].attn_proj_weight;
    const float *proj_b = vt->sam_layers[layer_idx].attn_proj_bias;
    const float *rel_h = vt->sam_layers[layer_idx].rel_pos_h;
    const float *rel_w = vt->sam_layers[layer_idx].rel_pos_w;
    const float *norm2_w = vt->sam_layers[layer_idx].norm2_weight;
    const float *norm2_b = vt->sam_layers[layer_idx].norm2_bias;
    const float *lin1_w = vt->sam_layers[layer_idx].mlp_lin1_weight;
    const float *lin1_b = vt->sam_layers[layer_idx].mlp_lin1_bias;
    const float *lin2_w = vt->sam_layers[layer_idx].mlp_lin2_weight;
    const float *lin2_b = vt->sam_layers[layer_idx].mlp_lin2_bias;

    /* Pre-norm (LayerNorm1) */
    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    ds_layer_norm(x_norm, x, norm1_w, norm1_b, seq_len, dim, 1e-6f);

    if (layer_idx == 11) {
        /* C layout: [H*W, D] = [4096, 768], Python: [1, H, W, D] = [1, 64, 64, 768] */
        ds_dump_tensor("sam_block11_after_norm1", x_norm, seq_len * dim, "[4096, 768]");
    }

    /* Fused QKV projection: [seq, dim] @ [dim, 3*dim] → Q, K, V */
    float *Q = (float *)malloc(seq_len * dim * sizeof(float));
    float *K = (float *)malloc(seq_len * dim * sizeof(float));
    float *V = (float *)malloc(seq_len * dim * sizeof(float));

    /* Fused QKV: qkv_w is [3*dim, dim], qkv_b is [3*dim] */
    float *qkv = (float *)malloc(seq_len * 3 * dim * sizeof(float));
    ds_linear(qkv, x_norm, qkv_w, qkv_b, seq_len, dim, 3 * dim);

    /* Split QKV */
    for (int s = 0; s < seq_len; s++) {
        memcpy(Q + s * dim, qkv + s * 3 * dim, dim * sizeof(float));
        memcpy(K + s * dim, qkv + s * 3 * dim + dim, dim * sizeof(float));
        memcpy(V + s * dim, qkv + s * 3 * dim + 2 * dim, dim * sizeof(float));
    }

    if (layer_idx == 11) {
        ds_dump_tensor("sam_block11_Q", Q, seq_len * dim, "[4096, 768]");
        ds_dump_tensor("sam_block11_K", K, seq_len * dim, "[4096, 768]");
        ds_dump_tensor("sam_block11_V", V, seq_len * dim, "[4096, 768]");
    }

    free(qkv);

    /* For global attention blocks (2,5,8,11), rel_pos is stored for
     * the original 1024x1024 input (size=127). For 768x768 input,
     * we need size=95 (2*48-1). Python's get_rel_pos() interpolates
     * rel_pos from 127→95 using F.interpolate(mode='linear') and
     * scales coordinates. We must do the same here.
     *
     * Window attention blocks already have rel_pos size=27 (2*14-1),
     * which doesn't depend on input size, so no interpolation needed.
     */
    const float *eff_rel_h = rel_h;
    const float *eff_rel_w = rel_w;
    float *interp_rel_h = NULL;
    float *interp_rel_w = NULL;

    if (use_global_attn) {
        int stored_size = 2 * (DS_IMAGE_SIZE / DS_SAM_PATCH_SIZE) - 1;  /* 2*64-1=127 */
        int needed_size = 2 * seq_h - 1;  /* 2*48-1=95 for 768x768 */

        if (needed_size != stored_size) {
            /* Linear interpolation of rel_pos from stored_size → needed_size.
             * Matches Python: F.interpolate(rel_pos.reshape(1,L,-1).permute(0,2,1),
             *                                size=needed_size, mode='linear')
             * rel_pos shape: [stored_size, head_dim] → interpolate each dim independently */
            interp_rel_h = (float *)malloc(needed_size * head_dim * sizeof(float));
            interp_rel_w = (float *)malloc(needed_size * head_dim * sizeof(float));

            for (int d = 0; d < head_dim; d++) {
                for (int i = 0; i < needed_size; i++) {
                    /* PyTorch align_corners=False linear interpolation:
                     * src = (dst + 0.5) * src_size / dst_size - 0.5 */
                    float src_f = (float)(i + 0.5f) * (float)stored_size / (float)needed_size - 0.5f;
                    if (src_f < 0.0f) src_f = 0.0f;
                    if (src_f > stored_size - 1.0f) src_f = (float)(stored_size - 1);
                    int s0 = (int)src_f;
                    int s1 = s0 + 1;
                    if (s1 >= stored_size) s1 = stored_size - 1;
                    float frac = src_f - s0;
                    interp_rel_h[i * head_dim + d] = rel_h[s0 * head_dim + d] * (1.0f - frac)
                                                     + rel_h[s1 * head_dim + d] * frac;
                    interp_rel_w[i * head_dim + d] = rel_w[s0 * head_dim + d] * (1.0f - frac)
                                                     + rel_w[s1 * head_dim + d] * frac;
                }
            }
            eff_rel_h = interp_rel_h;
            eff_rel_w = interp_rel_w;
        }
    }

    /* Attention */
    float *attn_out = (float *)calloc(seq_len * dim, sizeof(float));

    if (use_global_attn) {
        /* Global attention with float64 softmax for numerical stability.
         * SAM global attn score range can exceed 500 (head 7: -493 to 68),
         * causing severe float32 precision loss in softmax.
         * We compute scores and softmax in double, then convert back.
         */
        double scale_d = 1.0 / sqrt((double)head_dim);

        for (int h = 0; h < n_heads; h++) {
            /* Allocate per-query buffers */
            double *score_buf = (double *)malloc(seq_len * sizeof(double));

            for (int i = 0; i < seq_len; i++) {
                int qy = i / seq_w, qx = i % seq_w;
                const float *q_vec = Q + i * dim + h * head_dim;

                /* Pass 1: compute scores in double, find max */
                double max_s = -1e30;
                for (int j = 0; j < seq_len; j++) {
                    int ky = j / seq_w, kx = j % seq_w;
                    const float *k_vec = K + j * dim + h * head_dim;
                    double s = 0.0;
                    for (int d = 0; d < head_dim; d++)
                        s += (double)q_vec[d] * (double)k_vec[d];
                    s *= scale_d;

                    /* Rel pos bias in double (using interpolated rel_pos for global attn) */
                    int rh_off = qy - ky + seq_h - 1;
                    int rw_off = qx - kx + seq_w - 1;
                    if (rh_off >= 0 && rh_off < 2 * seq_h - 1) {
                        const float *rh_vec = eff_rel_h + rh_off * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            s += (double)q_vec[d] * (double)rh_vec[d];
                    }
                    if (rw_off >= 0 && rw_off < 2 * seq_w - 1) {
                        const float *rw_vec = eff_rel_w + rw_off * head_dim;
                        for (int d = 0; d < head_dim; d++)
                            s += (double)q_vec[d] * (double)rw_vec[d];
                    }
                    score_buf[j] = s;
                    if (s > max_s) max_s = s;
                }

                /* Pass 2: exp(score - max), weighted sum, normalize in double */
                double sum_e = 0.0;
                float *o_row = attn_out + i * dim + h * head_dim;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = 0; j < seq_len; j++) {
                    double w = exp(score_buf[j] - max_s);
                    sum_e += w;
                    const float *v_row = V + j * dim + h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] += (float)(w * (double)v_row[d]);
                }

                if (sum_e > 0.0) {
                    float inv = (float)(1.0 / sum_e);
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] *= inv;
                }
            }

            free(score_buf);
        }
    } else {
        /* Window attention: partition into windows (with padding), attend within each */
        int n_windows, padded_h, padded_w;
        float *win_Q = window_partition_pad(Q, seq_h, seq_w, win_size, dim, &n_windows, &padded_h, &padded_w);
        float *win_K = window_partition_pad(K, seq_h, seq_w, win_size, dim, &n_windows, &padded_h, &padded_w);
        float *win_V = window_partition_pad(V, seq_h, seq_w, win_size, dim, &n_windows, &padded_h, &padded_w);

        int win_tokens = win_size * win_size;
        float *win_attn = (float *)calloc((size_t)n_windows * win_tokens * dim, sizeof(float));

        for (int w = 0; w < n_windows; w++) {
            float *wq = win_Q + (size_t)w * win_tokens * dim;
            float *wk = win_K + (size_t)w * win_tokens * dim;
            float *wv = win_V + (size_t)w * win_tokens * dim;
            float *wout = win_attn + (size_t)w * win_tokens * dim;

            window_attn_forward(wout, wq, wk, wv, win_tokens, n_heads, head_dim,
                                rel_h, rel_w, win_size, win_size);
        }

        /* Unpartition back to full sequence (removes padding) */
        window_unpartition_pad(attn_out, seq_h, seq_w, win_size, dim, padded_h, padded_w, win_attn);

        free(win_Q); free(win_K); free(win_V); free(win_attn);
    }

    free(Q); free(K); free(V);

    /* Output projection + residual */
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(proj_out, attn_out, proj_w, proj_b, seq_len, dim, dim);
    free(attn_out);
    if (interp_rel_h) free(interp_rel_h);
    if (interp_rel_w) free(interp_rel_w);

    for (int i = 0; i < seq_len * dim; i++) out[i] = x[i] + proj_out[i];

    if (layer_idx == 11) {
        /* Dump attn output after proj (before residual = proj_out, after residual = out) */
        ds_dump_tensor("sam_block11_attn_proj_out", proj_out, seq_len * dim, "[4096, 768]");
        ds_dump_tensor("sam_block11_after_attn_resid", out, seq_len * dim, "[4096, 768]");
    }

    free(proj_out);

    /* FFN: Pre-norm (LayerNorm2) + GELU MLP */
    float *ffn_norm = (float *)malloc(seq_len * dim * sizeof(float));
    ds_layer_norm(ffn_norm, out, norm2_w, norm2_b, seq_len, dim, 1e-6f);

    float *fc1 = (float *)malloc(seq_len * mlp_dim * sizeof(float));
    ds_linear(fc1, ffn_norm, lin1_w, lin1_b, seq_len, dim, mlp_dim);
    ds_gelu(fc1, seq_len * mlp_dim);

    float *fc2 = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(fc2, fc1, lin2_w, lin2_b, seq_len, mlp_dim, dim);

    if (layer_idx == 11) {
        ds_dump_tensor("sam_block11_mlp_out", fc2, seq_len * dim, "[4096, 768]");
    }

    for (int i = 0; i < seq_len * dim; i++) out[i] += fc2[i];

    free(x_norm); free(ffn_norm); free(fc1); free(fc2);
}

/* ========================================================================
 * SAM Neck + Downsample
 * ======================================================================== */

/* Apply SAM neck: 2×(Conv2d 1×1 + LayerNorm2d)
 * Input: [768, H, W] spatial feature map
 * Output: [256, H, W] spatial feature map
 */
static float *sam_neck_forward(const float *spatial, int h, int w,
                                const ds_visual_tokenizer_t *vt) {
    int c_in = DS_SAM_EMBED_DIM; /* 768 */
    int c_mid = DS_SAM_NECK_DIM; /* 256 */
    int spatial_sz = h * w;

    /* Neck conv1: [768, h, w] → [256, h, w] (1×1 conv, no padding) */
    float *conv1_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    ds_conv2d(conv1_out, spatial, vt->sam_neck_conv1_weight, vt->sam_neck_conv1_bias,
              c_in, c_mid, h, w, 1, 1, 1, 0);

    /* LayerNorm2d after conv1 */
    float *ln1_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    layer_norm_2d(ln1_out, conv1_out, vt->sam_neck_ln1_weight, vt->sam_neck_ln1_bias,
                  c_mid, spatial_sz);
    free(conv1_out);

    /* Neck conv2: [256, h, w] → [256, h, w]
     * V1: 1×1 conv, no padding
     * V2: 3×3 conv, stride=1, padding=1 */
    float *conv2_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    if (g_sam_is_v2) {
        ds_conv2d(conv2_out, ln1_out, vt->sam_neck_conv2_weight, vt->sam_neck_conv2_bias,
                  c_mid, c_mid, h, w, 3, 3, 1, 1);
    } else {
        ds_conv2d(conv2_out, ln1_out, vt->sam_neck_conv2_weight, vt->sam_neck_conv2_bias,
                  c_mid, c_mid, h, w, 1, 1, 1, 0);
    }

    /* LayerNorm2d after conv2 */
    float *ln2_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    layer_norm_2d(ln2_out, conv2_out, vt->sam_neck_ln2_weight, vt->sam_neck_ln2_bias,
                  c_mid, spatial_sz);
    free(conv2_out); free(ln1_out);

    return ln2_out;
}

/* Downsample via net_2 and net_3
 * Input: [256, H, W] → net_2 → [512, H/2, W/2] → net_3 → [ds2_dim, H/4, W/4]
 * Each: Conv2d(k=3, s=2, p=1)
 */
static float *sam_downsample_forward(const float *spatial, int h, int w,
                                      const ds_visual_tokenizer_t *vt,
                                      int ds2_dim,
                                      int *out_h, int *out_w) {
    /* net_2: Conv2d(256→512, k3, s2, p1) */
    int h2 = (h + 2 * 1 - 3) / 2 + 1; /* = h/2 */
    int w2 = (w + 2 * 1 - 3) / 2 + 1;
    int sp2 = h2 * w2;

    float *net2_out = (float *)malloc(DS_SAM_DS1_DIM * sp2 * sizeof(float));
    ds_conv2d(net2_out, spatial, vt->sam_net2_weight, vt->sam_net2_bias,
              DS_SAM_NECK_DIM, DS_SAM_DS1_DIM, h, w, 3, 3, 2, 1);

    /* net_3: Conv2d(512→ds2_dim, k3, s2, p1) */
    int h3 = (h2 + 2 * 1 - 3) / 2 + 1; /* = h/4 */
    int w3 = (w2 + 2 * 1 - 3) / 2 + 1;
    int sp3 = h3 * w3;

    float *net3_out = (float *)malloc(ds2_dim * sp3 * sizeof(float));
    ds_conv2d(net3_out, net2_out, vt->sam_net3_weight, vt->sam_net3_bias,
              DS_SAM_DS1_DIM, ds2_dim, h2, w2, 3, 3, 2, 1);
    free(net2_out);

    *out_h = h3;
    *out_w = w3;
    return net3_out;
}

/* ========================================================================
 * Bicubic Interpolation for Position Embeddings
 * Matches Python: F.interpolate(mode='bicubic', align_corners=False, antialias=True)
 * Input:  pos_embed [1, src_h, src_w, embed_dim]  (NHWC layout)
 * Output: [1, tgt_h, tgt_w, embed_dim]
 * ======================================================================== */

/* Cubic interpolation kernel (used by bicubic)
 * Matches scipy/PIL bicubic with a=-0.75 (not a=-1 as in PyTorch default)
 * But F.interpolate with antialias=True uses a=-0.75, while without it uses a=-1.
 * Python code: F.interpolate(mode='bicubic', antialias=True) → uses a=-0.75
 */
static float bicubic_weight(float x) {
    const float a = -0.75f;
    float ax = fabsf(x);
    if (ax <= 1.0f) {
        return (a + 2.0f) * ax * ax * ax - (a + 3.0f) * ax * ax + 1.0f;
    } else if (ax < 2.0f) {
        return a * ax * ax * ax - 5.0f * a * ax * ax + 8.0f * a * ax - 4.0f * a;
    }
    return 0.0f;
}

/* Interpolate a 2D grid [src_h, src_w, embed_dim] to [tgt_h, tgt_w, embed_dim]
 * Matches Python: F.interpolate(mode='bicubic', align_corners=False, antialias=True)
 *
 * For downsampling (scale < 1), antialias=True widens the kernel:
 *   support_radius = 2 / scale (instead of fixed 2)
 *   This requires a larger neighborhood (6x6 or more instead of 4x4).
 *
 * Coordinate mapping (PyTorch align_corners=False):
 *   src = (dst + 0.5) * src_size / dst_size - 0.5
 *
 * Returns newly allocated float array of size tgt_h * tgt_w * embed_dim.
 */
static float *bicubic_interpolate_2d(const float *src, int src_h, int src_w,
                                       int tgt_h, int tgt_w, int embed_dim) {
    float *dst = (float *)malloc(tgt_h * tgt_w * embed_dim * sizeof(float));

    /* Compute scale factors and kernel support for antialiasing.
     * For downsampling (scale < 1), the filter is widened by 1/scale.
     * For upsampling (scale >= 1), the standard 4x4 kernel is used. */
    float y_scale = (float)src_h / (float)tgt_h;
    float x_scale = (float)src_w / (float)tgt_w;
    int y_support = (int)ceilf(2.0f / fminf(y_scale, 1.0f));
    int x_support = (int)ceilf(2.0f / fminf(x_scale, 1.0f));

    for (int ty = 0; ty < tgt_h; ty++) {
        float sy = (float)src_h * ((float)ty + 0.5f) / (float)tgt_h - 0.5f;
        int sy0 = (int)floorf(sy);

        for (int tx = 0; tx < tgt_w; tx++) {
            float sx = (float)src_w * ((float)tx + 0.5f) / (float)tgt_w - 0.5f;
            int sx0 = (int)floorf(sx);

            for (int d = 0; d < embed_dim; d++) {
                float val = 0.0f;
                float w_sum = 0.0f;

                for (int ky = -y_support; ky <= y_support; ky++) {
                    int y_clamped = sy0 + ky;
                    if (y_clamped < 0) y_clamped = 0;
                    if (y_clamped >= src_h) y_clamped = src_h - 1;
                    float wy = bicubic_weight((float)ky - (sy - sy0) * fminf(y_scale, 1.0f));

                    for (int kx = -x_support; kx <= x_support; kx++) {
                        int x_clamped = sx0 + kx;
                        if (x_clamped < 0) x_clamped = 0;
                        if (x_clamped >= src_w) x_clamped = src_w - 1;
                        float wx = bicubic_weight((float)kx - (sx - sx0) * fminf(x_scale, 1.0f));

                        float w = wx * wy;
                        val += w * src[(y_clamped * src_w + x_clamped) * embed_dim + d];
                        w_sum += w;
                    }
                }
                /* Normalize to prevent drift from boundary clamping */
                dst[(ty * tgt_w + tx) * embed_dim + d] = (w_sum > 0.0f) ? val / w_sum : 0.0f;
            }
        }
    }
    return dst;
}

/* Interpolate pos_embed from src_grid to tgt_grid (V2: [grid_h, grid_w, 768])
 * If src_grid == tgt_grid, returns a copy of the original.
 * This matches Python's get_abs_pos_sam().
 */
static float *interpolate_pos_embed(const float *pos_embed, int src_grid,
                                     int tgt_grid, int embed_dim) {
    if (src_grid == tgt_grid) {
        float *out = (float *)malloc(src_grid * src_grid * embed_dim * sizeof(float));
        memcpy(out, pos_embed, src_grid * src_grid * embed_dim * sizeof(float));
        return out;
    }
    if (ds_verbose >= 1)
        fprintf(stderr, "Interpolating pos_embed: %dx%d -> %dx%d (bicubic)\n",
                src_grid, src_grid, tgt_grid, tgt_grid);
    return bicubic_interpolate_2d(pos_embed, src_grid, src_grid,
                                   tgt_grid, tgt_grid, embed_dim);
}

/* ========================================================================
 * Patch Embedding
 * ======================================================================== */

static float *sam_patch_embed(const float *image_chw, int img_h, int img_w,
                               const float *weight, const float *bias,
                               int *out_h, int *out_w) {
    int patch_size = DS_SAM_PATCH_SIZE;
    int embed_dim = DS_SAM_EMBED_DIM;
    *out_h = img_h / patch_size;
    *out_w = img_w / patch_size;

    int n_patches = (*out_h) * (*out_w);
    float *patches = (float *)malloc(n_patches * embed_dim * sizeof(float));

    /* Conv2D: weight [768, 3, 16, 16], stride 16, no padding */
    ds_conv2d(patches, image_chw, weight, bias, 3, embed_dim, img_h, img_w,
              16, 16, 16, 0);

    if (ds_dump_enabled()) {
        /* Detailed patch_embed diagnostics */
        /* 1. Per-channel unique value count for first few channels */
        for (int ch = 0; ch < 3; ch++) {
            int unique = 1;
            float *ch_data = patches + ch * n_patches;
            /* Simple unique count: sort and count transitions */
            float *tmp = (float *)malloc(n_patches * sizeof(float));
            memcpy(tmp, ch_data, n_patches * sizeof(float));
            /* Count unique by naive O(n^2) — n_patches=4096, cheap enough */
            unique = 0;
            for (int i = 0; i < n_patches; i++) {
                int is_new = 1;
                for (int j = 0; j < i; j++) {
                    if (tmp[j] == tmp[i]) { is_new = 0; break; }
                }
                if (is_new) unique++;
            }
            free(tmp);
            fprintf(stderr, "[DUMP] patch_embed ch%d: unique=%d first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    ch, unique,
                    ch_data[0], ch_data[1], ch_data[2], ch_data[3], ch_data[4]);
        }

        /* 2. Verify first output patch manually: output[oc, 0] = sum(weight[oc,ic]*input[ic,0:16,0:16]) + bias[oc] */
        /* Input patch: image_chw[c, 0:16, 0:16] for c=0..2 */
        float *input_patch = (float *)malloc(3 * 16 * 16 * sizeof(float));
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                    input_patch[c * 256 + y * 16 + x] = image_chw[c * img_h * img_w + y * img_w + x];
        ds_dump_tensor("input_patch_0_0", input_patch, 3 * 16 * 16, "[3, 16, 16]");

        /* Manual conv for channel 0: dot(weight[0], input_patch) + bias[0] */
        /* weight layout: [768, 3*16*16] = [out_ch, in_ch * kH * kW] */
        float manual_out = bias[0];
        int patch_size_sq = 3 * 16 * 16;
        for (int p = 0; p < patch_size_sq; p++) {
            manual_out += weight[0 * patch_size_sq + p] * input_patch[p];
        }
        fprintf(stderr, "[DUMP] patch_embed manual ch0 patch(0,0) = %.6f, conv2d output = %.6f\n",
                manual_out, patches[0]);

        /* Dump weight and bias stats */
        float w_sum = 0, w_abs_sum = 0;
        for (int i = 0; i < embed_dim * 3 * 16 * 16; i++) {
            w_sum += weight[i];
            w_abs_sum += fabsf(weight[i]);
        }
        fprintf(stderr, "[DUMP] patch_embed weight: n=%d mean=%.6f abs_mean=%.6f\n",
                embed_dim * 3 * 16 * 16, w_sum / (embed_dim * 3 * 16 * 16),
                w_abs_sum / (embed_dim * 3 * 16 * 16));
        float b_sum = 0;
        for (int i = 0; i < embed_dim; i++) b_sum += bias[i];
        fprintf(stderr, "[DUMP] patch_embed bias: n=%d mean=%.6f first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                embed_dim, b_sum / embed_dim,
                bias[0], bias[1], bias[2], bias[3], bias[4]);

        free(input_patch);
    }

    /* Also return the patch embeds for CLIP input (before adding pos embed) */
    /* patches is in [768, n_patches] format (CHW) */

    return patches;
}

/* ========================================================================
 * Main Forward Pass
 * ======================================================================== */

/* ========================================================================
 * Core SAM Forward (from float CHW image, variable input size)
 *
 * Takes a pre-processed float CHW image [3, img_h, img_w] and runs the
 * full SAM pipeline: patch_embed -> pos_embed (with interpolation) ->
 * 12 transformer layers -> neck -> downsample.
 *
 * Returns: [n_tokens, ds2_dim] (caller must free).
 * *out_n_tokens = number of output tokens.
 * *out_patch_embeds (optional): copy of patch_embed output in [768, n_patches] CHW.
 * ======================================================================== */
static float *sam_forward_from_chw(ds_ctx_t *ctx, const float *image_chw,
                                    int img_h, int img_w,
                                    int *out_n_tokens, float **out_patch_embeds) {
    ds_visual_tokenizer_t *vt = &ctx->vis_tokenizer;
    ds_config_t *cfg = &ctx->config;

    /* Patch embedding */
    int grid_h, grid_w;
    float *patch_spatial = sam_patch_embed(image_chw, img_h, img_w,
                                            vt->sam_patch_embed_weight,
                                            vt->sam_patch_embed_bias,
                                            &grid_h, &grid_w);
    int n_patches = grid_h * grid_w;

    /* Save patch_embeds for CLIP input (in [768, n_patches] format) */
    if (out_patch_embeds) {
        *out_patch_embeds = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));
        memcpy(*out_patch_embeds, patch_spatial, n_patches * DS_SAM_EMBED_DIM * sizeof(float));
    }

    /* Debug: dump patch_embed and input image for comparison */
    if (getenv("DS_DUMP_PATCH_EMBED")) {
        FILE *df = fopen("dump/c_sam_patch_embed.bin", "wb");
        if (df) { fwrite(patch_spatial, sizeof(float), n_patches * DS_SAM_EMBED_DIM, df); fclose(df); }
        df = fopen("dump/c_sam_input_image.bin", "wb");
        if (df) { fwrite(image_chw, sizeof(float), 3 * img_h * img_w, df); fclose(df); }
        fprintf(stderr, "Dumped patch_embed (%d patches x %d dim) and input image (%dx%d)\n",
                n_patches, DS_SAM_EMBED_DIM, img_w, img_h);
    }

    /* Add positional embedding (with interpolation for non-1024 inputs) */
    float *x = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));

    if (cfg->model_version == 2) {
        /* V2: pos_embed is 2D spatial [1, src_grid, src_grid, 768]
         * src_grid = image_size / patch_size = 1024/16 = 64
         * If input is 768x768, grid = 48, need bicubic interpolation 64->48 */
        int src_grid = cfg->image_size / DS_SAM_PATCH_SIZE;  /* 64 */
        int tgt_grid = grid_h;  /* may differ from src_grid for 768x768 inputs */

        float *pos_embed = vt->sam_pos_embed;
        float *interp_pos = NULL;

        if (tgt_grid != src_grid) {
            /* Check for precomputed pos_embed override (avoids C interpolation mismatch) */
            const char *pe_file = getenv("DS_SAM_POSEMBED_FILE");
            if (pe_file) {
                FILE *pf = fopen(pe_file, "rb");
                if (pf) {
                    interp_pos = (float *)malloc(tgt_grid * tgt_grid * DS_SAM_EMBED_DIM * sizeof(float));
                    size_t nread = fread(interp_pos, sizeof(float),
                                          tgt_grid * tgt_grid * DS_SAM_EMBED_DIM, pf);
                    fclose(pf);
                    if ((int)nread == tgt_grid * tgt_grid * DS_SAM_EMBED_DIM) {
                        pos_embed = interp_pos;
                        /* Verify first values match file content */
                        fprintf(stderr, "Loaded precomputed pos_embed from %s (%dx%d), first3=[%.6f,%.6f,%.6f]\n",
                                pe_file, tgt_grid, tgt_grid,
                                interp_pos[0], interp_pos[1], interp_pos[2]);
                    } else {
                        fprintf(stderr, "Warning: DS_SAM_POSEMBED_FILE size mismatch, using interpolation\n");
                        free(interp_pos);
                        interp_pos = interpolate_pos_embed(pos_embed, src_grid, tgt_grid, DS_SAM_EMBED_DIM);
                        pos_embed = interp_pos;
                    }
                } else {
                    interp_pos = interpolate_pos_embed(pos_embed, src_grid, tgt_grid, DS_SAM_EMBED_DIM);
                    pos_embed = interp_pos;
                }
            } else {
                interp_pos = interpolate_pos_embed(pos_embed, src_grid, tgt_grid, DS_SAM_EMBED_DIM);
                pos_embed = interp_pos;
            }
        }

        for (int p = 0; p < n_patches; p++) {
            int row = p / grid_w;
            int col = p % grid_w;
            for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
                x[p * DS_SAM_EMBED_DIM + d] = patch_spatial[d * n_patches + p]
                    + pos_embed[(row * tgt_grid + col) * DS_SAM_EMBED_DIM + d];
            }
        }

        if (interp_pos) free(interp_pos);
    } else {
        /* V1: pos_embed is 1D sequence [1, n_patches+1, 768] (with CLS token at index 0) */
        for (int p = 0; p < n_patches; p++) {
            for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
                x[p * DS_SAM_EMBED_DIM + d] = patch_spatial[d * n_patches + p]
                    + vt->sam_pos_embed[(p + 1) * DS_SAM_EMBED_DIM + d];
            }
        }
    }
    free(patch_spatial);

    if (ds_verbose >= 2)
        fprintf(stderr, "SAM patch embed: %dx%d -> %d patches (grid %dx%d)\n",
                img_h, img_w, n_patches, grid_h, grid_w);

    /* SAM transformer layers (12 layers with window/global attention) */
    for (int l = 0; l < 12; l++) {
        float *next = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));
        sam_layer_forward(next, x, vt, l, grid_h, grid_w);
        
        /* Debug: dump layer output for comparison with Python */
        if (getenv("DS_DUMP_SAM_LAYERS")) {
            char _lpath[256];
            snprintf(_lpath, sizeof(_lpath), "dump/c_sam_layer%d_out.bin", l);
            FILE *df = fopen(_lpath, "wb");
            if (df) { fwrite(next, sizeof(float), n_patches * DS_SAM_EMBED_DIM, df); fclose(df); }
            /* Also dump input to SAM layer 0 (after pos_embed) */
            if (l == 0) {
                df = fopen("dump/c_sam_after_posembed.bin", "wb");
                if (df) { fwrite(x, sizeof(float), n_patches * DS_SAM_EMBED_DIM, df); fclose(df); }
            }
            fprintf(stderr, "Dumped SAM layer %d I/O for comparison\n", l);
        }
        
        free(x);
        x = next;

        if (ds_verbose >= 2)
            fprintf(stderr, "SAM layer %d/%d done (global_attn=%d)\n", l + 1, 12,
                    (l == 2 || l == 5 || l == 8 || l == 11));
    }

    /* Reshape to spatial [768, grid_h, grid_w] for neck */
    float *spatial = (float *)malloc(DS_SAM_EMBED_DIM * n_patches * sizeof(float));
    for (int p = 0; p < n_patches; p++) {
        for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
            spatial[d * n_patches + p] = x[p * DS_SAM_EMBED_DIM + d];
        }
    }
    free(x);

    /* SAM neck: [768, H, W] -> [256, H, W] */
    float *neck_out = sam_neck_forward(spatial, grid_h, grid_w, vt);
    free(spatial);

    /* SAM downsample: [256, H, W] -> [512, H/2, W/2] -> [ds2_dim, H/4, W/4] */
    int ds_h, ds_w;
    int ds2_dim = cfg->sam_ds2_dim;
    float *ds_out = sam_downsample_forward(neck_out, grid_h, grid_w, vt, ds2_dim, &ds_h, &ds_w);
    free(neck_out);

    int ds_spatial = ds_h * ds_w;

    /* Convert [ds2_dim, ds_h*ds_w] -> [ds_h*ds_w, ds2_dim] for downstream use */
    float *tokens = (float *)malloc(ds_spatial * ds2_dim * sizeof(float));
    for (int p = 0; p < ds_spatial; p++) {
        for (int d = 0; d < ds2_dim; d++) {
            tokens[p * ds2_dim + d] = ds_out[d * ds_spatial + p];
        }
    }
    free(ds_out);

    *out_n_tokens = ds_spatial;

    if (ds_verbose >= 1)
        fprintf(stderr, "SAM forward: %dx%d -> %d tokens (grid %dx%d, downsampled %dx%d)\n",
                img_h, img_w, *out_n_tokens, grid_h, grid_w, ds_h, ds_w);

    return tokens;
}

/* ========================================================================
 * Public API: SAM forward from pre-processed ds_image_t
 *
 * Takes a ds_image_t (already resized/padded/cropped to target dimensions),
 * converts to float CHW, then runs SAM forward.
 * Used for multi-crop encoding where each crop is 768x768.
 * ======================================================================== */
float *ds_sam_forward_image(ds_ctx_t *ctx, const ds_image_t *img,
                             int *out_n_tokens, float **out_patch_embeds) {
    if (!img || !img->pixels) return NULL;
    float *image_chw = ds_image_to_float_chw(img);
    if (!image_chw) return NULL;

    /* Debug: dump SAM input pixels */
    if (getenv("DS_DUMP_TENSORS")) {
        static int _dump_count = 0;
        char _dump_path[256];
        snprintf(_dump_path, sizeof(_dump_path), "dump/sam_input_%d.bin", _dump_count);
        FILE *_df = fopen(_dump_path, "wb");
        if (_df) { fwrite(image_chw, sizeof(float), 3 * img->height * img->width, _df); fclose(_df); }
        fprintf(stderr, "Dumped SAM input pixels: %dx%d to %s (count=%d)\n",
                img->width, img->height, _dump_path, _dump_count);
        _dump_count++;
    }

    float *tokens = sam_forward_from_chw(ctx, image_chw, img->height, img->width,
                                          out_n_tokens, out_patch_embeds);
    free(image_chw);
    return tokens;
}

/* ========================================================================
 * Public API: Visual tokenizer forward (original interface)
 *
 * Takes raw pixels, preprocesses (resize/pad), then runs SAM.
 * For 1024x1024 global image (no interpolation needed).
 * ======================================================================== */
float *ds_visual_tokenizer_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                                    int width, int height, int channels,
                                    int *out_n_tokens, float **out_patch_embeds) {
    ds_config_t *cfg = &ctx->config;
    g_sam_is_v2 = (cfg->model_version == 2);

    /* Preprocess: resize/pad to target image size (e.g. 1024x1024) */
    ds_image_t img = { .pixels = (unsigned char *)pixels, .width = width, .height = height, .channels = channels };
    int target_size = cfg->image_size;

    ds_image_t *resized_img = NULL;
    if (width != target_size || height != target_size) {
        if (ds_verbose >= 1)
            fprintf(stderr, "Resizing %dx%d -> %dx%d\n",
                    width, height, target_size, target_size);
        if (cfg->model_version == 2) {
            resized_img = ds_image_pad(&img, target_size, 127);
        } else {
            resized_img = ds_image_resize(&img, target_size, target_size);
        }
        if (!resized_img) {
            fprintf(stderr, "Failed to resize image\n");
            return NULL;
        }
    }

    /* Convert to float CHW and run SAM forward */
    ds_image_t *effective_img = resized_img ? resized_img : &img;
    float *image_chw = ds_image_to_float_chw(effective_img);
    if (resized_img) ds_image_free(resized_img);
    if (!image_chw) return NULL;

    float *tokens = sam_forward_from_chw(ctx, image_chw, target_size, target_size,
                                          out_n_tokens, out_patch_embeds);
    free(image_chw);

    if (ds_verbose >= 1)
        fprintf(stderr, "Visual tokenizer: %dx%d image -> %d SAM tokens\n",
                width, height, *out_n_tokens);

    return tokens;
}

/* SAM forward pass (wrapper) */
float *ds_sam_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                       int width, int height, int channels,
                       int *out_n_tokens, float **out_patch_embeds) {
    return ds_visual_tokenizer_forward(ctx, pixels, width, height, channels,
                                       out_n_tokens, out_patch_embeds);
}

/* Weight Loading (done in ds_ocr.c during ds_load()) */
int ds_visual_tokenizer_load(ds_ctx_t *ctx) {
    return 0;
}
