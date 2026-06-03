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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    free(qkv);

    /* Attention */
    float *attn_out = (float *)calloc(seq_len * dim, sizeof(float));

    if (use_global_attn) {
        /* Global attention: all tokens attend to all tokens */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                const float *q_row = Q + i * dim + h * head_dim;
                float *o_row = attn_out + i * dim + h * head_dim;
                int qy = i / seq_w, qx = i % seq_w;

                float max_s = -1e30f;
                float sum_e = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = 0; j < seq_len; j++) {
                    const float *k_row = K + j * dim + h * head_dim;
                    const float *v_row = V + j * dim + h * head_dim;
                    int ky = j / seq_w, kx = j % seq_w;

                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += q_row[d] * k_row[d];
                    score *= scale;

                    /* Rel pos bias: dot(q, Rh[offset]) + dot(q, Rw[offset]) */
                    int rh_off = qy - ky + seq_h - 1;
                    int rw_off = qx - kx + seq_w - 1;
                    if (rh_off >= 0 && rh_off < 2 * seq_h - 1) {
                        const float *rh_vec = rel_h + rh_off * head_dim;
                        for (int d = 0; d < head_dim; d++) score += q_row[d] * rh_vec[d];
                    }
                    if (rw_off >= 0 && rw_off < 2 * seq_w - 1) {
                        const float *rw_vec = rel_w + rw_off * head_dim;
                        for (int d = 0; d < head_dim; d++) score += q_row[d] * rw_vec[d];
                    }

                    if (score > max_s) {
                        float c = expf(max_s - score);
                        sum_e = sum_e * c + 1.0f;
                        for (int d = 0; d < head_dim; d++) o_row[d] = o_row[d] * c + v_row[d];
                        max_s = score;
                    } else {
                        float w = expf(score - max_s);
                        sum_e += w;
                        for (int d = 0; d < head_dim; d++) o_row[d] += w * v_row[d];
                    }
                }
                if (sum_e > 0.0f) {
                    float inv = 1.0f / sum_e;
                    for (int d = 0; d < head_dim; d++) o_row[d] *= inv;
                }
            }
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

    for (int i = 0; i < seq_len * dim; i++) out[i] = x[i] + proj_out[i];
    free(proj_out);

    /* FFN: Pre-norm (LayerNorm2) + GELU MLP */
    float *ffn_norm = (float *)malloc(seq_len * dim * sizeof(float));
    ds_layer_norm(ffn_norm, out, norm2_w, norm2_b, seq_len, dim, 1e-6f);

    float *fc1 = (float *)malloc(seq_len * mlp_dim * sizeof(float));
    ds_linear(fc1, ffn_norm, lin1_w, lin1_b, seq_len, dim, mlp_dim);
    ds_gelu(fc1, seq_len * mlp_dim);

    float *fc2 = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(fc2, fc1, lin2_w, lin2_b, seq_len, mlp_dim, dim);

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

    /* Also return the patch embeds for CLIP input (before adding pos embed) */
    /* patches is in [768, n_patches] format (CHW) */

    return patches;
}

/* ========================================================================
 * Main Forward Pass
 * ======================================================================== */

float *ds_visual_tokenizer_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                                    int width, int height, int channels,
                                    int *out_n_tokens, float **out_patch_embeds) {
    ds_visual_tokenizer_t *vt = &ctx->vis_tokenizer;
    ds_config_t *cfg = &ctx->config;

    /* Step 1: Resize to target image size (e.g. 1024x1024) */
    g_sam_is_v2 = (cfg->model_version == 2);
    ds_image_t img = { .pixels = (unsigned char *)pixels, .width = width, .height = height, .channels = channels };
    int target_size = cfg->image_size;

    ds_image_t *resized_img = NULL;
    int need_resize = (width != target_size || height != target_size);
    if (need_resize) {
        if (ds_verbose >= 1)
            fprintf(stderr, "Resizing %dx%d -> %dx%d\n",
                    width, height, target_size, target_size);
        if (cfg->model_version == 2) {
            /* V2: pad to maintain aspect ratio (like ImageOps.pad) with gray fill */
            resized_img = ds_image_pad(&img, target_size, 128);
        } else {
            resized_img = ds_image_resize(&img, target_size, target_size);
        }
        if (!resized_img) {
            fprintf(stderr, "Failed to resize image\n");
            return NULL;
        }
    }

    /* Convert to float CHW */
    ds_image_t *effective_img = resized_img ? resized_img : &img;
    float *image_chw = ds_image_to_float_chw(effective_img);
    if (resized_img) ds_image_free(resized_img);
    if (!image_chw) return NULL;

    /* Step 2: SAM patch embedding */
    int grid_h, grid_w;
    float *patch_spatial = sam_patch_embed(image_chw, target_size, target_size,
                                            vt->sam_patch_embed_weight,
                                            vt->sam_patch_embed_bias,
                                            &grid_h, &grid_w);
    free(image_chw);

    int n_patches = grid_h * grid_w;

    /* Save patch_embeds for CLIP input (in [768, n_patches] format) */
    if (out_patch_embeds) {
        *out_patch_embeds = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));
        memcpy(*out_patch_embeds, patch_spatial, n_patches * DS_SAM_EMBED_DIM * sizeof(float));
    }

    /* Step 3: Add positional embedding */
    /* Convert [768, n_patches] → [n_patches, 768] and add pos_embed */
    float *x = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));

    if (cfg->model_version == 2) {
        /* V2: pos_embed is 2D spatial [1, grid_h, grid_w, 768] (e.g. [1, 64, 64, 768]) */
        for (int p = 0; p < n_patches; p++) {
            int row = p / grid_w;
            int col = p % grid_w;
            for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
                x[p * DS_SAM_EMBED_DIM + d] = patch_spatial[d * n_patches + p]
                    + vt->sam_pos_embed[(row * grid_w + col) * DS_SAM_EMBED_DIM + d];
            }
        }
    } else {
        /* V1: pos_embed is 1D sequence [1, n_patches+1, 768] (with CLS token at index 0) */
        for (int p = 0; p < n_patches; p++) {
            for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
                x[p * DS_SAM_EMBED_DIM + d] = patch_spatial[d * n_patches + p]
                    + vt->sam_pos_embed[(p + 1) * DS_SAM_EMBED_DIM + d]; /* +1 for CLS token pos */
            }
        }
    }
    free(patch_spatial);

    if (ds_verbose >= 2)
        fprintf(stderr, "SAM patch embed: %dx%d -> %d patches\n", grid_h, grid_w, n_patches);

    /* Step 4: SAM transformer layers (12 layers with window/global attention) */
    for (int l = 0; l < 12; l++) {
        float *next = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));
        sam_layer_forward(next, x, vt, l, grid_h, grid_w);
        free(x);
        x = next;
        if (ds_verbose >= 2)
            fprintf(stderr, "SAM layer %d/%d done (global_attn=%d)\n", l + 1, 12,
                    (l == 2 || l == 5 || l == 8 || l == 11));
    }

    /* Step 5: Reshape to spatial [768, grid_h, grid_w] for neck */
    float *spatial = (float *)malloc(DS_SAM_EMBED_DIM * n_patches * sizeof(float));
    for (int p = 0; p < n_patches; p++) {
        for (int d = 0; d < DS_SAM_EMBED_DIM; d++) {
            spatial[d * n_patches + p] = x[p * DS_SAM_EMBED_DIM + d];
        }
    }
    free(x);

    /* Step 6: SAM neck: [768, H, W] → [256, H, W] */
    float *neck_out = sam_neck_forward(spatial, grid_h, grid_w, vt);
    free(spatial);

    /* Step 7: SAM downsample: [256, H, W] → [512, H/2, W/2] → [ds2_dim, H/4, W/4] */
    int ds_h, ds_w;
    int ds2_dim = cfg->sam_ds2_dim;
    float *ds_out = sam_downsample_forward(neck_out, grid_h, grid_w, vt, ds2_dim, &ds_h, &ds_w);
    free(neck_out);

    int ds_spatial = ds_h * ds_w;

    /* Convert [ds2_dim, ds_h*ds_w] → [ds_h*ds_w, ds2_dim] for downstream use */
    float *tokens = (float *)malloc(ds_spatial * ds2_dim * sizeof(float));
    for (int p = 0; p < ds_spatial; p++) {
        for (int d = 0; d < ds2_dim; d++) {
            tokens[p * ds2_dim + d] = ds_out[d * ds_spatial + p];
        }
    }
    free(ds_out);

    *out_n_tokens = ds_spatial;

    if (ds_verbose >= 1)
        fprintf(stderr, "Visual tokenizer: %dx%d image -> %d SAM tokens (%dx%d)\n",
                width, height, *out_n_tokens, ds_h, ds_w);

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
