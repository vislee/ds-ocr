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

#include <time.h>
#include "ds_visual_tokenizer.h"
#include "ds_kernels.h"
#include "ds_image.h"
#include "ds_safetensors.h"
#include "ds_ocr.h"
#include "ds_dump.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif
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
/* V2 rel_pos bias: computed as dot product between query and rel_pos embedding.
 * Python's add_decomposed_rel_pos: bias = q @ Rh[idx] + q @ Rw[idx]
 * rel_pos shape: [2*rel_size-1, head_dim], shared across heads.
 * q_val: the query vector for this head at position (qy, qx), shape [head_dim].
 * Returns: sum_d(q_val[d] * rel_pos[idx, d])
 */
static float get_rel_pos_v2(const float *rel_pos, int head_dim,
                             const float *q_val,
                             int q_coord, int k_coord, int rel_size) {
    int rel_offset = q_coord - k_coord + rel_size - 1;
    if (rel_offset < 0 || rel_offset >= 2 * rel_size - 1) return 0.0f;
    const float *rp = rel_pos + rel_offset * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) sum += q_val[d] * rp[d];  /* dot product */
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
 * V2: rel_pos_h/w shape [2*window_h/w-1, head_dim] (shared across heads)
 *   bias = sum_d(q[h, qi, d] * Rh[qi-ki+size-1, d]) + sum_d(q[h, qi, d] * Rw[qi-ki+size-1, d])
 *   This is the decomposed relative position encoding from SAM/DeepSeek-OCR.
 * V1: rel_pos_h/w shape [n_heads, head_dim, 2*window_h/w-1]
 *   (Not used in V2 — kept for backward compat)
 *
 * q: query tensor, shape [n_heads * seq_len * head_dim] in (heads, seq, dim) layout
 *    For V2: q[h * seq_len * head_dim + qi * head_dim + d]
 */
static void add_rel_pos_bias(float *attn_scores, int n_heads,
                              int q_h, int q_w, int k_h, int k_w,
                              const float *rel_pos_h, const float *rel_pos_w,
                              int head_dim, int is_v2,
                              const float *q) {
    int seq_len = q_h * q_w;
    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < seq_len; qi++) {
            int qy = qi / q_w;
            int qx = qi % q_w;
            const float *q_val = q + (size_t)h * seq_len * head_dim + qi * head_dim;
            for (int ki = 0; ki < k_h * k_w; ki++) {
                int ky = ki / k_w;
                int kx = ki % k_w;
                float bias;
                if (is_v2) {
                    bias = get_rel_pos_v2(rel_pos_h, head_dim, q_val, qy, ky, q_h)
                         + get_rel_pos_v2(rel_pos_w, head_dim, q_val, qx, kx, q_w);
                } else {
                    bias = get_rel_pos_v1(rel_pos_h, n_heads, head_dim, qy, ky, q_h, h)
                         + get_rel_pos_v1(rel_pos_w, n_heads, head_dim, qx, kx, q_w, h);
                }
                attn_scores[(size_t)h * seq_len * seq_len + qi * seq_len + ki] += bias;
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

    /* BLAS-accelerated window attention.
     * Q, K, V are [n_tokens, hidden] in row-major.
     * For each head h: Q_h is [n_tokens, head_dim] with stride=hidden.
     *
     * Step 1: scores[h] = Q_h @ K_h^T * scale  (BLAS sgemm)
     * Step 2: Add rel_pos bias (block-structured)
     * Step 3: Row-wise softmax
     * Step 4: attn_out_h = scores @ V_h  (BLAS sgemm)
     */
    float *scores = (float *)malloc((size_t)n_tokens * n_tokens * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        const float *Q_h = Q + h * head_dim;   /* stride = hidden */
        const float *K_h = K + h * head_dim;
        const float *V_h = V + h * head_dim;
        float *out_h = out + h * head_dim;

        /* Step 1: QK^T with BLAS */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_tokens, n_tokens, head_dim,
                    scale, Q_h, hidden, K_h, hidden,
                    0.0f, scores, n_tokens);
#else
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    s += Q_h[i * hidden + d] * K_h[j * hidden + d];
                scores[i * n_tokens + j] = s * scale;
            }
        }
#endif

        /* Step 2: Add rel_pos bias (block-structured for window attention) */
        /* Height bias: depends on (qy, ky) only */
        for (int dy = 0; dy < 2 * win_h - 1; dy++) {
            int actual_dy = dy - (win_h - 1);
            const float *rh_vec = rel_pos_h + dy * head_dim;

            for (int qy = 0; qy < win_h; qy++) {
                int ky = qy - actual_dy;
                if (ky < 0 || ky >= win_h) continue;

                int row_start = qy * win_w;
                int col_start = ky * win_w;

                for (int qx = 0; qx < win_w; qx++) {
                    const float *q_vec = Q_h + (row_start + qx) * hidden;
                    float bias = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        bias += q_vec[d] * rh_vec[d];

                    float *row = scores + (row_start + qx) * n_tokens + col_start;
                    for (int kx = 0; kx < win_w; kx++)
                        row[kx] += bias;
                }
            }
        }

        /* Width bias: depends on (qx, kx) only */
        for (int dx = 0; dx < 2 * win_w - 1; dx++) {
            int actual_dx = dx - (win_w - 1);
            const float *rw_vec = rel_pos_w + dx * head_dim;

            for (int qx = 0; qx < win_w; qx++) {
                int kx = qx - actual_dx;
                if (kx < 0 || kx >= win_w) continue;

                for (int qy = 0; qy < win_h; qy++) {
                    const float *q_vec = Q_h + (qy * win_w + qx) * hidden;
                    float bias = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        bias += q_vec[d] * rw_vec[d];

                    for (int ky = 0; ky < win_h; ky++) {
                        int row = qy * win_w + qx;
                        int col = ky * win_w + kx;
                        scores[row * n_tokens + col] += bias;
                    }
                }
            }
        }

        /* Step 3: Row-wise softmax in float64 for precision */
        {
            double *softmax_buf = (double *)malloc(n_tokens * sizeof(double));
            for (int i = 0; i < n_tokens; i++) {
                float *row = scores + i * n_tokens;
                double max_s = -1e30;
                for (int j = 0; j < n_tokens; j++) {
                    double s = (double)row[j];
                    if (s > max_s) max_s = s;
                    softmax_buf[j] = s;
                }
                double sum_e = 0.0;
                for (int j = 0; j < n_tokens; j++) {
                    softmax_buf[j] = exp(softmax_buf[j] - max_s);
                    sum_e += softmax_buf[j];
                }
                if (sum_e > 0.0) {
                    double inv = 1.0 / sum_e;
                    for (int j = 0; j < n_tokens; j++)
                        row[j] = (float)(softmax_buf[j] * inv);
                }
            }
            free(softmax_buf);
        }

        /* Step 4: attn_out_h = scores @ V_h  (BLAS sgemm) */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_tokens, head_dim, n_tokens,
                    1.0f, scores, n_tokens, V_h, hidden,
                    0.0f, out_h, hidden);
#else
        for (int i = 0; i < n_tokens; i++) {
            const float *srow = scores + i * n_tokens;
            float *o_row = out_h + i * hidden;
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < n_tokens; j++)
                    sum += srow[j] * V_h[j * hidden + d];
                o_row[d] = sum;
            }
        }
#endif
    }

    free(scores);
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
            /* Try loading precomputed rel_pos from model_dir first */
            char rp_h_path[1024], rp_w_path[1024];
            snprintf(rp_h_path, sizeof(rp_h_path), "%s/rel_pos_h_layer%d_size%d.bin",
                     vt->model_dir, layer_idx, needed_size);
            snprintf(rp_w_path, sizeof(rp_w_path), "%s/rel_pos_w_layer%d_size%d.bin",
                     vt->model_dir, layer_idx, needed_size);
            FILE *rfh = fopen(rp_h_path, "rb");
            FILE *rfw = fopen(rp_w_path, "rb");
            if (rfh && rfw) {
                interp_rel_h = (float *)malloc(needed_size * head_dim * sizeof(float));
                interp_rel_w = (float *)malloc(needed_size * head_dim * sizeof(float));
                size_t rh_read = fread(interp_rel_h, sizeof(float), needed_size * head_dim, rfh);
                size_t rw_read = fread(interp_rel_w, sizeof(float), needed_size * head_dim, rfw);
                fclose(rfh); rfh = NULL;
                fclose(rfw); rfw = NULL;
                if ((int)rh_read == needed_size * head_dim && (int)rw_read == needed_size * head_dim) {
                    eff_rel_h = interp_rel_h;
                    eff_rel_w = interp_rel_w;
                    if (ds_verbose >= 1)
                        fprintf(stderr, "Loaded precomputed rel_pos for layer %d from %s\n",
                                layer_idx, rp_h_path);
                    goto rel_pos_done;
                }
                free(interp_rel_h); free(interp_rel_w);
                interp_rel_h = interp_rel_w = NULL;
            }
            if (rfh) fclose(rfh);
            if (rfw) fclose(rfw);

            /* Fallback: linear interpolation of rel_pos from stored_size → needed_size.
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

    rel_pos_done:
    ; /* label target for precomputed rel_pos skip */

    /* Attention */
    float *attn_out = (float *)calloc(seq_len * dim, sizeof(float));

    if (use_global_attn) {
        /* BLAS-accelerated global attention with float64 softmax.
         *
         * Decomposition:
         *   scores[h] = Q_h @ K_h^T * scale + rel_pos_bias_h + rel_pos_bias_w
         *   attn_weights[h] = softmax(scores[h])    (float64 for stability)
         *   attn_out_h = attn_weights[h] @ V_h
         *
         * Q_h, K_h, V_h are [seq_len, head_dim] slices of Q, K, V.
         * Steps 1 and 3 use BLAS sgemm; step 2 (rel_pos + softmax) is scalar.
         */
        float scale = 1.0f / sqrtf((float)head_dim);

        /* Allocate score buffer: [seq_len, seq_len] in row-major */
        float *scores = (float *)malloc((size_t)seq_len * seq_len * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            const float *Q_h = Q + h * head_dim;   /* [seq_len, dim] with stride dim */
            const float *K_h = K + h * head_dim;
            const float *V_h = V + h * head_dim;
            float *out_h = attn_out + h * head_dim;

            /* Step 1: scores = Q_h @ K_h^T * scale
             * Q_h is [seq_len, dim] with leading dim = dim, but we only want
             * the head_dim columns. We use a stride-aware sgemm:
             * Q_h has stride=dim, but only head_dim contiguous elements per row.
             * K_h similarly.
             *
             * Since Q/K are laid out as [seq_len, n_heads*head_dim], and we want
             * columns [h*head_dim .. (h+1)*head_dim), we need to use the
             * column-major BLAS with appropriate leading dimensions.
             *
             * Equivalently: scores[i,j] = sum_d Q[i, h*hd+d] * K[j, h*hd+d]
             * = Q[:, h*hd:(h+1)*hd] @ K[:, h*hd:(h+1)*hd]^T
             *
             * Using CblasRowMajor, CblasNoTrans, CblasTrans:
             *   C = alpha * A * B^T + beta * C
             *   A = Q_h [seq_len, head_dim], lda = dim (row stride)
             *   B = K_h [seq_len, head_dim], ldb = dim (row stride)
             *   C = scores [seq_len, seq_len], ldc = seq_len
             */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, head_dim,
                        scale, Q_h, dim, K_h, dim,
                        0.0f, scores, seq_len);
#else
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        s += Q_h[i * dim + d] * K_h[j * dim + d];
                    scores[i * seq_len + j] = s * scale;
                }
            }
#endif

            /* Step 2: Add rel_pos bias using block-structured computation.
             *
             * Key insight: rel_pos bias decomposes as:
             *   bias_h[i,j] = Q[i] · rh[qy-ky+seq_h-1]  (depends on qy,ky only)
             *   bias_w[i,j] = Q[i] · rw[qx-kx+seq_w-1]  (depends on qx,kx only)
             *
             * For height: within a (qy,ky) block, bias is row-constant.
             * For width: within a (qx,kx) block, bias is col-constant.
             * This reduces O(seq²*head_dim) to O(seq_h²*seq_w*head_dim + seq_w²*seq_h*head_dim).
             */

            /* Height bias: for each dy offset, fill matching blocks */
            for (int dy = 0; dy < 2 * seq_h - 1; dy++) {
                int actual_dy = dy - (seq_h - 1);
                const float *rh_vec = eff_rel_h + dy * head_dim;

                for (int qy = 0; qy < seq_h; qy++) {
                    int ky = qy - actual_dy;
                    if (ky < 0 || ky >= seq_h) continue;

                    int row_start = qy * seq_w;
                    int col_start = ky * seq_w;

                    /* Compute dot products for each qx: [seq_w] vector */
                    for (int qx = 0; qx < seq_w; qx++) {
                        const float *q_vec = Q_h + (row_start + qx) * dim;
                        float bias = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            bias += q_vec[d] * rh_vec[d];

                        /* Fill entire row in this block (same value for all kx) */
                        float *row = scores + (row_start + qx) * seq_len + col_start;
                        for (int kx = 0; kx < seq_w; kx++)
                            row[kx] += bias;
                    }
                }
            }

            /* Width bias: for each dx offset, fill matching positions */
            for (int dx = 0; dx < 2 * seq_w - 1; dx++) {
                int actual_dx = dx - (seq_w - 1);
                const float *rw_vec = eff_rel_w + dx * head_dim;

                for (int qx = 0; qx < seq_w; qx++) {
                    int kx = qx - actual_dx;
                    if (kx < 0 || kx >= seq_w) continue;

                    /* For each qy: compute dot product and add to score */
                    for (int qy = 0; qy < seq_h; qy++) {
                        const float *q_vec = Q_h + (qy * seq_w + qx) * dim;
                        float bias = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            bias += q_vec[d] * rw_vec[d];

                        /* Add to all ky positions at column kx */
                        for (int ky = 0; ky < seq_h; ky++) {
                            int row = qy * seq_w + qx;
                            int col = ky * seq_w + kx;
                            scores[row * seq_len + col] += bias;
                        }
                    }
                }
            }

            /* Debug: dump scores with bias for block 11 head 0 */
            if (layer_idx == 11 && h == 0 && getenv("DS_DUMP_TENSORS")) {
                FILE *sf = fopen("dump/c_block11_h0_scores_with_bias.bin", "wb");
                if (sf) { fwrite(scores, sizeof(float), (size_t)seq_len * seq_len, sf); fclose(sf); }
                fprintf(stderr, "Dumped block11 head0 scores+relpos_bias (%d x %d)\n", seq_len, seq_len);
            }

            /* Step 2b: Softmax in float64 for stability (row-wise) */
            double *softmax_buf = (double *)malloc(seq_len * sizeof(double));

            for (int i = 0; i < seq_len; i++) {
                float *score_row = scores + i * seq_len;

                double max_s = -1e30;
                for (int j = 0; j < seq_len; j++) {
                    double s = (double)score_row[j];
                    softmax_buf[j] = s;
                    if (s > max_s) max_s = s;
                }

                double sum_e = 0.0;
                for (int j = 0; j < seq_len; j++) {
                    softmax_buf[j] = exp(softmax_buf[j] - max_s);
                    sum_e += softmax_buf[j];
                }
                if (sum_e > 0.0) {
                    for (int j = 0; j < seq_len; j++)
                        score_row[j] = (float)(softmax_buf[j] / sum_e);
                }
            }
            free(softmax_buf);

            /* Step 3: attn_out_h = softmax_scores @ V_h
             * scores is [seq_len, seq_len], V_h is [seq_len, head_dim] with stride dim
             * out_h is [seq_len, head_dim] with stride dim
             */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, head_dim, seq_len,
                        1.0f, scores, seq_len, V_h, dim,
                        0.0f, out_h, dim);
#else
            for (int i = 0; i < seq_len; i++) {
                const float *srow = scores + i * seq_len;
                float *o_row = out_h + i * dim;
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++)
                        sum += srow[j] * V_h[j * dim + d];
                    o_row[d] = sum;
                }
            }
#endif
        }

    } else {
        /* Window attention: partition into windows (with padding), attend within each.
         * Python's order: norm1 -> pad -> window_partition -> QKV -> attn -> window_unpartition
         * We must pad x_norm first, then compute QKV on the padded data,
         * so that padded positions get QKV(0) = qkv_b (non-zero), matching Python.
         * Previously we computed QKV on unpadded data then padded Q/K/V with zeros,
         * which caused real tokens near the padding boundary to have different
         * attention scores (diff max=0.209 at Block 0). */
        int n_windows, padded_h, padded_w;
        float *padded_x = window_partition_pad(x_norm, seq_h, seq_w, win_size, dim, &n_windows, &padded_h, &padded_w);
        int win_tokens = win_size * win_size;

        float *win_attn = (float *)calloc((size_t)n_windows * win_tokens * dim, sizeof(float));

        for (int w = 0; w < n_windows; w++) {
            float *wx = padded_x + (size_t)w * win_tokens * dim;

            /* QKV projection for this window (on padded data, matching Python) */
            float *wqkv = (float *)malloc((size_t)win_tokens * 3 * dim * sizeof(float));
            ds_linear(wqkv, wx, qkv_w, qkv_b, win_tokens, dim, 3 * dim);

            float *wQ = (float *)malloc((size_t)win_tokens * dim * sizeof(float));
            float *wK = (float *)malloc((size_t)win_tokens * dim * sizeof(float));
            float *wV = (float *)malloc((size_t)win_tokens * dim * sizeof(float));
            for (int s = 0; s < win_tokens; s++) {
                memcpy(wQ + s * dim, wqkv + s * 3 * dim, dim * sizeof(float));
                memcpy(wK + s * dim, wqkv + s * 3 * dim + dim, dim * sizeof(float));
                memcpy(wV + s * dim, wqkv + s * 3 * dim + 2 * dim, dim * sizeof(float));
            }
            free(wqkv);

            float *wout = win_attn + (size_t)w * win_tokens * dim;
            window_attn_forward(wout, wQ, wK, wV, win_tokens, n_heads, head_dim,
                                rel_h, rel_w, win_size, win_size);

            free(wQ); free(wK); free(wV);
        }
        free(padded_x);

        /* Unpartition back to full sequence (removes padding) */
        window_unpartition_pad(attn_out, seq_h, seq_w, win_size, dim, padded_h, padded_w, win_attn);
        free(win_attn);
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

    /* DUMP: after conv1 */
    if (getenv("DS_DUMP_SAM_LAYERS")) {
        FILE *f = fopen("dump/c_neck_after_conv1.bin", "wb");
        if (f) { fwrite(conv1_out, sizeof(float), c_mid * spatial_sz, f); fclose(f); }
        /* DUMP: conv1 weight for debugging */
        f = fopen("dump/c_neck_conv1_weight.bin", "wb");
        if (f) { fwrite(vt->sam_neck_conv1_weight, sizeof(float), 256 * 768 * 1 * 1, f); fclose(f); }
        /* DUMP: conv2 weight for debugging */
        f = fopen("dump/c_neck_conv2_weight.bin", "wb");
        if (f) { fwrite(vt->sam_neck_conv2_weight, sizeof(float), 256 * 256 * 3 * 3, f); fclose(f); }
        /* DUMP: ln1 weight/bias */
        f = fopen("dump/c_neck_ln1_weight.bin", "wb");
        if (f) { fwrite(vt->sam_neck_ln1_weight, sizeof(float), 256, f); fclose(f); }
        f = fopen("dump/c_neck_ln1_bias.bin", "wb");
        if (f) { fwrite(vt->sam_neck_ln1_bias, sizeof(float), 256, f); fclose(f); }
    }

    /* LayerNorm2d after conv1 */
    float *ln1_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    layer_norm_2d(ln1_out, conv1_out, vt->sam_neck_ln1_weight, vt->sam_neck_ln1_bias,
                  c_mid, spatial_sz);
    free(conv1_out);

    /* DUMP: after ln1 */
    if (getenv("DS_DUMP_SAM_LAYERS")) {
        FILE *f = fopen("dump/c_neck_after_ln1.bin", "wb");
        if (f) { fwrite(ln1_out, sizeof(float), c_mid * spatial_sz, f); fclose(f); }
    }

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
    free(ln1_out);

    /* DUMP: after conv2 */
    if (getenv("DS_DUMP_SAM_LAYERS")) {
        FILE *f = fopen("dump/c_neck_after_conv2.bin", "wb");
        if (f) { fwrite(conv2_out, sizeof(float), c_mid * spatial_sz, f); fclose(f); }
    }

    /* LayerNorm2d after conv2 */
    float *ln2_out = (float *)malloc(c_mid * spatial_sz * sizeof(float));
    layer_norm_2d(ln2_out, conv2_out, vt->sam_neck_ln2_weight, vt->sam_neck_ln2_bias,
                  c_mid, spatial_sz);
    free(conv2_out);

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
 * PyTorch F.interpolate(bicubic, antialias=True) uses Keys cubic with a=-0.5.
 * This matches PIL/Pillow BICUBIC default behavior.
 * Previously we used a=-0.75 which was incorrect.
 */
static float bicubic_weight(float x) {
    const float a = -0.5f;
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
                    float dy = (float)(sy0 + ky) - sy;
                    float wy = bicubic_weight(dy * fminf(y_scale, 1.0f));

                    for (int kx = -x_support; kx <= x_support; kx++) {
                        int x_clamped = sx0 + kx;
                        if (x_clamped < 0) x_clamped = 0;
                        if (x_clamped >= src_w) x_clamped = src_w - 1;
                        float dx = (float)(sx0 + kx) - sx;
                        float wx = bicubic_weight(dx * fminf(x_scale, 1.0f));

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
            /* Try loading precomputed pos_embed from model_dir (avoids C interpolation mismatch) */
            char pe_path[1024];
            snprintf(pe_path, sizeof(pe_path), "%s/pos_embed_%dx%d.bin", vt->model_dir, tgt_grid, tgt_grid);
            FILE *pf = fopen(pe_path, "rb");
            if (pf) {
                interp_pos = (float *)malloc(tgt_grid * tgt_grid * DS_SAM_EMBED_DIM * sizeof(float));
                size_t nread = fread(interp_pos, sizeof(float),
                                      tgt_grid * tgt_grid * DS_SAM_EMBED_DIM, pf);
                fclose(pf);
                if ((int)nread == tgt_grid * tgt_grid * DS_SAM_EMBED_DIM) {
                    pos_embed = interp_pos;
                    if (ds_verbose >= 1)
                        fprintf(stderr, "Loaded precomputed pos_embed from %s (%dx%d)\n",
                                pe_path, tgt_grid, tgt_grid);
                } else {
                    free(interp_pos);
                    interp_pos = interpolate_pos_embed(pos_embed, src_grid, tgt_grid, DS_SAM_EMBED_DIM);
                    pos_embed = interp_pos;
                }
            } else {
                /* Fallback to C bicubic interpolation */
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
    static int _sam_call_idx = 0;  /* track which SAM call (crop 0-5 or global 6) */
    double _sam_t0 = 0, _sam_t1 = 0; struct timespec _sam_ts; clock_gettime(CLOCK_MONOTONIC, &_sam_ts); _sam_t0 = _sam_ts.tv_sec + _sam_ts.tv_nsec/1e9;
    for (int l = 0; l < 12; l++) {
        float *next = (float *)malloc(n_patches * DS_SAM_EMBED_DIM * sizeof(float));
        sam_layer_forward(next, x, vt, l, grid_h, grid_w);
        
        /* Debug: dump layer output for comparison with Python (only crop 0 to avoid overwrite) */
        if (getenv("DS_DUMP_SAM_LAYERS") && _sam_call_idx == 0) {
            char _lpath[256];
            snprintf(_lpath, sizeof(_lpath), "dump/c_sam_crop0_layer%d_out.bin", l);
            FILE *df = fopen(_lpath, "wb");
            if (df) { fwrite(next, sizeof(float), n_patches * DS_SAM_EMBED_DIM, df); fclose(df); }
            /* Also dump input to SAM layer 0 (after pos_embed) */
            if (l == 0) {
                df = fopen("dump/c_sam_crop0_after_posembed.bin", "wb");
                if (df) { fwrite(x, sizeof(float), n_patches * DS_SAM_EMBED_DIM, df); fclose(df); }
            }
            fprintf(stderr, "Dumped SAM layer %d I/O for comparison\n", l);
        }
        
        free(x);
        x = next;

        if (ds_verbose >= 1)
            clock_gettime(CLOCK_MONOTONIC, &_sam_ts); _sam_t1 = _sam_ts.tv_sec + _sam_ts.tv_nsec/1e9; \
            if (ds_verbose >= 2) fprintf(stderr, "  SAM layer %d: %.2fs\n", l, _sam_t1 - _sam_t0); _sam_t0 = _sam_t1;

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

    /* Dump neck output for debugging */
    if (getenv("DS_DUMP_SAM_LAYERS")) {
        char fname[256];
        snprintf(fname, sizeof(fname), "dump/c_sam_crop%d_neck_out.bin", _sam_call_idx);
        FILE *f = fopen(fname, "wb");
        if (f) { fwrite(neck_out, sizeof(float), 256 * n_patches, f); fclose(f); }
    }

    /* SAM downsample: [256, H, W] -> [512, H/2, W/2] -> [ds2_dim, H/4, W/4] */
    int ds_h, ds_w;
    int ds2_dim = cfg->sam_ds2_dim;
    float *ds_out = sam_downsample_forward(neck_out, grid_h, grid_w, vt, ds2_dim, &ds_h, &ds_w);
    free(neck_out);

    int ds_spatial = ds_h * ds_w;

    /* Dump downsample output for debugging */
    if (getenv("DS_DUMP_SAM_LAYERS")) {
        char fname[256];
        snprintf(fname, sizeof(fname), "dump/c_sam_crop%d_downsample_out.bin", _sam_call_idx);
        FILE *f = fopen(fname, "wb");
        if (f) { fwrite(ds_out, sizeof(float), ds2_dim * ds_spatial, f); fclose(f); }
    }

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

    _sam_call_idx++;  /* increment for next SAM call (next crop or global view) */

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
    g_sam_is_v2 = (ctx->config.model_version == 2);
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
            /* V2: pad with gray (127) to maintain aspect ratio */
            resized_img = ds_image_pad(&img, target_size, 127);
        } else if (cfg->model_version == 3) {
            /* V3 (Unlimited-OCR): Python uses ImageOps.pad() with mean color (0.5*255=127).
             * Same as V2 — pad to maintain aspect ratio, don't stretch. */
            resized_img = ds_image_pad(&img, target_size, 127);
        } else {
            /* V1: stretch to target size */
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
