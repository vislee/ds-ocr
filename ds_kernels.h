/*
 * ds_kernels.h - Math kernels for DeepSeek-OCR inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Adapted from antirez/qwen-asr project.
 */

#ifndef DS_KERNELS_H
#define DS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* Maximum experts per layer (for static array sizing) */
#define DS_MAX_EXPERTS 64

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void ds_add_inplace(float *a, const float *b, int n);
void ds_mul_inplace(float *a, const float *b, int n);
void ds_scale(float *x, float s, int n);
void ds_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void ds_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
void ds_linear(float *y, const float *x, const float *W, const float *b,
               int seq_len, int in_dim, int out_dim);

void ds_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */
void ds_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                    const float *b, int seq_len, int in_dim, int out_dim);

void ds_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim);

/* seq=1 decoder fast path: compute Q/K/V matvecs with one threaded dispatch */
void ds_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                const uint16_t *Wq_bf16,
                                const uint16_t *Wk_bf16,
                                const uint16_t *Wv_bf16,
                                int in_dim, int q_dim, int kv_dim);

void ds_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N);

/* ========================================================================
 * 2D Convolution (for vision tokenizer SAM patch embed + conv compression)
 * ======================================================================== */

void ds_conv2d(float *out, const float *in, const float *weight, const float *bias,
               int c_in, int c_out, int h_in, int w_in,
               int kh, int kw, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* LayerNorm with bias: out = (x - mean) / sqrt(var + eps) * weight + bias */
void ds_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                   int seq_len, int hidden, float eps);

/* RMS Normalization: out = x / rms(x) * weight */
void ds_rms_norm(float *out, const float *x, const float *weight,
                 int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder */
void ds_rms_norm_per_head(float *x, const float *weight,
                           int seq_len, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void ds_silu(float *x, int n);
void ds_gelu(float *x, int n);
void ds_softmax(float *x, int rows, int cols);

/* out[seq,inter] = SiLU(gate_up[seq,2*inter][:,even]) * gate_up[:,odd] */
void ds_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Bidirectional attention (encoder).
 * Q, K, V: [seq, n_heads * head_dim]
 * out: [seq, n_heads * head_dim]
 */
void ds_bidirectional_attention(float *out, const float *Q, const float *K,
                                const float *V, int seq, int n_heads,
                                int head_dim, float scale);

/*
 * Causal attention with GQA (decoder).
 * Q: [seq_q, n_heads * head_dim]
 * K: [seq_k, n_kv_heads * head_dim]
 * V: [seq_k, n_kv_heads * head_dim]
 * q_offset: global position of first query (for causal mask)
 */
void ds_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int q_offset);

/*
 * Mixed attention for DeepEncoder V2: visual tokens use bidirectional,
 * causal flow queries use causal attention.
 * visual_len: number of visual tokens (bidirectional)
 * total_len: visual_len + n_causal_queries (visual + causal flow queries)
 */
void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

/* RoPE: compute cos/sin for positions (NeoX-style) */
void ds_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                           int seq, int head_dim, float theta);

/* Apply RoPE to Q or K (in-place) */
void ds_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                         int seq, int n_heads, int head_dim);

/* 2D position embeddings for vision tokens (row + column) */
void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols,
                                        int embed_dim);

/* ========================================================================
 * MoE Operations
 * ======================================================================== */

/* Compute router gate scores: scores[n_experts] = gate_weight[n_experts, hidden] @ x[hidden] */
void ds_moe_router(float *scores, const float *x, const float *gate_weight,
                   int hidden, int n_experts);

/* Select top-K experts from gate scores. Returns expert indices and weights. */
void ds_moe_top_k(int *top_indices, float *top_weights, const float *scores,
                  int n_experts, int top_k);

/* Single expert SwiGLU forward: out = down(SiLU(gate(x)) * up(x)) */
void ds_expert_forward(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate,
                       float *gate_buf, float *up_buf,
                       float *gate_up_buf, float *hidden_buf);

/* Legacy wrapper that allocates internally */
void ds_expert_forward_legacy(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate);

/* Combine expert outputs with router weights */
void ds_expert_combine(float *output, const float *expert_outputs,
                       const int *top_indices, const float *top_weights,
                       int top_k, int hidden);

/* ========================================================================
 * Sampling
 * ======================================================================== */

/* Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits */
int ds_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                           int in_dim, int out_dim);

/* Compute logits = W_bf16 @ x + bias, storing full [out_dim] vector */
void ds_bf16_matvec_pub(float *y, const float *x, const uint16_t *W_bf16,
                         const float *b, int in_dim, int out_dim);

/* ========================================================================
 * Threading
 * ======================================================================== */

/* Set number of threads for parallel operations (default: 1) */
void ds_set_threads(int n);

/* Get number of available CPU cores */
int ds_get_num_cpus(void);

/* Global verbose flag */
extern int ds_verbose;

#endif /* DS_KERNELS_H */
