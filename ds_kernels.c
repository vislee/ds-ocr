/*
 * ds_kernels.c - Math kernels for Qwen3-ASR inference
 * Adapted from voxtral-realtime project.
 */

#include "ds_kernels.h"
#include "ds_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

/* Helper: truncate F32 to BF16 precision and back (matches Python BF16 intermediate precision) */
static inline float ds_f32_to_bf16_to_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u = (u + 0x8000) & 0xFFFF0000u;  /* Round to nearest BF16 */
    float r;
    memcpy(&r, &u, sizeof(r));
    return r;
}

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define DS_MAX_THREADS 16

typedef void (*ds_parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[DS_MAX_THREADS - 1];
    int tids[DS_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    ds_parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *ds_worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        ds_parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

/* ========================================================================
 * MoE Operations
 * ======================================================================== */

void ds_moe_router(float *scores, const float *x, const float *gate_weight,
                   int hidden, int n_experts) {
    /* scores[e] = gate_weight[e, hidden] @ x[hidden] */
    ds_linear_nobias(scores, x, gate_weight, 1, hidden, n_experts);
}

void ds_moe_router_bf16(float *scores, const float *x, const uint16_t *gate_weight_bf16,
                          int hidden, int n_experts) {
    /* scores[e] = gate_weight_bf16[e, hidden] @ x[hidden] (BF16 weights for precision matching) */
    ds_linear_nobias_bf16(scores, x, gate_weight_bf16, 1, hidden, n_experts);
}

void ds_moe_top_k(int *top_indices, float *top_weights, const float *scores,
                  int n_experts, int top_k) {
    /* Find top-K experts using simple selection */
    int selected[DS_MAX_EXPERTS];
    float selected_scores[DS_MAX_EXPERTS];
    int n_selected = 0;

    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_score = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            /* Skip already selected */
            int already = 0;
            for (int j = 0; j < n_selected; j++) {
                if (selected[j] == e) { already = 1; break; }
            }
            if (already) continue;
            if (scores[e] > best_score) {
                best_score = scores[e];
                best = e;
            }
        }
        if (best >= 0) {
            selected[n_selected] = best;
            selected_scores[n_selected] = best_score;
            n_selected++;
        }
    }

    /* Softmax over top-K scores */
    float max_score = -1e30f;
    for (int k = 0; k < n_selected; k++) {
        if (selected_scores[k] > max_score) max_score = selected_scores[k];
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < n_selected; k++) {
        selected_scores[k] = expf(selected_scores[k] - max_score);
        sum_exp += selected_scores[k];
    }
    float inv_sum = 1.0f / sum_exp;

    for (int k = 0; k < top_k; k++) {
        if (k < n_selected) {
            top_indices[k] = selected[k];
            top_weights[k] = selected_scores[k] * inv_sum;
        } else {
            top_indices[k] = 0;
            top_weights[k] = 0.0f;
        }
    }
}

void ds_expert_forward(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate,
                       float *gate_buf, float *up_buf,
                       float *gate_up_buf, float *hidden_buf) {
    /* gate = gate_bf16 @ x, up = up_bf16 @ x */
    ds_linear_nobias_bf16(gate_buf, x, gate_bf16, 1, hidden, intermediate);
    ds_linear_nobias_bf16(up_buf, x, up_bf16, 1, hidden, intermediate);

    /* Truncate gate/up to BF16 precision to match Python's BF16 matmul output.
     * Python: BF16 × BF16 → BF16 (linear output is BF16 before next op).
     * C:      F32 × BF16 → F32 (NEON matvec gives F32 output).
     * Without truncation, gate_buf and up_buf have extra F32 mantissa bits
     * that Python doesn't, causing SwiGLU results to diverge. */
    extern int ds_bf16_simulate_python;
    if (ds_bf16_simulate_python) {
        for (int i = 0; i < intermediate; i++) {
            gate_buf[i] = ds_f32_to_bf16_to_f32(gate_buf[i]);
            up_buf[i] = ds_f32_to_bf16_to_f32(up_buf[i]);
        }
    }

    /* Fused gate+up for SwiGLU */
    for (int i = 0; i < intermediate; i++) {
        gate_up_buf[2 * i] = gate_buf[i];
        gate_up_buf[2 * i + 1] = up_buf[i];
    }

    /* SwiGLU: SiLU(gate) * up */
    ds_swiglu_multiply(hidden_buf, gate_up_buf, 1, intermediate);

    /* Truncate SwiGLU output to BF16 (Python's down projection input is BF16) */
    if (ds_bf16_simulate_python) {
        for (int i = 0; i < intermediate; i++) {
            hidden_buf[i] = ds_f32_to_bf16_to_f32(hidden_buf[i]);
        }
    }

    /* down projection */
    ds_linear_nobias_bf16(out, hidden_buf, down_bf16, 1, intermediate, hidden);

    /* Truncate final output to BF16 */
    if (ds_bf16_simulate_python) {
        for (int i = 0; i < hidden; i++) {
            out[i] = ds_f32_to_bf16_to_f32(out[i]);
        }
    }
}

/* Backward-compatible wrapper that allocates internally */
void ds_expert_forward_legacy(float *out, const float *x,
                       const uint16_t *gate_bf16, const uint16_t *up_bf16,
                       const uint16_t *down_bf16,
                       int hidden, int intermediate) {
    float *gate_buf = (float *)malloc(intermediate * sizeof(float));
    float *up_buf = (float *)malloc(intermediate * sizeof(float));
    float *gate_up_buf = (float *)malloc(2 * intermediate * sizeof(float));
    float *hidden_buf = (float *)malloc(intermediate * sizeof(float));

    if (!gate_buf || !up_buf || !gate_up_buf || !hidden_buf) {
        fprintf(stderr, "ds_expert_forward: allocation failed\n");
        free(gate_buf); free(up_buf); free(gate_up_buf); free(hidden_buf);
        return;
    }

    ds_expert_forward(out, x, gate_bf16, up_bf16, down_bf16,
                      hidden, intermediate, gate_buf, up_buf, gate_up_buf, hidden_buf);

    free(gate_buf); free(up_buf); free(gate_up_buf); free(hidden_buf);
}

void ds_expert_combine(float *output, const float *expert_outputs,
                       const int *top_indices, const float *top_weights,
                       int top_k, int hidden) {
    /* output[hidden] = sum_k(top_weights[k] * expert_outputs[k, hidden]) */
    memset(output, 0, hidden * sizeof(float));
    for (int k = 0; k < top_k; k++) {
        float w = top_weights[k];
        const float *expert_out = expert_outputs + (size_t)k * hidden;
        for (int i = 0; i < hidden; i++) {
            output[i] += w * expert_out[i];
        }
    }
}

/* ========================================================================
 * Mixed Attention (DeepEncoder V2: bidirectional for visual, causal for flow queries)
 * ======================================================================== */

void ds_mixed_attention(float *out, const float *Q, const float *K, const float *V,
                        int visual_len, int total_len, int n_heads,
                        int head_dim, float scale) {
    int hidden = n_heads * head_dim;
    double dscale = (double)scale;

    /* Allocate double buffers for V accumulation (float64 precision) */
    double *o_row_d = (double *)malloc(head_dim * sizeof(double));
    double *v_buf = (double *)malloc(head_dim * sizeof(double));

    for (int h = 0; h < n_heads; h++) {
        /* Visual tokens: bidirectional attention over visual tokens */
        for (int i = 0; i < visual_len; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            double max_score = -1e30;
            double sum_exp = 0.0;
            for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

            for (int j = 0; j < visual_len; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                /* Float64 QK dot product */
                double score = 0.0;
                for (int d = 0; d < head_dim; d++)
                    score += (double)q_row[d] * (double)k_row[d];
                score *= dscale;

                if (score > max_score) {
                    double correction = exp(max_score - score);
                    sum_exp = sum_exp * correction + 1.0;
                    for (int d = 0; d < head_dim; d++) {
                        v_buf[d] = (double)v_row[d] * correction;
                        o_row_d[d] = o_row_d[d] * correction + v_buf[d];
                    }
                    max_score = score;
                } else {
                    double wt = exp(score - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] += wt * (double)v_row[d];
                }
            }

            if (sum_exp > 0.0) {
                double inv_sum = 1.0 / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    o_row[d] = (float)(o_row_d[d] * inv_sum);
            } else {
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
            }
        }

        /* Causal flow queries: attend to all visual tokens + previous causal queries */
        for (int i = visual_len; i < total_len; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            double max_score = -1e30;
            double sum_exp = 0.0;
            for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

            int k_end = i + 1;
            if (k_end > total_len) k_end = total_len;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                /* Float64 QK dot product */
                double score = 0.0;
                for (int d = 0; d < head_dim; d++)
                    score += (double)q_row[d] * (double)k_row[d];
                score *= dscale;

                if (score > max_score) {
                    double correction = exp(max_score - score);
                    sum_exp = sum_exp * correction + 1.0;
                    for (int d = 0; d < head_dim; d++) {
                        v_buf[d] = (double)v_row[d] * correction;
                        o_row_d[d] = o_row_d[d] * correction + v_buf[d];
                    }
                    max_score = score;
                } else {
                    double wt = exp(score - max_score);
                    sum_exp += wt;
                    for (int d = 0; d < head_dim; d++)
                        o_row_d[d] += wt * (double)v_row[d];
                }
            }

            if (sum_exp > 0.0) {
                double inv_sum = 1.0 / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    o_row[d] = (float)(o_row_d[d] * inv_sum);
            } else {
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
            }
        }
    }

    free(o_row_d);
    free(v_buf);
}

/* ========================================================================
 * 2D Position Embeddings for vision tokens
 * ======================================================================== */

void ds_compute_2d_position_embeddings(float *pos_embed, int n_rows, int n_cols,
                                         int embed_dim) {
    /* Split embedding dimension in half: row position + column position */
    int half = embed_dim / 2;
    int total = n_rows * n_cols;

    for (int i = 0; i < total; i++) {
        int row = i / n_cols;
        int col = i % n_cols;
        float *emb = pos_embed + i * embed_dim;

        /* Row position encoding (first half) */
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)half);
            float angle = (float)row * freq;
            if (d % 2 == 0) emb[d] = sinf(angle);
            else emb[d] = cosf(angle);
        }

        /* Column position encoding (second half) */
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * (d / 2)) / (float)half);
            float angle = (float)col * freq;
            if (d % 2 == 0) emb[half + d] = sinf(angle);
            else emb[half + d] = cosf(angle);
        }
    }
}

void ds_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > DS_MAX_THREADS) n = DS_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, ds_worker_loop, &tp.tids[i]);
    }

    if (ds_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

int ds_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* Dispatch work to all threads; main thread is tid=0 */
static void ds_parallel_for(ds_parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void ds_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void ds_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void ds_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void ds_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void ds_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void ds_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void ds_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    ds_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void ds_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Reusable scratch buffer for bf16->f32 conversion */
static float *ds_bf16_scratch = NULL;
static size_t ds_bf16_scratch_cap = 0;

static float *ds_bf16_get_scratch(size_t n) {
    if (n > ds_bf16_scratch_cap) {
        free(ds_bf16_scratch);
        ds_bf16_scratch = (float *)malloc(n * sizeof(float));
        ds_bf16_scratch_cap = ds_bf16_scratch ? n : 0;
    }
    return ds_bf16_scratch;
}

typedef struct {
    const uint16_t *src;
    size_t n;
    float *dst_f32;
} ds_bf16_cache_entry_t;

static ds_bf16_cache_entry_t *bf16_cache = NULL;
static int bf16_cache_len = 0;
static int bf16_cache_cap = 0;
static size_t bf16_cache_bytes = 0;
static size_t bf16_cache_limit_bytes = 0;
static int bf16_cache_limit_init = 0;

static void bf16_cache_init_limit(void) {
    if (bf16_cache_limit_init) return;
    bf16_cache_limit_init = 1;

    /* Default OFF. Override with DS_BF16_CACHE_MB=<n> to enable. */
    unsigned long long mb = 0;
    const char *env = getenv("DS_BF16_CACHE_MB");
    if (env && env[0] != '\0') {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = v;
    }
    bf16_cache_limit_bytes = (size_t)(mb * 1024ULL * 1024ULL);

    if (ds_verbose >= 2) {
        fprintf(stderr, "BF16 cache: limit=%llu MB\n", mb);
    }
}

static const float *ds_bf16_get_cached_f32(const uint16_t *src, size_t n) {
    bf16_cache_init_limit();

    for (int i = 0; i < bf16_cache_len; i++) {
        if (bf16_cache[i].src == src && bf16_cache[i].n == n) {
            return bf16_cache[i].dst_f32;
        }
    }

    if (bf16_cache_limit_bytes == 0) return NULL;

    size_t bytes = n * sizeof(float);
    if (bytes > bf16_cache_limit_bytes) return NULL;
    if (bf16_cache_bytes + bytes > bf16_cache_limit_bytes) return NULL;

    float *dst = (float *)malloc(bytes);
    if (!dst) return NULL;
    ds_bf16_to_f32_buf(dst, src, n);

    if (bf16_cache_len == bf16_cache_cap) {
        int new_cap = bf16_cache_cap > 0 ? bf16_cache_cap * 2 : 256;
        ds_bf16_cache_entry_t *tmp = (ds_bf16_cache_entry_t *)realloc(
            bf16_cache, (size_t)new_cap * sizeof(ds_bf16_cache_entry_t));
        if (!tmp) {
            free(dst);
            return NULL;
        }
        bf16_cache = tmp;
        bf16_cache_cap = new_cap;
    }

    bf16_cache[bf16_cache_len].src = src;
    bf16_cache[bf16_cache_len].n = n;
    bf16_cache[bf16_cache_len].dst_f32 = dst;
    bf16_cache_len++;
    bf16_cache_bytes += bytes;
    return dst;
}

static const float *ds_bf16_get_f32_view(const uint16_t *src, size_t n) {
    const float *cached = ds_bf16_get_cached_f32(src, n);
    if (cached) return cached;

    float *scratch = ds_bf16_get_scratch(n);
    if (!scratch) return NULL;
    ds_bf16_to_f32_buf(scratch, src, n);
    return scratch;
}

/* Expert forward using F32 weights + BLAS sgemm (high precision, slower).
 * Used for prefill when DS_USE_F32_EXPERTS is set. */
void ds_expert_forward_f32(float *out, const float *x,
                            const uint16_t *gate_bf16, const uint16_t *up_bf16,
                            const uint16_t *down_bf16,
                            int hidden, int intermediate,
                            float *gate_buf, float *up_buf,
                            float *gate_up_buf, float *hidden_buf) {
    size_t n_gate = (size_t)intermediate * hidden;
    size_t n_up = (size_t)intermediate * hidden;
    size_t n_down = (size_t)hidden * intermediate;
    const float *gate_f32 = ds_bf16_get_f32_view(gate_bf16, n_gate);
    const float *up_f32 = ds_bf16_get_f32_view(up_bf16, n_up);
    const float *down_f32 = ds_bf16_get_f32_view(down_bf16, n_down);
    if (!gate_f32 || !up_f32 || !down_f32) {
        ds_expert_forward(out, x, gate_bf16, up_bf16, down_bf16,
                          hidden, intermediate, gate_buf, up_buf, gate_up_buf, hidden_buf);
        return;
    }
    ds_linear_nobias(gate_buf, x, gate_f32, 1, hidden, intermediate);
    ds_linear_nobias(up_buf, x, up_f32, 1, hidden, intermediate);
    for (int i = 0; i < intermediate; i++) {
        gate_up_buf[2 * i] = gate_buf[i];
        gate_up_buf[2 * i + 1] = up_buf[i];
    }
    ds_swiglu_multiply(hidden_buf, gate_up_buf, 1, intermediate);
    ds_linear_nobias(out, hidden_buf, down_f32, 1, intermediate, hidden);
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    ds_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} ds_matvec_task_t;

static void ds_matvec_worker(int tid, int n_threads, void *arg) {
    ds_matvec_task_t *t = (ds_matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

static void ds_bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    ds_matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    ds_parallel_for(ds_matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_ds_matvec_task_t;

static void qkv_ds_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_ds_matvec_task_t *t = (qkv_ds_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

void ds_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_ds_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    ds_parallel_for(qkv_ds_matvec_worker, &task);
}

void ds_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        ds_bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = ds_bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    ds_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void ds_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        ds_bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = ds_bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    ds_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* Public wrapper: compute logits = W_bf16 @ x + bias (single token, full output vector) */
void ds_bf16_matvec_pub(float *y, const float *x, const uint16_t *W_bf16,
                         const float *b, int in_dim, int out_dim) {
    ds_bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused). */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    ds_argmax_bf16_range_impl(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[DS_MAX_THREADS];
    float best_val[DS_MAX_THREADS];
} ds_argmax_task_t;

static void ds_argmax_worker(int tid, int n_threads, void *arg) {
    ds_argmax_task_t *t = (ds_argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

int ds_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    ds_argmax_task_t task;
    task.x = x;
    task.W_bf16 = W_bf16;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    ds_parallel_for(ds_argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}

void ds_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M == 1) {
        ds_bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        const float *B_f32 = ds_bf16_get_f32_view(B_bf16, n);
        if (!B_f32) return;
        ds_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void ds_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* DEBUG: dump im2col for any 3x3 s1 conv with c_in=256 */
    {
        static int _conv2d_call_count = 0;
        _conv2d_call_count++;
        if (getenv("DS_DUMP_CONV2_IM2COL")) {
            fprintf(stderr, "[DUMP] ds_conv2d call #%d: c_in=%d c_out=%d h=%d w=%d kh=%d kw=%d s=%d p=%d h_out=%d w_out=%d\n",
                    _conv2d_call_count, c_in, c_out, h_in, w_in, kh, kw, stride, padding, h_out, w_out);
            if (c_in == 256 && c_out == 256 && kh == 3 && stride == 1 && padding == 1) {
                FILE *df = fopen("dump/c_conv2_im2col.bin", "wb");
                if (df) { fwrite(cols, sizeof(float), (size_t)patch_size * spatial_out, df); fclose(df); }
                fprintf(stderr, "[DUMP] *** neck conv2 im2col saved: %d x %d ***\n", patch_size, spatial_out);
            }
        }
    }

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int s = 0; s < spatial_out; s++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
            }
            out[oc * spatial_out + s] = sum;
        }
    }
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void ds_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void ds_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void ds_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void ds_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void ds_gelu(float *x, int n) {
    /* Exact GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
     * Matches PyTorch nn.GELU(approximate='none') */
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float arg = val * 0.7071067811865475f; /* 1/sqrt(2) */
        float erf_val = erff(arg);
        x[i] = 0.5f * val * (1.0f + erf_val);
    }
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} ds_swiglu_task_t;

static void ds_swiglu_worker(int tid, int n_threads, void *arg) {
    ds_swiglu_task_t *t = (ds_swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): keep single-pass scalar to avoid alias hazards. */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
        }
    }
}

void ds_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    ds_swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        ds_parallel_for(ds_swiglu_worker, &task);
    } else {
        ds_swiglu_worker(0, 1, &task);
    }
}

void ds_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float ds_dot_f32(const float *a, const float *b, int n) {
    return ds_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void ds_vec_scale_inplace(float *dst, float scale, int n) {
    ds_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void ds_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    ds_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void ds_vec_scale_add(float *dst, const float *src, float correction, int n) {
    ds_vec_scale_add_impl(dst, src, correction, n);
}

void ds_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq,
                                   int n_heads, int head_dim, float scale) {
    int hidden = n_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq; i++) {
            const float *q_row = Q + i * hidden + h * head_dim;
            float *o_row = out + i * hidden + h * head_dim;

            /* Online softmax with full bidirectional attention */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < seq; j++) {
                const float *k_row = K + j * hidden + h * head_dim;
                const float *v_row = V + j * hidden + h * head_dim;

                float score = ds_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    ds_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    ds_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                ds_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

static void ds_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = ds_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    ds_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    ds_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                ds_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} ds_causal_attn_task_t;

static void ds_causal_attn_worker(int tid, int n_threads, void *arg) {
    ds_causal_attn_task_t *t = (ds_causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    ds_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

void ds_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    if (tp.n_threads > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        ds_causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        ds_parallel_for(ds_causal_attn_worker, &task);
        return;
    }

    ds_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void ds_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void ds_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    /* Split-half RoPE cos/sin cache (matches LlamaAttention used by DeepSeek-OCR-2):
     * Layout: [seq, head_dim] where cos[0:half] == cos[half:end] (repeated).
     * Python LlamaAttention's RoPE:
     *   freqs = outer(t, inv_freq)  # [seq, half]
     *   emb = cat(freqs, freqs, dim=-1)  # [seq, dim]
     *   cos_cached = emb.cos()
     */
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Split-half: first half and second half have identical values */
            cos_out[s * head_dim + d]         = c;
            cos_out[s * head_dim + d + half]  = c;
            sin_out[s * head_dim + d]         = sn;
            sin_out[s * head_dim + d + half]  = sn;
        }
    }
}

void ds_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * LLaMA-style split-half RoPE (matches LlamaAttention used by DeepSeek-OCR-2):
     * rotate_half(x) = cat(-x[half:], x[:half])
     * q_embed = q * cos + rotate_half(q) * sin
     *
     * For each element d in [0, half):
     *   out[d]       = x[d]*cos[d] - x[d+half]*sin[d]
     *   out[d+half]  = x[d]*sin[d] + x[d+half]*cos[d]
     *
     * Note: cos[d] == cos[d+half] and sin[d] == sin[d+half] (split-half repeated)
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

            /* Apply split-half rotate: save first half, then update in-place */
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[d + half];
                vec[d]         = x1 * c[d]         - x2 * sn[d];
                vec[d + half]  = x1 * sn[d + half] + x2 * c[d + half];
            }
        }
    }
}
