/*
 * qwen_asr_kernels_neon.c - ARM NEON hot kernels
 */

#include "ds_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <string.h>

/* ========================================================================
 * NEON-optimized BF16 → F32 batch conversion
 * ======================================================================== */

void ds_bf16_to_f32_neon(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i + 4 <= n; i += 4) {
        uint16x4_t bf = vld1_u16(src + i);
        uint32x4_t f32 = vshll_n_u16(bf, 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(f32));
    }
    for (; i < n; i++) {
        uint32_t u = ((uint32_t)src[i]) << 16;
        memcpy(dst + i, &u, sizeof(float));
    }
}

/* ========================================================================
 * NEON-optimized F32 → BF16 conversion
 * ======================================================================== */

void ds_f32_to_bf16_neon(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        /* Round to nearest even: add bias then shift right 16 */
        uint32x4_t b0 = vaddq_u32(vreinterpretq_u32_f32(v0), vdupq_n_u32(0x8000));
        uint32x4_t b1 = vaddq_u32(vreinterpretq_u32_f32(v1), vdupq_n_u32(0x8000));
        uint16x4_t r0 = vshrn_n_u32(b0, 16);
        uint16x4_t r1 = vshrn_n_u32(b1, 16);
        vst1q_u16(dst + i, vcombine_u16(r0, r1));
    }
    for (; i < n; i++) {
        uint32_t u;
        memcpy(&u, src + i, sizeof(u));
        u = (u + 0x8000) & 0xFFFF0000u;
        dst[i] = (uint16_t)(u >> 16);
    }
}

void ds_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time, 32 elements/iter, 8 accumulators */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }
        for (; k + 8 <= in_dim; k += 8) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), x1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), x1);
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }
        y[o] = s0;
        y[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void ds_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 2 rows at a time, 32 elements/iter, 8 accumulators per row */
    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }

        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }

        if (s0 > best_val) { best_val = s0; best = o; }
        if (s1 > best_val) { best_val = s1; best = o + 1; }
    }

    for (; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
}

float ds_dot_f32_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void ds_vec_scale_inplace_neon(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++) dst[i] *= scale;
}

void ds_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void ds_vec_scale_add_neon(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(s0, d0, c));
        vst1q_f32(dst + i + 4, vfmaq_f32(s1, d1, c));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

#endif /* __ARM_NEON */

/* ========================================================================
 * NEON BF16 dot product matvec — native BF16 dot (bfdot) for 2x throughput
 * on Apple M2+ / ARMv8.6-A with FEAT_BF16.
 *
 * bfdot v0.4s, v1.8h, v2.8h: 8 BF16 MACs → 4 F32 accs in 1 instruction.
 * vs vshll+vfma: 2 instructions for 4 BF16 MACs → 4 F32 accs.
 * Uses inline asm because Apple clang 15 lacks the vdotq_f32 bfloat16 intrinsic.
 * ======================================================================== */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

#include <arm_neon.h>
#include <string.h>

/* BF16 dot product: acc = acc + dot(a_bf16, b_bf16), 8 BF16 → 4 F32 */
static inline float32x4_t bfdot_f32(float32x4_t acc, bfloat16x8_t a, bfloat16x8_t b) {
    __asm__("bfdot %0.4s, %1.8h, %2.8h" : "+w"(acc) : "w"(a), "w"(b));
    return acc;
}

/* Convert 2x F32x4 vectors to 1x BF16x8 (round to nearest even) */
static inline bfloat16x8_t f32x8_to_bf16x8(float32x4_t lo, float32x4_t hi) {
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(lo), hi);
}

#define BP(p) ((const __bf16 *)(const void *)(p))

void ds_bf16_matvec_fused_neon_bf16(float *y, const float *x, const uint16_t *W_bf16,
                                      const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* 2 rows at a time, 32 BF16 elements/iter, 4 dot products per row */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
        float32x4_t b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            bfloat16x8_t xb0 = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            bfloat16x8_t xb1 = f32x8_to_bf16x8(vld1q_f32(x+k+8), vld1q_f32(x+k+12));
            bfloat16x8_t xb2 = f32x8_to_bf16x8(vld1q_f32(x+k+16), vld1q_f32(x+k+20));
            bfloat16x8_t xb3 = f32x8_to_bf16x8(vld1q_f32(x+k+24), vld1q_f32(x+k+28));

            a0 = bfdot_f32(a0, vld1q_bf16(BP(w0+k)),    xb0);
            a1 = bfdot_f32(a1, vld1q_bf16(BP(w0+k+8)),  xb1);
            a2 = bfdot_f32(a2, vld1q_bf16(BP(w0+k+16)), xb2);
            a3 = bfdot_f32(a3, vld1q_bf16(BP(w0+k+24)), xb3);

            b0 = bfdot_f32(b0, vld1q_bf16(BP(w1+k)),    xb0);
            b1 = bfdot_f32(b1, vld1q_bf16(BP(w1+k+8)),  xb1);
            b2 = bfdot_f32(b2, vld1q_bf16(BP(w1+k+16)), xb2);
            b3 = bfdot_f32(b3, vld1q_bf16(BP(w1+k+24)), xb3);
        }
        for (; k + 8 <= in_dim; k += 8) {
            bfloat16x8_t xb = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            a0 = bfdot_f32(a0, vld1q_bf16(BP(w0+k)), xb);
            b0 = bfdot_f32(b0, vld1q_bf16(BP(w1+k)), xb);
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3)));

        for (; k < in_dim; k++) {
            uint32_t u0 = ((uint32_t)w0[k]) << 16, u1 = ((uint32_t)w1[k]) << 16;
            float v0, v1; memcpy(&v0, &u0, 4); memcpy(&v1, &u1, 4);
            s0 += v0 * x[k]; s1 += v1 * x[k];
        }
        y[o] = s0; y[o+1] = s1;
    }

    /* Single row */
    for (; o < out_dim; o++) {
        const uint16_t *wr = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;
        float32x4_t aa = vdupq_n_f32(0), ab = vdupq_n_f32(0);
        for (; k + 16 <= in_dim; k += 16) {
            bfloat16x8_t xb0 = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            bfloat16x8_t xb1 = f32x8_to_bf16x8(vld1q_f32(x+k+8), vld1q_f32(x+k+12));
            aa = bfdot_f32(aa, vld1q_bf16(BP(wr+k)),   xb0);
            ab = bfdot_f32(ab, vld1q_bf16(BP(wr+k+8)), xb1);
        }
        sum += vaddvq_f32(vaddq_f32(aa, ab));
        for (; k + 8 <= in_dim; k += 8) {
            bfloat16x8_t xb = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            float32x4_t tmp = bfdot_f32(vdupq_n_f32(0), vld1q_bf16(BP(wr+k)), xb);
            sum += vaddvq_f32(tmp);
        }
        for (; k < in_dim; k++) {
            uint32_t u = ((uint32_t)wr[k]) << 16; float v; memcpy(&v, &u, 4);
            sum += v * x[k];
        }
        y[o] = sum;
    }
}

#undef BP

void ds_argmax_bf16_range_neon_bf16(const float *x, const uint16_t *W_bf16,
                                       int in_dim, int start, int end,
                                       int *best_out, float *best_val_out) {
    int best = start; float best_val = -1e30f; int o = start;
    #define BP(p) ((const __bf16 *)(const void *)(p))

    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o+1) * in_dim;
        float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
        float32x4_t b0=vdupq_n_f32(0),b1=vdupq_n_f32(0),b2=vdupq_n_f32(0),b3=vdupq_n_f32(0);
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            bfloat16x8_t xb0=f32x8_to_bf16x8(vld1q_f32(x+k),vld1q_f32(x+k+4));
            bfloat16x8_t xb1=f32x8_to_bf16x8(vld1q_f32(x+k+8),vld1q_f32(x+k+12));
            bfloat16x8_t xb2=f32x8_to_bf16x8(vld1q_f32(x+k+16),vld1q_f32(x+k+20));
            bfloat16x8_t xb3=f32x8_to_bf16x8(vld1q_f32(x+k+24),vld1q_f32(x+k+28));
            a0=bfdot_f32(a0,vld1q_bf16(BP(w0+k)),xb0);   a1=bfdot_f32(a1,vld1q_bf16(BP(w0+k+8)),xb1);
            a2=bfdot_f32(a2,vld1q_bf16(BP(w0+k+16)),xb2); a3=bfdot_f32(a3,vld1q_bf16(BP(w0+k+24)),xb3);
            b0=bfdot_f32(b0,vld1q_bf16(BP(w1+k)),xb0);   b1=bfdot_f32(b1,vld1q_bf16(BP(w1+k+8)),xb1);
            b2=bfdot_f32(b2,vld1q_bf16(BP(w1+k+16)),xb2); b3=bfdot_f32(b3,vld1q_bf16(BP(w1+k+24)),xb3);
        }
        for (; k+8 <= in_dim; k += 8) {
            bfloat16x8_t xb=f32x8_to_bf16x8(vld1q_f32(x+k),vld1q_f32(x+k+4));
            a0=bfdot_f32(a0,vld1q_bf16(BP(w0+k)),xb); b0=bfdot_f32(b0,vld1q_bf16(BP(w1+k)),xb);
        }
        float s0=vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3)));
        float s1=vaddvq_f32(vaddq_f32(vaddq_f32(b0,b1),vaddq_f32(b2,b3)));
        for (; k < in_dim; k++) {
            uint32_t u0=((uint32_t)w0[k])<<16,u1=((uint32_t)w1[k])<<16;
            float v0,v1; memcpy(&v0,&u0,4); memcpy(&v1,&u1,4);
            s0+=v0*x[k]; s1+=v1*x[k];
        }
        if (s0>best_val){best_val=s0;best=o;} if(s1>best_val){best_val=s1;best=o+1;}
    }
    for (; o < end; o++) {
        const uint16_t *wr = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f; int k = 0;
        float32x4_t aa=vdupq_n_f32(0),ab=vdupq_n_f32(0);
        for (; k+16 <= in_dim; k += 16) {
            bfloat16x8_t xb0=f32x8_to_bf16x8(vld1q_f32(x+k),vld1q_f32(x+k+4));
            bfloat16x8_t xb1=f32x8_to_bf16x8(vld1q_f32(x+k+8),vld1q_f32(x+k+12));
            aa=bfdot_f32(aa,vld1q_bf16(BP(wr+k)),xb0); ab=bfdot_f32(ab,vld1q_bf16(BP(wr+k+8)),xb1);
        }
        sum += vaddvq_f32(vaddq_f32(aa,ab));
        for (; k+8 <= in_dim; k += 8) {
            bfloat16x8_t xb=f32x8_to_bf16x8(vld1q_f32(x+k),vld1q_f32(x+k+4));
            sum += vaddvq_f32(bfdot_f32(vdupq_n_f32(0),vld1q_bf16(BP(wr+k)),xb));
        }
        for (; k < in_dim; k++) {
            uint32_t u=((uint32_t)wr[k])<<16; float v; memcpy(&v,&u,4); sum+=v*x[k];
        }
        if (sum>best_val){best_val=sum;best=o;}
    }
    #undef BP
    *best_out = best; *best_val_out = best_val;
}

#endif /* __ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC */
