/*
 * ds_kernels_impl.h - internal architecture dispatch for hot kernels
 */

#ifndef DS_KERNELS_IMPL_H
#define DS_KERNELS_IMPL_H

#include <stdint.h>

void ds_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out);
float ds_dot_f32_generic(const float *a, const float *b, int n);
void ds_vec_scale_inplace_generic(float *dst, float scale, int n);
void ds_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void ds_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#ifdef __ARM_NEON
void ds_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float ds_dot_f32_neon(const float *a, const float *b, int n);
void ds_vec_scale_inplace_neon(float *dst, float scale, int n);
void ds_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void ds_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

#define ds_bf16_matvec_fused_impl ds_bf16_matvec_fused_neon
#define ds_argmax_bf16_range_impl ds_argmax_bf16_range_neon
#define ds_dot_f32_impl ds_dot_f32_neon
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_neon
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_neon
#define ds_vec_scale_add_impl ds_vec_scale_add_neon

#elif defined(__AVX2__) && defined(__FMA__)
void ds_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float ds_dot_f32_avx(const float *a, const float *b, int n);
void ds_vec_scale_inplace_avx(float *dst, float scale, int n);
void ds_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void ds_vec_scale_add_avx(float *dst, const float *src, float correction, int n);

#define ds_bf16_matvec_fused_impl ds_bf16_matvec_fused_avx
#define ds_argmax_bf16_range_impl ds_argmax_bf16_range_avx
#define ds_dot_f32_impl ds_dot_f32_avx
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_avx
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_avx
#define ds_vec_scale_add_impl ds_vec_scale_add_avx

#else
#define ds_bf16_matvec_fused_impl ds_bf16_matvec_fused_generic
#define ds_argmax_bf16_range_impl ds_argmax_bf16_range_generic
#define ds_dot_f32_impl ds_dot_f32_generic
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_generic
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_generic
#define ds_vec_scale_add_impl ds_vec_scale_add_generic
#endif

#endif /* DS_KERNELS_IMPL_H */
