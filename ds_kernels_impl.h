/*
 * ds_kernels_impl.h - internal architecture dispatch for hot kernels
 */

/* ═══════════════════════════════════════════════════════════════════════════
 * 【内核分发层】"编译期定候选、运行时定人选"的两级结构
 * （详见 docs/09-量化与部署篇.md 第 6 节、docs/03-推理引擎实战篇.md 第 4 节）
 *
 * 三级内核架构：
 *   ds_kernels.h            ← 平台无关统一 API，上层只认 ds_xxx_impl(...) 语法
 *   ds_kernels_impl.h       ← 本文件：编译期平台检测 + 宏/函数指针分发
 *     ├── ds_kernels_generic.c   标量参考实现（任何平台都编译，正确性基准）
 *     ├── ds_kernels_neon.c      ARM NEON：vshll 移位路径 + bfdot 硬件路径
 *     └── ds_kernels_avx.c       x86：AVX-512F+BW 或 AVX2+FMA 路径
 *
 * 编译期检测机制：__ARM_NEON、__AVX2__、__FMA__、__ARM_FEATURE_BF16_VECTOR_
 * ARITHMETIC 等都是编译器依据 -mcpu/-march 标志预定义的宏（gcc -mcpu=native
 * 在 Apple M2 上会同时定义 __ARM_NEON 与 __ARM_FEATURE_BF16_VECTOR_ARITHMETIC；
 * 在 x86 上按 CPU 能力定义 __AVX2__/__FMA__/__AVX512F__…）。本文件没有使用
 * __attribute__ 技巧，全部依赖这些预置宏做条件编译。
 *
 * 【为什么 matvec/argmax 在 ARM 上走函数指针，其余走纯宏】
 * - ARM：同一份二进制要同时跑 M1（无 FEAT_BF16）和 M2+（有）两种芯片，
 *   CPU 能力只能运行时探测（Apple 用 sysctl，Linux 用 getauxval，见
 *   ds_kernels.c 的 ds_kernels_dispatch_init()），编译期无法二选一——于是
 *   把 vshll 版与 bfdot 版都编进二进制，启动时把函数指针切到更快的一份。
 * - x86：AVX-512/AVX2 由 -march=native 在编译期定死，不存在"一份二进制
 *   适配多代 CPU"的运行时选择需求，直接宏展开为静态调用（零间接开销，
 *   编译器还能内联优化）。
 * - 两组 _impl 宏（展开成函数名，或展开成指针变量名）保证了上层调用语法
 *   完全一致；函数指针的间接跳转对 [1280,1792] 量级的 matvec 可忽略，
 *   换来的是单二进制适配多代芯片的能力。
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* 头文件保护：本文件定义 typedef 与 _impl 分发宏，会被多个 .c 重复包含；
 * 首次包含时定义 DS_KERNELS_IMPL_H，再次包含时预处理器整体跳过，
 * 避免 typedef 重复声明与宏重定义告警。 */
#ifndef DS_KERNELS_IMPL_H
#define DS_KERNELS_IMPL_H

#include <stdint.h> /* BF16 以 uint16_t 位模式承载，转换出的 F32 位模式用 uint32_t */

/* ── 标量参考实现（实现在 ds_kernels_generic.c，任何平台都编译）─────────────
 * 它们既是无 SIMD 平台的回退路径，也是 NEON/AVX 版本的正确性对照基准。 */

/* ds_bf16_matvec_fused_generic - BF16 权重 × F32 输入的融合矩阵向量乘（标量基准）
 * 【算法/数学】y[o] = bias[o] + Σ_{k=0..in_dim-1} W_bf16[o][k]·x[k]，行主序。
 *   "fused" 指 BF16→F32 在线转换（位模式 <<16）+ 点积累加 + bias 加法一趟
 *   完成；权重始终保持 BF16 存储，不生成 F32 副本（省一半内存带宽）。
 * 【参数】y[out_dim] 输出；x[in_dim] F32 输入向量；W_bf16 指向 [out_dim][in_dim]
 *   行主序 BF16 权重；bias[out_dim] 可为 NULL；in_dim/out_dim 为两维大小。
 * 【为什么】decode 的核心原语：每步对 QKV/路由专家/共享专家权重各做 matvec，
 *   算术强度 ≈1 FLOP/B，内存带宽受限（docs/09 §1）——SIMD 优化的是转换与
 *   乘加的指令数，权重读取量不变。
 * 【参考】docs/03 §3.2-§3.3、§4.4；docs/09 §1。
 */
void ds_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim);

/* ds_argmax_bf16_range_generic - 行区间 [start,end) 上的流式 argmax（标量基准）
 * 【算法/数学】对 o ∈ [start,end) 逐行计算 y[o] = Σ_k W[o][k]·x[k]（无 bias），
 *   边算边比较只保留当前最大值，不物化 logits 数组。
 * 【参数】x[in_dim]；W_bf16 行主序权重；start/end 为行号区间（左闭右开）；
 *   best_out/best_val_out 输出最大行号与最大值；区间为空时返回 (start, -1e30f)。
 * 【为什么】LM head 有 129280×1280 ≈ 331 MB 权重，贪心解码只需 argmax；
 *   区间接口（而非整表）是为了按线程分块：ds_argmax_worker 把词表切成
 *   DS_MAX_THREADS 段并行扫描，再归约各线程的最大值（ds_kernels.c）。
 * 【参考】docs/03 §3.4；docs/05 §4.1。
 */
void ds_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out);

/* ds_dot_f32_generic - F32 点积 Σ a[i]·b[i]（标量基准）。
 * 注意力 QK 得分等小向量（head_dim=128）的基础原语，数据常驻 L1；
 * SIMD 版用 4/8 lane 并行 + 多累加器打破 FMA 依赖链。 */
float ds_dot_f32_generic(const float *a, const float *b, int n);

/* ds_vec_scale_inplace_generic - 原地缩放 dst[i] *= scale。
 * 注意力输出归一化（乘 1/Σexp）使用。 */
void ds_vec_scale_inplace_generic(float *dst, float scale, int n);

/* ds_vec_axpy_inplace_generic - 原地 axpy：dst[i] += alpha·src[i]。
 * 注意力在线 softmax 中累加新 V 行（权重 exp(score-max)）使用。 */
void ds_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);

/* ds_vec_scale_add_generic - 原地融合 dst[i] = dst[i]·correction + src[i]。
 * 在线 softmax 遇到新最大值时，把已累加的输出按 correction = exp(m_old-m_new)
 * 整体重缩放后并入新项——一次遍历完成"缩放旧和 + 加新和"。 */
void ds_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

/* 预处理器自上而下取第一个命中的分支：NEON 与 AVX2 属不同指令集架构，
 * 两类平台宏不会同时定义，故同一份编译产物只会激活 NEON / AVX2 / 回退
 * 三者之一，分支互斥、不叠加。 */
#ifdef __ARM_NEON
/* ── ARM NEON 候选实现（实现在 ds_kernels_neon.c）───────────────────────────
 * ds_bf16_matvec_fused_neon / ds_argmax_bf16_range_neon：vshll+vfma 路径——
 *   每条 vshll_n_u16 把 4 个 BF16 加宽左移成 F32 位模式，再 vfmaq_f32 乘加，
 *   每 4 个 BF16 MAC 需 2 条指令；M1 等无 FEAT_BF16 芯片上的运行时默认。
 * 其余 4 个 F32 向量工具只有一份实现，直接宏绑定（见下方宏定义）。 */
void ds_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float ds_dot_f32_neon(const float *a, const float *b, int n);
void ds_vec_scale_inplace_neon(float *dst, float scale, int n);
void ds_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void ds_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

/* BF16 dot product variants (ARMv8.6-A FEAT_BF16, Apple M2+).
 * Always compiled when __ARM_FEATURE_BF16_VECTOR_ARITHMETIC is defined;
 * runtime dispatch picks this path only if the CPU supports FEAT_BF16. */
/* 【补充】是否编入由 -mcpu 决定（编译器在 ARMv8.6-A 及以上才定义该宏）；
 * 是否【选用】由运行时探测决定。一条 bfdot = 8 个 BF16 MAC → 4 个 F32 累加，
 * 指令密度是 vshll+vfma 路径的 4 倍（docs/09 §6.3）。 */
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
void ds_bf16_matvec_fused_neon_bf16(float *y, const float *x, const uint16_t *W_bf16,
                                      const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_neon_bf16(const float *x, const uint16_t *W_bf16,
                                       int in_dim, int start, int end,
                                       int *best_out, float *best_val_out);
#endif

/* Runtime dispatch: function pointers set by ds_kernels_dispatch_init().
 * On Apple M2+, points to bfdot variants; on M1, falls back to vshll. */
/* 【补充】指针本体（定义 + 初始化）在 ds_kernels.c：初值取安全下限
 * （NEON vshll 版 / 非 ARM 的 generic 版），首次 matvec 前经
 * ds_dispatch_ensure_init() 惰性调用 dispatch_init，按运行时 CPU 能力切换。 */
typedef void (*ds_bf16_matvec_fused_fn_t)(float *y, const float *x,
                                           const uint16_t *W_bf16,
                                           const float *bias, int in_dim, int out_dim);
typedef void (*ds_argmax_bf16_range_fn_t)(const float *x, const uint16_t *W_bf16,
                                            int in_dim, int start, int end,
                                            int *best_out, float *best_val_out);

extern ds_bf16_matvec_fused_fn_t ds_bf16_matvec_fused_impl_ptr;
extern ds_argmax_bf16_range_fn_t ds_argmax_bf16_range_impl_ptr;

/* Convenience macros — call through function pointers */
/* 【补充】ARM 上 ds_bf16_matvec_fused_impl(...) 即经指针的间接调用；
 * x86/回退平台上同名宏直接展开成静态函数调用，上层代码两平台零改动。 */
#define ds_bf16_matvec_fused_impl  ds_bf16_matvec_fused_impl_ptr
#define ds_argmax_bf16_range_impl  ds_argmax_bf16_range_impl_ptr

/* F32 向量工具：无运行时候选，即使是 ARM 也编译期直连（零间接开销） */
#define ds_dot_f32_impl ds_dot_f32_neon
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_neon
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_neon
#define ds_vec_scale_add_impl ds_vec_scale_add_neon

#elif defined(__AVX2__) && defined(__FMA__)
/* ── x86 AVX2+FMA 候选实现（实现在 ds_kernels_avx.c）────────────────────────
 * 进入本分支要求编译器同时定义 __AVX2__ 与 __FMA__（由 -march/-mcpu 在
 * x86 上定义）；文件内部再按 __AVX512F__ && __AVX512BW__ 二选一编译
 * 512 位或 256 位实现。x86 没有 BF16 硬件点积（Intel AVX512-BF16 仅存在于
 * 少数服务器 CPU），仍走"u16 加宽 + 左移 16 位 + FMADD"的软件转换路径；
 * 全部编译期定死，无运行时探测。 */
void ds_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void ds_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float ds_dot_f32_avx(const float *a, const float *b, int n);
void ds_vec_scale_inplace_avx(float *dst, float scale, int n);
void ds_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void ds_vec_scale_add_avx(float *dst, const float *src, float correction, int n);

/* AVX2: compile-time dispatch (no runtime BF16 variant to choose) */
#define ds_bf16_matvec_fused_impl ds_bf16_matvec_fused_avx
#define ds_argmax_bf16_range_impl ds_argmax_bf16_range_avx
#define ds_dot_f32_impl ds_dot_f32_avx
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_avx
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_avx
#define ds_vec_scale_add_impl ds_vec_scale_add_avx

#else
/* ── 回退：无 NEON/AVX 的平台（如未开 -mavx2 的老 x86、32 位 ARM），
 *    六个 _impl 全部直连标量实现 ── */
#define ds_bf16_matvec_fused_impl ds_bf16_matvec_fused_generic
#define ds_argmax_bf16_range_impl ds_argmax_bf16_range_generic
#define ds_dot_f32_impl ds_dot_f32_generic
#define ds_vec_scale_inplace_impl ds_vec_scale_inplace_generic
#define ds_vec_axpy_inplace_impl ds_vec_axpy_inplace_generic
#define ds_vec_scale_add_impl ds_vec_scale_add_generic
#endif

#endif /* DS_KERNELS_IMPL_H */
