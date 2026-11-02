/*
 * qwen_asr_kernels_generic.c - architecture-generic hot kernels
 */

/* ═══════════════════════════════════════════════════════════════════════════
 * 【标量参考实现】（文件名沿用 qwen-asr 时代的历史命名）
 *
 * 本文件的三个角色：
 * 1. 无 SIMD 平台的回退路径（ds_kernels_impl.h 的 #else 分支直连这里）；
 * 2. NEON/AVX 版本的正确性对照基准——SIMD 版累加顺序不同会有末位 ULP 差异，
 *    回归测试（tools/regress.sh + test_unit）以标量/官方输出为准绳；
 * 3. 非 ARM 平台上函数指针的初始值（ds_kernels.c: ds_bf16_matvec_fused_impl_ptr）。
 *
 * 写法要点（所有平台内核共享的模式）：
 * - BF16→F32 转换只有一条移位：BF16 位模式左移 16 位补零即 F32 位模式
 *   （BF16 指数位与 F32 完全对齐，见 docs/03 §3.2、docs/09 §2.2）；
 * - 用 memcpy 做类型双关而非 *(float*)&bits，避免违反严格别名规则；
 * - 权重永远保持 BF16 存储，转换发生在寄存器里（on-the-fly），
 *   3GB 级解码器权重不翻倍（docs/03 §3.3）。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "ds_kernels_impl.h"

#include <string.h>   /* memcpy：仅做位模式重解释，编译器会优化成寄存器移动 */

/* ds_bf16_matvec_fused_generic - BF16 权重 × F32 输入的融合矩阵向量乘（标量基准）
 * 【算法/数学】y[o] = bias[o] + Σ_{k=0..in_dim-1} W_bf16[o][k]·x[k]
 *   维度：y[out_dim]，x[in_dim]，W_bf16 视作 [out_dim][in_dim] 行主序矩阵。
 *   "fused" = BF16→F32 在线转换 + 点积累加 + bias 加法，一趟完成。
 * 【参数】y 输出；x 输入向量；W_bf16 权重；bias 可为 NULL；in_dim/out_dim 维度。
 * 【为什么】这是分发层所有 SIMD 版本的语义定义；SIMD 版只改指令形式，
 *   逐元素数学完全一致。
 * 【参考】docs/03 §3.2-§3.3、§4.4。
 */
void ds_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim) {
    /* 逐输出行遍历：每行一次点积。行主序下第 o 行起始 = 基址 + o*in_dim */
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim; // (size_t) 防御大矩阵时 int 乘积溢出
        float sum = bias ? bias[o] : 0.0f; // bias 融合进累加器，省一次单独遍历
        for (int k = 0; k < in_dim; k++) {
            // BF16→F32：位模式左移 16 位（指数对齐、尾数低位补零）即合法 F32；
            // 先转 uint32_t：不转的话整型提升到 int，高半区值 <<16 会溢出有符号数（UB）
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float)); // 类型双关：整数位模式重解释为 float（严格别名安全）
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

/* ds_argmax_bf16_range_generic - 行区间 [start,end) 上的流式 argmax（标量基准）
 * 【算法/数学】对 o ∈ [start,end) 计算 y[o] = Σ_k W[o][k]·x[k]（无 bias），
 *   维护 (best, best_val)，只保留当前最大——不物化 logits 数组。
 * 【参数】x[in_dim]；W_bf16 行主序权重；[start,end) 行号区间左闭右开；
 *   best_out/best_val_out 输出；区间为空时 best=start、best_val=-1e30f。
 * 【为什么】LM head 129280×1280（≈331 MB）每步全量扫描，贪心解码只要
 *   最大值；流式省掉 129280×4B 的 logits 缓冲与一轮写读。区间接口供
 *   ds_argmax_worker 按线程分块（ds_kernels.c）。
 * 【参考】docs/03 §3.4；docs/05 §4.1。
 */
void ds_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f; // 哨兵值：远小于任何正常 logit，保证首行必胜

    for (int o = start; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f; // 无 bias：LM head 不带偏置
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        // 严格大于：logit 并列时保留先出现的行号（SIMD 版同样约定，保证可复现）
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

/* ds_dot_f32_generic - F32 点积 Σ a[i]·b[i]（标量基准）。
 * 注意力 QK 得分等小向量（head_dim=128）的基础原语，数据常驻 L1；
 * SIMD 版用多 lane 并行 + 多累加器打破 FMA 依赖链提速，语义不变。 */
float ds_dot_f32_generic(const float *a, const float *b, int n) {
    float sum = 0.0f;
    // 串行依赖链：本次累加必须等上一次的 sum——SIMD 版用多累加器打破它
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ds_vec_scale_inplace_generic - 原地缩放 dst[i] *= scale。
 * 注意力输出按 1/Σexp 归一化使用（ds_kernels.c 的在线 softmax 收尾）。 */
void ds_vec_scale_inplace_generic(float *dst, float scale, int n) {
    // 逐元素读-改-写，元素间零依赖——SIMD 版只是把多个元素并成一拍处理
    for (int i = 0; i < n; i++) dst[i] *= scale;
}

/* ds_vec_axpy_inplace_generic - 原地 axpy：dst[i] += alpha·src[i]。
 * 注意力在线 softmax 中把新 V 行按权重 exp(score-max) 累入输出。 */
void ds_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    // axpy = alpha·x plus y（BLAS 命名传统）：每元素一次乘、一次加
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

/* ds_vec_scale_add_generic - 原地融合 dst[i] = dst[i]·correction + src[i]。
 * 在线 softmax 遇到新最大值时：旧部分和按 correction = exp(m_old-m_new)
 * 重缩放后并入新 V 行——"缩放旧和 + 加新和"一次遍历完成（见 ds_kernels.c
 * 的 attention 主循环）。 */
void ds_vec_scale_add_generic(float *dst, const float *src, float correction, int n) {
    // 本式恰为一个 FMA 的形状（dst·correction + src）：SIMD 版直接映射成
    // 一条融合乘加指令（vfmaq_f32 / _mm*_fmadd_ps），一次完成乘与加、中间不舍入
    for (int i = 0; i < n; i++) dst[i] = dst[i] * correction + src[i];
}
