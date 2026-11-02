/*
 * qwen_asr_kernels_neon.c - ARM NEON hot kernels
 */

#include "ds_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * 【ARM NEON 热内核】（文件名沿用 qwen-asr 时代的历史命名）
 *
 * 两条 BF16 matvec 路径，运行时二选一（ds_kernels.c: ds_kernels_dispatch_init）：
 * 1. vshll 路径（本文件上半部，所有 ARMv8 可用，M1 的默认）：
 *    每条 vshll_n_u16 把 4 个 u16 加宽成 u32 并左移 16 位——一条指令完成
 *    BF16→F32 位模式转换（docs/09 §2.2 移位原理的指令级体现），
 *    再 vfmaq_f32 乘加：每 4 个 BF16 MAC 需 2 条指令。
 * 2. bfdot 路径（本文件下半部，需 ARMv8.6-A FEAT_BF16，Apple M2+）：
 *    一条 bfdot 直接完成 8 个 BF16 MAC → 4 个 F32 累加，指令密度 4×；
 *    代价是两个操作数都必须是 BF16，输入 x（F32）要先压窄。
 *
 * NEON 基础（128 位寄存器 = 16 字节）：
 * - float32x4_t：4 个 F32 lane；uint16x8_t：8 个 u16 lane；lane 编号从低到高
 *   对应内存地址从低到高（小端序下 vld1q_u16 的 lane0 = src[0]）；
 * - 所有 vld1q/vst1q 都不做对齐检查（等价 x86 的 loadu/storeu），本文件
 *   对指针没有任何对齐假设——mmap 的 BF16 权重行首只需 2 字节自然对齐；
 * - ARM64 有 32 个 v 寄存器：主循环按"8 个 x 向量 + 8 个累加器 + 4 个行
 *   加载寄存器 + 少量临时"设计，恰好塞满寄存器堆、无溢出（spill）。
 *
 * 【数值一致性】SIMD 水平求和的顺序与标量不同，结果会有末位 ULP 差异；
 * 本工程用回归门控（30 组冻结标签 + CER 阈值）保证分派切换不改变 OCR
 * 输出（docs/05 §10.2"逐位一致优先"）。
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* ========================================================================
 * NEON-optimized BF16 → F32 batch conversion
 * ======================================================================== */

/* ds_bf16_to_f32_neon - 批量 BF16→F32 转换（编码器权重预转换等场景）
 * 【算法/数学】dst[i] = bitcast_f32(src[i] << 16)，逐元素位模式扩展。
 * 【参数】dst[n] 输出 F32；src[n] 输入 BF16（uint16 位模式）；n 元素数。
 * 【为什么】编码器权重较小、按 F32 加载更利于批量 sgemm（docs/03 §2.4
 *   的分层策略）；转换本身零乘加，纯移位 + 重解释。
 * 【参考】docs/09 §2.2。
 */
void ds_bf16_to_f32_neon(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {          // 主循环：8 元素/次，128 位满载
        uint16x8_t bf = vld1q_u16(src + i); // 一次加载 8 个 BF16（lane0=src[i]…lane7=src[i+7]）
        // vget_low/high_u16：把 128 位寄存器拆成低/高 64 位半区（各 4 个 u16 lane）
        // vshll_n_u16(v,16)：shift-left-long——每条 u16 lane 零扩展成 u32 再左移
        //                    16 位，一条指令融合"加宽 + 移位"，得到 F32 位模式
        uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
        // vreinterpretq：纯类型重解释，编译期零指令（位模式按 F32 解读）
        vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i + 4 <= n; i += 4) {          // 次级循环：4 元素/次（消化 n%8 的 4..7 段）
        uint16x4_t bf = vld1_u16(src + i); // 64 位加载 4 个 BF16
        uint32x4_t f32 = vshll_n_u16(bf, 16); // 4 条 lane 加宽到 128 位 u32
        vst1q_f32(dst + i, vreinterpretq_f32_u32(f32));
    }
    for (; i < n; i++) {                  // 标量兜底：n%4 的 <4 个尾巴
        uint32_t u = ((uint32_t)src[i]) << 16;
        memcpy(dst + i, &u, sizeof(float));
    }
}

/* ========================================================================
 * NEON-optimized F32 → BF16 conversion
 * ======================================================================== */

/* ds_f32_to_bf16_neon - 批量 F32→BF16 转换（带舍入）
 * 【算法/数学】dst[i] = round_to_bf16(src[i])：位模式加 0x8000（被丢弃的
 *   低 16 位的中点 = 半个 BF16 尾数 ULP）再右移 16 位截断——靠进位传播
 *   实现就近舍入，全程无乘除。
 * 【参数】dst[n] 输出 BF16；src[n] 输入 F32；n 元素数。
 * 【舍入细节】现有注释称 "round to nearest even"：加 0x8000 后截半在一般
 *   情况下等价于就近舍入，但严格 IEEE RNE 在"被丢弃部分恰为 0x8000"
 *   （精确半途）时应向偶数舍入，本实现恒进位（向远离零方向）。
 *   差异仅出现在精确半途点、幅度半个 ULP，工程上可接受；下半部 bfdot
 *   路径用的硬件 BFCVT 指令才是严格 RNE（docs/09 §2.2 亦采用"加 0x8000
 *   再截半"口径，与本实现一致）。
 * 【参考】docs/09 §2.2。
 */
void ds_f32_to_bf16_neon(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {              // 主循环：8 元素/次（两条 128 位 F32 载入）
        float32x4_t v0 = vld1q_f32(src + i);    // 低 4 个 F32
        float32x4_t v1 = vld1q_f32(src + i + 4); // 高 4 个 F32
        /* Round to nearest even: add bias then shift right 16 */
        // 把 F32 位模式当 u32 处理，加 0x8000 让进位把"四舍五入"带进高 16 位
        uint32x4_t b0 = vaddq_u32(vreinterpretq_u32_f32(v0), vdupq_n_u32(0x8000));
        uint32x4_t b1 = vaddq_u32(vreinterpretq_u32_f32(v1), vdupq_n_u32(0x8000));
        // vshrn_n_u32：narrowing shift right——u32 右移 16 位同时收窄成 u16（一条指令两件事）
        uint16x4_t r0 = vshrn_n_u32(b0, 16);
        uint16x4_t r1 = vshrn_n_u32(b1, 16);
        vst1q_u16(dst + i, vcombine_u16(r0, r1)); // 两个 64 位半区拼回 128 位存储
    }
    for (; i < n; i++) {                      // 标量兜底：n%8 的尾巴
        uint32_t u;
        memcpy(&u, src + i, sizeof(u));
        u = (u + 0x8000) & 0xFFFF0000u; // 加半 ULP 再截低 16 位（与 NEON 路径逐位一致）
        dst[i] = (uint16_t)(u >> 16);
    }
}

/* ds_bf16_matvec_fused_neon - BF16 matvec 的 NEON vshll+vfma 实现（M1 默认路径）
 * 【算法/数学】同 generic：y[o] = bias[o] + Σ_k W[o][k]·x[k]。
 * 【参数】y[out_dim] 输出；x[in_dim] F32 输入；W_bf16 [out_dim][in_dim] 行主序
 *   BF16 权重；bias 可为 NULL；in_dim/out_dim 维度。
 * 【为什么】相对标量版的收益：
 *   - 4 lane 并行 FMA：每条 vfmaq_f32 完成 4 个乘加；
 *   - BF16→F32 转换由 vshll 在寄存器内完成，不落内存；
 *   - 每行 4 个独立累加器打破 FMA 依赖链（Apple Firestorm FMA 延迟约 4 拍，
 *     单链串行会停等），每累加器每轮 2 次 FMA、4 条链交替 → 足够 ILP；
 *   - 外层一次处理 2 个输出行：x 的 8 个 128 位载入被两行复用，
 *     访存与计算交错便于乱序执行重叠。
 * 【参考】docs/09 §6.2；docs/03 §4.4。
 */
void ds_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time, 32 elements/iter, 8 accumulators */
    // 累加器→元素映射（每行，k 为本轮迭代基址，共 32 元素）：
    //   a0: 元素 k+0..3   与 k+16..19      a1: k+4..7   与 k+20..23
    //   a2: k+8..11      与 k+24..27      a3: k+12..15 与 k+28..31
    //   即每个累加器每轮吃 2 个 vshll+vfma 对，行内 4 条独立依赖链
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;       // 第 o   行行首
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim; // 第 o+1 行行首
        float s0 = bias ? bias[o] : 0.0f;     // bias 先入累加器（fused 的一部分）
        float s1 = bias ? bias[o + 1] : 0.0f;

        // 8 个累加器清零：vdupq_n_f32 把标量广播到 4 条 lane
        // a0-a3 归行 o，b0-b3 归行 o+1
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) { // 主循环：32 元素/次（128 位 × 8 拍满载）
            // x 只载一次、两行共用（这是"一次处理 2 行"的主要收益）
            float32x4_t x0 = vld1q_f32(x + k);      // x[k+0..3]
            float32x4_t x1 = vld1q_f32(x + k + 4);  // x[k+4..7]
            float32x4_t x2 = vld1q_f32(x + k + 8);  // x[k+8..11]
            float32x4_t x3 = vld1q_f32(x + k + 12); // x[k+12..15]
            float32x4_t x4 = vld1q_f32(x + k + 16); // x[k+16..19]
            float32x4_t x5 = vld1q_f32(x + k + 20); // x[k+20..23]
            float32x4_t x6 = vld1q_f32(x + k + 24); // x[k+24..27]
            float32x4_t x7 = vld1q_f32(x + k + 28); // x[k+28..31]

            // 每行 4 条 u16x8 加载覆盖 32 个 BF16；下方 vfma 序列即上面的累加器映射
            uint16x8_t r0a = vld1q_u16(w0 + k);      // 行0 元素 k+0..7
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);  // 行0 元素 k+8..15
            uint16x8_t r0c = vld1q_u16(w0 + k + 16); // 行0 元素 k+16..23
            uint16x8_t r0d = vld1q_u16(w0 + k + 24); // 行0 元素 k+24..31
            // 标准三连：vget_low/high 取半区 → vshll 左移 16 位成 F32 位模式
            //          → vfmaq_f32(acc, m1, m2) = acc + m1*m2（融合乘加，中间不舍入）
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            // 后 16 个元素流回 a0-a3：依赖链被 3 条其他 FMA 隔开，隐藏 FMA 延迟
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);      // 行1 同构：b0-b3 与 a0-a3 共享 x0-x7
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
        for (; k + 8 <= in_dim; k += 8) { // 次级循环：8 元素/次（消化 in_dim%32 的余数）
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), x1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), x1);
        }
        // 水平归约：先 vaddq 两两配对成加树（求和顺序确定、可复现），
        // 再 vaddvq_f32 一步把 4 条 lane 收成一个标量
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) { // 标量兜底：in_dim%8 的 <8 个尾巴（与 generic 同式）
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
    // out_dim 为奇数时剩下的最后一行：结构同上但只有一行（in_dim 仍是
    // 8 元素主循环 + 标量尾巴的两级分块）
    for (; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

        // 单行只需 2 个累加器（8 元素/轮）：每累加器每轮 1 次 FMA，
        // 两链交替已足够隐藏延迟，寄存器压力也更小
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k); // 8 个 BF16：低 4 进 acc0、高 4 进 acc1
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1)); // 加树归约 + 水平求和

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

/* ds_argmax_bf16_range_neon - 行区间 [start,end) 流式 argmax 的 NEON vshll 版
 * 【算法/数学】同 generic 版：区间内逐行点积、边算边比；无 bias（LM head 无偏置）。
 * 【参数】同 generic；[start,end) 是本线程分到的词表行区间（左闭右开）。
 * 【为什么】LM head 129280×1280（≈331 MB）是 decode 每步单笔最大的内存
 *   事务（docs/09 §1.1），NEON 指令级加速直接缩短每行点积时间。
 *   matvec 内部结构与 ds_bf16_matvec_fused_neon 完全同构（2 行 × 32 元素
 *   × 8 累加器），intrinsic 逐条语义见该函数内注释，此处不重复。
 * 【参考】docs/03 §3.4；docs/05 §4.1。
 */
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

        // 严格大于：logit 并列时保留先出现的行号（与 generic/AVX 版约定一致）
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

/* ds_dot_f32_neon - F32 点积的 NEON 实现
 * 【算法/数学】Σ a[i]·b[i]；主循环 8 元素/次（2×128 位），尾巴标量。
 * 【参数】a/b[n] 输入；n 元素数；返回点积。
 * 【为什么】2 个独立累加器让两条 vfma 链交替执行，避免"下一条乘加依赖
 *   上一条结果"的停等；vaddvq_f32 是 ARMv8 专有的 4-lane 水平求和
 *   （x86 上需要多条 hadd/shuffle 才能等价）。
 * 【用途】注意力 QK 得分（head_dim=128，L1 常驻）等小向量点积。
 */
float ds_dot_f32_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f); // 广播 0.0 到 4 lane 作累加器
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {          // 8 元素/次：acc0 吃前 4、acc1 吃后 4
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);   // acc0 += a0*b0（逐 lane 独立乘加）
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1)); // 加树归约后水平求和
    for (; i < n; i++) sum += a[i] * b[i]; // 标量兜底：n%8 尾巴
    return sum;
}

/* ds_vec_scale_inplace_neon - 原地缩放 dst[i] *= scale 的 NEON 实现
 * 【参数】dst[n] 原地修改；scale 缩放系数；n 元素数。
 * 【为什么】纯流式访存（每元素 1 读 1 写），8 元素/次减少循环开销；
 *   用于注意力输出按 1/Σexp 归一化（在线 softmax 收尾）。
 */
void ds_vec_scale_inplace_neon(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale); // 系数广播一次，循环内不再重播
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        // 写成"加零 FMA"：vfmaq(0, d, s) = d*s，数值等价于 vmulq_f32
        // （+0.0 不引入额外舍入），与相邻内核保持统一的 FMA 代码风格
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++) dst[i] *= scale; // 标量兜底
}

/* ds_vec_axpy_inplace_neon - 原地 axpy dst[i] += alpha·src[i] 的 NEON 实现
 * 【参数】dst[n] 原地修改；src[n] 输入；alpha 标量系数；n 元素数。
 * 【为什么】一条 vfmaq 完成 4 个元素的乘加，融合写回省一次中间寄存；
 *   用于在线 softmax 把新 V 行按权重 exp(score-max) 累入注意力输出。
 */
void ds_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        // vfmaq_f32(acc, m1, m2) 参数顺序：第一个参数是加数（累加器），
        // 后两个相乘 —— 这里即 dst + src*alpha
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

/* ds_vec_scale_add_neon - 原地融合 dst[i] = dst[i]·correction + src[i] 的 NEON 实现
 * 【参数】dst[n] 原地修改；src[n] 加数向量；correction 缩放系数；n 元素数。
 * 【为什么】在线 softmax 遇到新最大值时"旧和重缩放 + 新和并入"合一：
 *   correction = exp(m_old - m_new)，一次遍历同时完成缩放与累加。
 */
void ds_vec_scale_add_neon(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        // vfmaq(s0, d0, c) = s0 + d0*c，即 dst*correction + src
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

/* 【生效条件】本段仅在编译器同时定义 __ARM_NEON 与
 * __ARM_FEATURE_BF16_VECTOR_ARITHMETIC（-mcpu 指到 ARMv8.6-A 及以上，
 * 如 Apple M2+）时编入二进制；运行时探测到 FEAT_BF16 后（sysctl/
 * getauxval，见 ds_kernels.c）函数指针才切到这里，M1 等无 FEAT_BF16 的
 * 芯片保持 vshll 路径。
 * 两个 #include 重复出现是因为上下两半的编译条件互相独立。 */

#include <arm_neon.h>
#include <string.h>

/* BF16 dot product: acc = acc + dot(a_bf16, b_bf16), 8 BF16 → 4 F32 */
/* bfdot_f32 - 单条 bfdot 指令的封装
 * 【算法/数学】对 4 个 32 位 lane 各做：acc[j] += a[2j]·b[2j] + a[2j+1]·b[2j+1]
 *   —— 8 个 BF16 MAC 归并进 4 个 F32 累加器（BF16 lane 对 2j/2j+1 → F32 lane j）。
 * 【参数】acc：4-lane F32 累加器（读-写）；a/b：8-lane BF16 操作数。
 * 【为什么】用内联汇编而非 intrinsic：Apple clang 15 未提供 bfloat16 版的
 *   vdotq 内建；"bfdot %0.4s, %1.8h, %2.8h" 中 4s/8h 标注向量视图，
 *   "+w" 约束表示 acc 同时是输入与输出（read-write 的 w 区向量寄存器）。
 * 【参考】docs/09 §6.3。
 */
static inline float32x4_t bfdot_f32(float32x4_t acc, bfloat16x8_t a, bfloat16x8_t b) {
    __asm__("bfdot %0.4s, %1.8h, %2.8h" : "+w"(acc) : "w"(a), "w"(b));
    return acc;
}

/* Convert 2x F32x4 vectors to 1x BF16x8 (round to nearest even) */
/* f32x8_to_bf16x8 - 两个 F32x4 拼成 1 个 BF16x8（硬件舍入）
 * 【算法/数学】vcvtq_low_bf16_f32 转换 lo 的 4 条 lane，vcvtq_high_bf16_f32
 *   再把 hi 的 4 条 lane 填进结果高半区——底层是 ARMv8.2 的 BFCVT 指令，
 *   严格 round-to-nearest-even（与上半部"加 0x8000"软件法在精确半途点
 *   有细微差异，见该处注释）。
 * 【为什么】bfdot 的两个操作数都必须是 BF16：F32 输入 x 每个数据块压窄
 *   一次即可，权重保持文件位模式、完全免转换；注意这使 x 带 BF16 尾数
 *   （7 位）的量化误差（≈0.4% 相对误差），与 vshll 路径的"全精度 x"
 *   理论上不等价——两条路径的切换由回归门控背书（docs/05 §10.2）。
 */
static inline bfloat16x8_t f32x8_to_bf16x8(float32x4_t lo, float32x4_t hi) {
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(lo), hi);
}

/* BP - 把 uint16_t* 重解释为 const __bf16*（vld1q_bf16 要求 bfloat16 指针类型）。
 * 经 (const void*) 中转完成类型双关：位模式不变、零运行时开销，
 * 权重从 mmap 到寄存器全程保持 BF16 存储格式。 */
#define BP(p) ((const __bf16 *)(const void *)(p))

/* ds_bf16_matvec_fused_neon_bf16 - BF16 matvec 的 bfdot 硬件点积实现（M2+）
 * 【算法/数学】同 vshll 版：y[o] = bias[o] + Σ_k W[o][k]·x[k]。
 * 【参数】同 generic。
 * 【为什么】相对 vshll 路径的收益：每 8 个 BF16 MAC 从 2 条指令
 *   （vshll+vfma）减到 1 条 bfdot，指令密度 4×（docs/09 §6.3）；
 *   x 每 32 元素块压窄成 4 个 BF16x8（xb0..xb3），两行复用同一份 xb，
 *   转换开销摊薄一半；权重直接按 BF16 位模式加载，无 F32 展开寄存器。
 * 【参考】docs/09 §6.3。
 */
void ds_bf16_matvec_fused_neon_bf16(float *y, const float *x, const uint16_t *W_bf16,
                                      const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* 2 rows at a time, 32 BF16 elements/iter, 4 dot products per row */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

        // 4 个 F32x4 累加器/行：每条 bfdot 恰好填满 4 lane，各管 8 元素
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
        float32x4_t b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            // x 每 8 元素压窄一次 BF16（共 4 份覆盖 32 元素），两行复用——
            // bfdot 要求双 BF16 操作数，这是本路径唯一的转换开销
            bfloat16x8_t xb0 = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            bfloat16x8_t xb1 = f32x8_to_bf16x8(vld1q_f32(x+k+8), vld1q_f32(x+k+12));
            bfloat16x8_t xb2 = f32x8_to_bf16x8(vld1q_f32(x+k+16), vld1q_f32(x+k+20));
            bfloat16x8_t xb3 = f32x8_to_bf16x8(vld1q_f32(x+k+24), vld1q_f32(x+k+28));

            // 权重保持 BF16 位模式直接加载：一条 bfdot = 8 MAC → 4 lane F32 累加
            a0 = bfdot_f32(a0, vld1q_bf16(BP(w0+k)),    xb0);
            a1 = bfdot_f32(a1, vld1q_bf16(BP(w0+k+8)),  xb1);
            a2 = bfdot_f32(a2, vld1q_bf16(BP(w0+k+16)), xb2);
            a3 = bfdot_f32(a3, vld1q_bf16(BP(w0+k+24)), xb3);

            b0 = bfdot_f32(b0, vld1q_bf16(BP(w1+k)),    xb0); // 行1 复用同一批 xb
            b1 = bfdot_f32(b1, vld1q_bf16(BP(w1+k+8)),  xb1);
            b2 = bfdot_f32(b2, vld1q_bf16(BP(w1+k+16)), xb2);
            b3 = bfdot_f32(b3, vld1q_bf16(BP(w1+k+24)), xb3);
        }
        for (; k + 8 <= in_dim; k += 8) { // 次级循环：8 元素/次（in_dim%32 余数）
            bfloat16x8_t xb = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            a0 = bfdot_f32(a0, vld1q_bf16(BP(w0+k)), xb);
            b0 = bfdot_f32(b0, vld1q_bf16(BP(w1+k)), xb);
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))); // 加树归约
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3)));

        for (; k < in_dim; k++) { // 标量兜底：in_dim%8 尾巴（<<16 位转换，同 generic）
            uint32_t u0 = ((uint32_t)w0[k]) << 16, u1 = ((uint32_t)w1[k]) << 16;
            float v0, v1; memcpy(&v0, &u0, 4); memcpy(&v1, &u1, 4);
            s0 += v0 * x[k]; s1 += v1 * x[k];
        }
        y[o] = s0; y[o+1] = s1;
    }

    /* Single row */
    // out_dim 为奇数时的最后一行：16 元素主循环 + 8 元素次级 + 标量尾巴
    for (; o < out_dim; o++) {
        const uint16_t *wr = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;
        float32x4_t aa = vdupq_n_f32(0), ab = vdupq_n_f32(0);
        for (; k + 16 <= in_dim; k += 16) { // 主循环：16 元素/次（2 个 BF16x8）
            bfloat16x8_t xb0 = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            bfloat16x8_t xb1 = f32x8_to_bf16x8(vld1q_f32(x+k+8), vld1q_f32(x+k+12));
            aa = bfdot_f32(aa, vld1q_bf16(BP(wr+k)),   xb0);
            ab = bfdot_f32(ab, vld1q_bf16(BP(wr+k+8)), xb1);
        }
        sum += vaddvq_f32(vaddq_f32(aa, ab));
        for (; k + 8 <= in_dim; k += 8) { // 次级：每 8 元素独立成点积再水平求和
            // 单条链不值得常驻累加器，直接零累加器起步、算完即归约
            bfloat16x8_t xb = f32x8_to_bf16x8(vld1q_f32(x+k), vld1q_f32(x+k+4));
            float32x4_t tmp = bfdot_f32(vdupq_n_f32(0), vld1q_bf16(BP(wr+k)), xb);
            sum += vaddvq_f32(tmp);
        }
        for (; k < in_dim; k++) { // 标量兜底
            uint32_t u = ((uint32_t)wr[k]) << 16; float v; memcpy(&v, &u, 4);
            sum += v * x[k];
        }
        y[o] = sum;
    }
}

/* 第一个函数用完即取消 BP：预处理器不认 C 作用域，只能手动 #undef 控制宏的
 * 可见范围（第二个函数体内会重新定义一份） */
#undef BP

/* ds_argmax_bf16_range_neon_bf16 - 行区间流式 argmax 的 bfdot 实现（M2+）
 * 【算法/数学/参数】同 ds_argmax_bf16_range_neon（vshll 版），点积换成
 *   bfdot；区间由 ds_argmax_worker 按线程切分（左闭右开）。
 * 【为什么】LM head 每步 331 MB 扫描是全引擎指令密度收益最大的循环；
 *   内部结构与 ds_bf16_matvec_fused_neon_bf16 同构，不再逐条注释。
 * 【参考】docs/03 §3.4；docs/05 §4.1；docs/09 §6.3。
 */
void ds_argmax_bf16_range_neon_bf16(const float *x, const uint16_t *W_bf16,
                                       int in_dim, int start, int end,
                                       int *best_out, float *best_val_out) {
    int best = start; float best_val = -1e30f; int o = start;
    #define BP(p) ((const __bf16 *)(const void *)(p)) // 函数体内重定义（前一函数已 #undef）

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
        // 严格大于：并列时保留先出现的行号（与其他平台版本约定一致）
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
    #undef BP // 同样用完即取消，宏不外泄到文件其他位置
    *best_out = best; *best_val_out = best_val;
}

#endif /* __ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC */
