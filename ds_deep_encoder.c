/*
 * ds_deep_encoder.c - Encoders for DeepSeek-OCR
 * ds_deep_encoder.c — DeepSeek-OCR 编码器实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【两种编码器的完整实现】
 * ─────────────────────────────────────────────────────────────────────
 *
 * V1/V3 — CLIP ViT-L/14 编码器:
 *   ┌─────────────────────────────────────────────────────┐
 *   │ 输入: SAM视觉特征 [256, 1024]                        │
 *   │ (V3模式: CLIP绕过patch_embed，直接接收SAM特征)       │
 *   │   ↓ + CLS token → [257, 1024]                       │
 *   │   ↓ + 位置编码 (可能需要双三次插值)                  │
 *   │   ↓ Pre-LayerNorm                                   │
 *   │   ↓ 24层标准ViT:                                     │
 *   │     LayerNorm → MHA(16头, 64维) → 残差               │
 *   │     LayerNorm → GELU FFN(4096) → 残差               │
 *   │   ↓ 移除CLS token → [256, 1024]                     │
 *   │   ↓ Concat(SAM特征, CLIP特征) → [256, 2048]         │
 *   │   ↓ Projector(2048→1280) → [256, 1280]              │
 *   │   ↓ + image_newline(每16个token后插入) + view_sep    │
 *   │ 输出: 273 tokens [273, 1280]                         │
 *   │   (16行×(16+1) + 1 = 273)                            │
 *   └─────────────────────────────────────────────────────┘
 *
 * V2 — DeepEncoder V2 (Qwen2-0.5B 架构):
 *   ┌─────────────────────────────────────────────────────┐
 *   │ 输入: [visual_tokens, causal_queries]               │
 *   │   = [256, 896] + [256, 896] → [512, 896]           │
 *   │   ↓ 24层 Transformer (混合注意力):                   │
 *   │     RMSNorm → GQA(14Q头/2KV头, RoPE) → 残差         │
 *   │     RMSNorm → SwiGLU FFN(4864) → 残差              │
 *   │     视觉token: 双向注意力(可互相看到)                │
 *   │     因果查询: 因果注意力(只看自己和之前)             │
 *   │   ↓ 最终 RMSNorm                                    │
 *   │   ↓ 取因果查询的输出 → [256, 896]                   │
 *   │   ↓ Projector(896→1280) → [256, 1280]              │
 *   │   ↓ + view_seperator                               │
 *   │ 输出: 257 tokens [257, 1280]                         │
 *   └─────────────────────────────────────────────────────┘
 *
 * 【CLIP位置编码插值】
 * CLIP预训练位置编码为257个(1 CLS + 16×16=256 patches)
 * 当输入正好256个patch时完美匹配，无需插值
 * 否则需要双三次插值: [16×16] → [tgt_h × tgt_w]
 *
 * 【image_newline 的2D网格布局】(V1/V3)
 * 256个视觉token按16×16网格排列，每行末尾插入1个newline token:
 *   Row 0: [patch_0..15, newline]   ← 17 tokens
 *   Row 1: [patch_16..31, newline]  ← 17 tokens
 *   ...
 *   Row 15: [patch_240..255, newline] ← 17 tokens
 *   Last: [view_seperator]           ← 1 token
 *   Total: 16×17 + 1 = 273 tokens
 * newline告诉解码器图像空间结构的行边界
 * ═══════════════════════════════════════════════════════════════════════
 *
 * V1: CLIP ViT-L/14 → takes SAM patch_embeds → CLIP output + SAM concat → projector
 * V2: Qwen2-0.5B based encoder with causal flow queries
 */

#include "ds_deep_encoder.h"
#include <pthread.h>
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include "ds_dump.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ========================================================================
 * CLIP ViT-L/14 Forward Pass (V1)
 * ======================================================================== */

/* clip_layer_forward - CLIP ViT 单层前向（Pre-LN 结构）
 * 【算法】out = x + OutProj(MHA(LN1(x)))
 *         out = out + FFN(LN2(out))                     （两次残差）
 *   维度: x [seq,1024] → LN1 → 融合 QKV 投影 [seq,3072] → 拆 Q/K/V 各 [seq,1024]
 *         → 16 头双向注意力（head_dim=64, scale=1/√64=0.125）→ out_proj [seq,1024]
 *         → +x → LN2 → fc1 [seq,4096] → GELU(erf 精确式) → fc2 [seq,1024] → +残差
 * 【为什么双向注意力】CLIP 是视觉语义精炼器：一张图内的 patch 没有"过去/未来"，
 *   每个 patch 必须看到全图才能聚合语义——与解码器语言侧的因果掩码刻意相反。
 *   （docs/06-视觉编码器篇.md §5）
 * 【为什么 Pre-LN】归一化放在残差分支内部（x + Attn(LN(x))），与 HF CLIP 的
 *   layer_norm1 → self_attn → 残差顺序逐位对齐；eps=1e-6 也是 CLIP 的约定。
 * 【参数】out/x: [seq_len,1024]（out 可与 x 不同缓冲）；layer_idx: 0..23；
 *   seq_len: 257（1 CLS + 256 patch，1024² 输入）
 * 【参考】PyTorch 等价物: HF transformers 的 CLIPEncoderLayer
 */
static void clip_layer_forward(float *out, const float *x,
                                ds_clip_encoder_t *clip, int layer_idx,
                                int seq_len) {
    int dim = DS_CLIP_HIDDEN;      /* 1024 */
    int n_heads = DS_CLIP_HEADS;    /* 16 */
    int head_dim = DS_CLIP_HEAD_DIM; /* 64 */
    int mlp_dim = DS_CLIP_MLP_DIM;  /* 4096 */
    /* 1/√head_dim：点积方差随 head_dim 线性增长（Var(q·k)=d·σq²·σk²），
     * 除以 √64 把 logit 方差归一到 O(1)，防止 softmax 饱和成 one-hot（docs/13 §1.5） */
    float scale = 1.0f / sqrtf((float)head_dim);

    /* 本层权重指针速查（checkpoint 张量名前缀 model.vision_model.transformer.layers.N.，
     * 加载见 ds_ocr.c 的 CLIP_WEIGHT 宏；线性权重形状按 [out, in] 行主序）:
     *   layer_norm1.weight/bias  [1024]                      Pre-LN1 的 γ/β
     *   self_attn.qkv_proj.weight [3072,1024] + bias [3072]  融合 QKV（行内 [Q|K|V] 各 1024）
     *   self_attn.out_proj.weight [1024,1024] + bias [1024]  注意力输出投影
     *   layer_norm2.weight/bias  [1024]                      Pre-LN2 的 γ/β
     *   mlp.fc1.weight [4096,1024] + bias [4096]             FFN 升维
     *   mlp.fc2.weight [1024,4096] + bias [1024]             FFN 降维
     */
    float *ln1_w = clip->layers[layer_idx].layer_norm1_weight;
    float *ln1_b = clip->layers[layer_idx].layer_norm1_bias;
    float *qkv_w = clip->layers[layer_idx].qkv_proj_weight;
    float *qkv_b = clip->layers[layer_idx].qkv_proj_bias;
    float *out_w = clip->layers[layer_idx].out_proj_weight;
    float *out_b = clip->layers[layer_idx].out_proj_bias;
    float *ln2_w = clip->layers[layer_idx].layer_norm2_weight;
    float *ln2_b = clip->layers[layer_idx].layer_norm2_bias;
    float *fc1_w = clip->layers[layer_idx].mlp_fc1_weight;
    float *fc1_b = clip->layers[layer_idx].mlp_fc1_bias;
    float *fc2_w = clip->layers[layer_idx].mlp_fc2_weight;
    float *fc2_b = clip->layers[layer_idx].mlp_fc2_bias;

    /* Pre-LayerNorm1
     * Pre-LN：norm 在残差分支内部，残差主路 x 本身不过 norm（梯度通路干净） */
    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    ds_layer_norm(x_norm, x, ln1_w, ln1_b, seq_len, dim, 1e-6f);

    /* Fused QKV: [seq, dim] @ [dim, 3*dim]
     * 三份投影融成一次 GEMM（Q/K/V 权重按行拼接为 [3*dim, dim]），比三次独立
     * matmul 少两遍 x 的读取；每行布局为 [Q(1024) | K(1024) | V(1024)] */
    float *qkv = (float *)malloc(seq_len * 3 * dim * sizeof(float));
    ds_linear(qkv, x_norm, qkv_w, qkv_b, seq_len, dim, 3 * dim);

    /* 把融合行的 [Q|K|V] 三段切片拆回三个连续缓冲，方便后续按头取切片 */
    float *Q = (float *)malloc(seq_len * dim * sizeof(float));
    float *K = (float *)malloc(seq_len * dim * sizeof(float));
    float *V = (float *)malloc(seq_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        memcpy(Q + s * dim, qkv + s * 3 * dim, dim * sizeof(float));
        memcpy(K + s * dim, qkv + s * 3 * dim + dim, dim * sizeof(float));
        memcpy(V + s * dim, qkv + s * 3 * dim + 2 * dim, dim * sizeof(float));
    }
    free(qkv);

    /* Bidirectional attention (CLIP uses full attention)
     * 双向注意力的掩码即"全 1 可见"：若用 0/1 掩码表达，每个 query 对全部
     * 257 个 KV 的掩码位都是 1（没有禁看位置）；换成加性掩码则全 0（没有
     * 任何 -inf 位置），softmax 分母含全部 257 个 token——图像 patch 没有
     * "未来"可言，谁也不该被遮挡。与因果版本（ds_causal_attention_aligned）
     * 的差别只在掩码，点积/softmax 数学相同 */
    float *attn_out = (float *)malloc(seq_len * dim * sizeof(float));
    ds_bidirectional_attention(attn_out, Q, K, V, seq_len, n_heads, head_dim, scale);
    free(Q); free(K); free(V);

    /* Output projection + residual
     * out_proj 把 16 个头各自的输出重新混合回单条 1024 维残差流 */
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(proj_out, attn_out, out_w, out_b, seq_len, dim, dim);
    free(attn_out);

    /* 残差一：注意力分支输出加回主路 */
    for (int i = 0; i < seq_len * dim; i++) out[i] = x[i] + proj_out[i];
    free(proj_out);

    /* Pre-LayerNorm2 + MLP (GELU)
     * CLIP 用 erf 精确式 GELU（不是 tanh 近似，更不是 SwiGLU）——与 HF CLIP 对齐 */
    ds_layer_norm(x_norm, out, ln2_w, ln2_b, seq_len, dim, 1e-6f);

    float *fc1 = (float *)malloc(seq_len * mlp_dim * sizeof(float));
    ds_linear(fc1, x_norm, fc1_w, fc1_b, seq_len, dim, mlp_dim);
    ds_gelu(fc1, seq_len * mlp_dim);

    float *fc2 = (float *)malloc(seq_len * dim * sizeof(float));
    ds_linear(fc2, fc1, fc2_w, fc2_b, seq_len, mlp_dim, dim);

    /* 残差二：FFN 分支输出加回主路（in-place 累加，out 即本层最终输出） */
    for (int i = 0; i < seq_len * dim; i++) out[i] += fc2[i];

    free(x_norm); free(fc1); free(fc2);
}

/* Full CLIP encoder forward pass —— 下方 ds_clip_encoder_forward() 的契约说明
 * （该函数体定义在 ds_cubic_kernel / ds_bicubic_resample_dim 两个插值辅助之后）
 *
 * For Unlimited-OCR (V3): patch_embeds is SAM's final output [1024, h_sam, w_sam] spatial,
 *   CLIP skips its patch_embedding Conv2d, directly flatten+transpose → [N, 1024]
 * For V1: patch_embeds is SAM's patch_embed [768, h, w], CLIP applies its Conv2d → [N, 1024]
 *   （注: 当前 C 实现 V1/V3/V4 统一走"跳过 Conv2d 直喂 SAM 特征"路径，见函数体）
 *
 * sam_features: SAM's downsampled output [n_sam_tokens, 1024] (for feature fusion with CLIP output)
 *
 * 【参数】ctx: 全局上下文（clip 权重 + projector + config）；
 *   sam_features: [n_sam_tokens, 1024] 行主序 token 特征；rgb_* 参数未使用；
 *   out_seq_len: [输出] 最终 token 数（含 newline/view_sep）
 * 【返回】[out_seq_len, 1280] 行主序（调用者 free），失败/空权重时可能返回
 *   部分降级结果（见函数内 concat/projector 的 fallback 分支）
 * 【为什么位置编码要插值】CLIP 位置表按 16×16=256 patch 训练（共 257 行含 CLS），
 *   输入 patch 数不是 256 时必须把 16×16 网格重采样成目标网格——早期实现
 *   "直接抄前 total_len 行"曾导致孤儿 HTML 标签事故（docs/06 §6.3 事故一）。
 */
/* ds_cubic_kernel - Keys 三次卷积核（Catmull-Rom 族），一维权重函数
 * 【数学】W(x) = ((A+2)|x| − (A+3))|x|² + 1          |x| ≤ 1
 *              = A(|x|³ − 5|x|² + 8|x| − 4)          1 < |x| < 2
 *              = 0                                   |x| ≥ 2
 *   代码把第二支写成 (((x−5)x+8)x−4)·A，展开即 A·x³−5A·x²+8A·x−4A，逐项一致。
 * 【为什么 A=-0.75】PyTorch F.interpolate(mode='bicubic') 用 A=-0.75，PIL 的
 *   BICUBIC 用 A=-0.5。本函数只服务 CLIP 位置表插值（对齐 PyTorch 参考）；
 *   SAM 位置表另有 A=-0.5 的 bicubic_weight（ds_visual_tokenizer.c）——
 *   两个重采样器各随其参考实现，不要互相"统一"（docs/06 §6.2）。
 * 【参考】PyTorch upsample_bicubic2d 的 cubic_convolution1/cubic_convolution2；
 *   Keys, "Cubic convolution interpolation for digital image processing" (1981)
 */
/* PyTorch bicubic convolution kernel (Catmull-Rom family, A = -0.75) —
 * matches F.interpolate(mode='bicubic') and PIL bicubic. */
static float ds_cubic_kernel(float x) {
    const float A = -0.75f;
    x = fabsf(x);
    if (x <= 1.0f)
        return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f)
        return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * A;
    return 0.0f;
}

/* ds_bicubic_resample_dim - 沿一个维度做抗锯齿 bicubic 重采样（可分离插值的单遍）
 * 【为什么分离两遍】bicubic 核可分离，2D 网格 16×16→10×10 拆成 H、W 两次 1D
 *   重采样（与 PyTorch 内部实现一致），复杂度从 O(D²) 降到 O(2D)。
 * 【数学】对目标格点 i，源坐标 center = (i + 0.5)·src/dst − 0.5
 *   （align_corners=False 半像素对齐，两端格点不强制重合）；
 *   权重 w[j] = ds_cubic_kernel(j − center)。
 * Matches F.interpolate(mode='bicubic', antialias=True, align_corners=False):
 *   - center of dest i in source coords: (i + 0.5) * src/dst - 0.5
 *   - downscale (src > dst): antialias — cubic kernel evaluated at
 *     (j - center) / scale over a 2*scale source window, normalized
 *     【为什么抗锯齿只影响缩小】缩小 1.6 倍时 4×4 核会"跳采"——部分源行根本
 *     摸不到，位置表的行间高频混叠成伪影；把支撑域从 ±2 拉宽到 ±2·scale
 *     （10/16 缩小时约 7×7 邻域）并归一化权重即可抑制。放大时 support 退回 2。
 *   - upscale / same: plain 4-point bicubic at center
 * Border handling: replicate (PyTorch upsample_get_value_bounded)。
 *
 * 【参数】src/dst: 输入/输出缓冲；src_n/dst_n: 被重采样维的元素数；
 *   in_stride/out_stride: 主维行距；in_o_stride/out_o_stride: 次维行距；
 *   other: 次维元素数；dim: 最内层连续元素数（同时一起被加权平均）
 *   —— 输入元素 (j, o) 位于 src + j·in_stride + o·in_o_stride，
 *      输出元素 (i, o) 位于 dst + i·out_stride + o·out_o_stride。
 * 【参考】docs/06-视觉编码器篇.md §6.1-§6.2（get_abs_pos 与两场插值事故）
 */
static void ds_bicubic_resample_dim(const float *src, float *dst,
                                     int src_n, int dst_n,
                                     long in_stride, long out_stride,
                                     long in_o_stride, long out_o_stride,
                                     int other, int dim) {
    float scale = (float)src_n / (float)dst_n;
    /* 抗锯齿只在缩小（scale>1 即 src_n>dst_n）时启用：支撑域 ±2 → ±2·scale */
    int antialias = src_n > dst_n;
    float support = antialias ? 2.0f * scale : 2.0f;

    float *w = (float *)malloc((src_n + 2) * sizeof(float));
    for (int i = 0; i < dst_n; i++) {
        /* align_corners=False: 目标格点 i 映射到源坐标 (i+0.5)·scale − 0.5 */
        float center = (i + 0.5f) * scale - 0.5f;
        /* 核支撑窗口 [center−support, center+support]，夹进源数组有效范围 */
        int j0 = (int)ceilf(center - support);
        int j1 = (int)floorf(center + support);
        if (j0 < 0) j0 = 0;
        if (j1 > src_n - 1) j1 = src_n - 1;

        float wsum = 0.0f;
        for (int j = j0; j <= j1; j++) {
            /* 缩小时除以 scale（坐标被压缩，间距变大）；放大时即原始偏移 */
            float arg = antialias ? (j - center) / scale : (j - center);
            w[j - j0] = ds_cubic_kernel(arg);
            wsum += w[j - j0];
        }
        /* 权重归一化：窗口被边界裁掉一角后防止整体增益漂移（边界亮度守恒） */
        if (wsum > 0.0f) {
            float inv = 1.0f / wsum;
            for (int j = j0; j <= j1; j++) w[j - j0] *= inv;
        }

        /* 对次维 other 份向量逐份做同一组权重的加权平均（权重只随 i/j 变，
         * 与 o、d 无关——所以提到循环外算一次、内层全部复用） */
        for (int o = 0; o < other; o++) {
            float *dst_row = dst + (long)i * out_stride + (long)o * out_o_stride;
            const float *src_base = src + (long)o * in_o_stride;
            for (int d = 0; d < dim; d++) dst_row[d] = 0.0f;
            for (int j = j0; j <= j1; j++) {
                float wj = w[j - j0];
                if (wj == 0.0f) continue;   /* 核的零区直接跳过，省一次内积 */
                const float *srow = src_base + (long)j * in_stride;
                for (int d = 0; d < dim; d++) dst_row[d] += wj * srow[d];
            }
        }
    }
    free(w);
}

/* ds_clip_encoder_forward - CLIP 编码器前向主函数
 * 完整契约见上方 "Full CLIP encoder forward pass" 注释块
 * （两个 bicubic 插值辅助函数定义在其与本体之间） */
float *ds_clip_encoder_forward(ds_ctx_t *ctx,
                                const unsigned char *rgb_pixels, int width, int height, int channels,
                                const float *sam_features, int n_sam_tokens,
                                int *out_seq_len) {
    /* CLIP 直接消费 SAM 特征，原始像素参数仅为保持公共 API 兼容 */
    (void)rgb_pixels; (void)width; (void)height; (void)channels;
    ds_clip_encoder_t *clip = &ctx->clip_encoder;
    ds_config_t *cfg = &ctx->config;
    int clip_dim = DS_CLIP_HIDDEN; /* 1024 */

    int clip_n_patches;
    float *clip_tokens;  /* [clip_n_patches, clip_dim] — CLIP patch tokens */

    /* Step 1: Build CLIP input embeddings
     *
     * Both V1 and V3: CLIP receives SAM's downsampled features directly
     * as patch_embeds, bypassing its own patch_embedding Conv2d.
     *
     * V1 flow: SAM → 16x Conv compression → 256 tokens (1024-dim) → CLIP
     * V3 flow: SAM → 256 tokens (1024-dim) → CLIP
     *
     * In both cases, sam_features is [n_sam_tokens, 1024] token format.
     * The patch_embedding Conv2d [1024, 3, 14, 14] is NOT used — CLIP
     * processes the SAM features directly through its transformer layers.
     */

    /* Both V1 and V3: use SAM features directly as CLIP patch_embeds */
    clip_n_patches = n_sam_tokens;  /* 256 for 1024×1024 input */
    clip_tokens = (float *)malloc(clip_n_patches * clip_dim * sizeof(float));
    memcpy(clip_tokens, sam_features, clip_n_patches * clip_dim * sizeof(float));

    if (ds_verbose >= 1)
        fprintf(stderr, "CLIP: using SAM features directly as patch_embeds (%d tokens x %d dim, V%d)\n",
                clip_n_patches, clip_dim, cfg->model_version);

    /* Build full input: CLS token + patch tokens
     * total_len = 1 + 256 = 257，与 CLIP 位置表 [257,1024] 的行数对齐 */
    int total_len = 1 + clip_n_patches;
    float *x = (float *)calloc(total_len * clip_dim, sizeof(float));

    /* CLS token at position 0
     * 【CLS 的进与出】class_embedding 只在第 0 位充当"全局信息汇聚点"——24 层
     * 双向注意力中每个 patch 都会看它；输出时被剥掉（OCR 要的是逐空间位置
     * 特征，不是分类向量），见下方 "remove CLS"（docs/06 §5.2） */
    if (clip->class_embedding) {
        memcpy(x, clip->class_embedding, clip_dim * sizeof(float));
    }

    /* Patch tokens */
    memcpy(x + clip_dim, clip_tokens, clip_n_patches * clip_dim * sizeof(float));
    free(clip_tokens);

    /* Add position embedding */
    if (clip->position_embedding) {
        /* Position embedding shape: [257, 1024] (1 CLS + 256 patches for 224x224/14=16x16)
         * For Unlimited-OCR with 1024x1024: clip_n_patches = 256, total_len = 257 → matches!
         * For V1: clip_n_patches may differ, need interpolation via get_abs_pos.
         *
         * Python's get_abs_pos: bicubic interpolation of spatial part to target size.
         * C implementation: for V3 with total_len=257, position_embedding[257][1024] matches exactly.
         * For other sizes, we need to interpolate (both smaller and larger). */
        int src_size = (int)sqrtf(256.0f);  /* 16x16 for original CLIP position embedding */
        int tgt_size = (int)sqrtf((float)clip_n_patches);  /* 实际 patch 网格边长（如 10×10=100） */
        if (tgt_size == src_size && total_len <= 257) {
            /* Direct copy — no interpolation needed
             * 1024² 输入恰好 256 patch：逐行相加位置表即可（含第 0 行给 CLS）。
             * 【历史 bug】早期实现遇到 total_len≠257 时直接截取 position_embedding
             * 前 total_len 行——等于拿 16×16 表的前几行当位置编码，101 token 的
             * 位置系统性错乱，输出挂一串孤儿 HTML 闭合标签（docs/06 §6.3 事故一，
             * git 817b129 修复记录）。现在的策略：精确匹配走直加，否则完整插值。 */
            for (int i = 0; i < total_len; i++) {
                for (int d = 0; d < clip_dim; d++) {
                    x[i * clip_dim + d] += clip->position_embedding[i * clip_dim + d];
                }
            }
        } else {
            /* Need bicubic interpolation (tgt_size != src_size).
             * This follows Python's get_abs_pos:
             * 1. Split CLS and spatial part
             * 2. Reshape spatial to [1, dim, src_h, src_w]
             * 3. Bicubic interpolate to [1, dim, tgt_h, tgt_w]
             * 4. Reshape back and concatenate with CLS */

            /* Interpolate spatial part: position_embedding[1:257] = [256, 1024]
             * 剥离 CLS 行：插值只作用于 16×16 空间网格，CLS 行原样保留 */
            float *spatial_pos = (float *)malloc(256 * clip_dim * sizeof(float));
            memcpy(spatial_pos, clip->position_embedding + clip_dim, 256 * clip_dim * sizeof(float));

            /* Bicubic + antialias interpolation [src,src,dim] → [tgt,tgt,dim],
             * separable H pass then W pass (matches PyTorch F.interpolate).
             * Layout: (row, col, d) at row*colspan*dim + col*dim + d. */
            float *tmp_pos = (float *)malloc(tgt_size * src_size * clip_dim * sizeof(float));
            /* H pass: (sh,sw,d) → (th,sw,d) */
            ds_bicubic_resample_dim(spatial_pos, tmp_pos,
                                     src_size, tgt_size,
                                     (long)src_size * clip_dim, (long)src_size * clip_dim,
                                     clip_dim, clip_dim, src_size, clip_dim);
            float *resized_pos = (float *)malloc(clip_n_patches * clip_dim * sizeof(float));
            /* W pass: (th,sw,d) → (th,tw,d) */
            ds_bicubic_resample_dim(tmp_pos, resized_pos,
                                     src_size, tgt_size,
                                     clip_dim, clip_dim,
                                     (long)src_size * clip_dim, (long)tgt_size * clip_dim,
                                     tgt_size, clip_dim);
            free(tmp_pos);
            free(spatial_pos);

            /* Apply: CLS + interpolated spatial
             * CLS 位置行（第 0 行）不插值直接加；patch 部分用重采样后的表 */
            for (int d = 0; d < clip_dim; d++) {
                x[d] += clip->position_embedding[d]; /* CLS */
            }
            for (int p = 0; p < clip_n_patches; p++) {
                for (int d = 0; d < clip_dim; d++) {
                    x[(1 + p) * clip_dim + d] += resized_pos[p * clip_dim + d];
                }
            }
            free(resized_pos);
        }
    }

    /* Pre-LayerNorm */
    if (clip->pre_layernorm_weight) {
        float *tmp = (float *)malloc(total_len * clip_dim * sizeof(float));
        ds_layer_norm(tmp, x, clip->pre_layernorm_weight, clip->pre_layernorm_bias,
                      total_len, clip_dim, 1e-6f);
        free(x);
        x = tmp;
    }

    /* CLIP transformer layers (24 layers)
     * 每层产出写入新分配的 next 缓冲再交换——层内残差要同时读输入 x，
     * 原地覆写会破坏未读完的输入（clip_layer_forward 的 out/x 是不同缓冲） */
    for (int l = 0; l < DS_CLIP_LAYERS; l++) {
        float *next = (float *)malloc(total_len * clip_dim * sizeof(float));
        clip_layer_forward(next, x, clip, l, total_len);
        free(x);
        x = next;
        if (ds_verbose >= 2)
            fprintf(stderr, "CLIP layer %d/%d done\n", l + 1, DS_CLIP_LAYERS);
    }

    /* Post-LayerNorm (optional — Unlimited-OCR doesn't have it) */
    if (clip->final_norm_weight) {
        float *tmp = (float *)malloc(total_len * clip_dim * sizeof(float));
        ds_layer_norm(tmp, x, clip->final_norm_weight, clip->final_norm_bias,
                      total_len, clip_dim, 1e-6f);
        free(x);
        x = tmp;
    }

    /* Extract CLIP output: remove CLS token (take [:, 1:]) → [clip_n_patches, 1024]
     * CLS"进得去、出不来"：只在 24 层里当全局汇聚点，输出剥掉
     * （OCR 需要每个空间位置的特征，非分类向量；docs/06 §5.2） */
    int clip_output_len = clip_n_patches;
    float *clip_output = (float *)malloc(clip_output_len * clip_dim * sizeof(float));
    memcpy(clip_output, x + clip_dim, clip_output_len * clip_dim * sizeof(float));
    free(x); /* Free CLIP transformer output */

    /* =====================================================================
     * Concatenate CLIP output + SAM flattened features → projector input
     * =====================================================================
     * Python modeling code:
     *   local_features = cat(clip_output[:, 1:], sam_output.flatten(2).permute(0,2,1), dim=-1)
     *
     * For Unlimited-OCR (V3):
     *   clip_output[:, 1:] = [B, 256, 1024]  (256 CLIP patch tokens)
     *   sam_output.flatten(2).permute(0,2,1) = [B, 256, 1024]  (256 SAM tokens)
     *   concat → [B, 256, 2048]
     *
     * For DeepSeek-OCR V1:
     *   clip_output[:, 1:] = [B, clip_n_patches, 1024]
     *   sam_output = [B, n_sam_tokens, 1024] (visual_tokens from SAM neck)
     *   For V1: clip_n_patches may differ from n_sam_tokens — this is known
     *   to be broken for V1. V3 is the primary target.
     * ===================================================================== */

    float *concat_features = NULL;
    int concat_len = 0;
    int proj_in_dim = 0;

    if ((cfg->model_version == 1 || cfg->model_version == 3 || cfg->model_version == 4)
        && clip_output_len == n_sam_tokens) {
        /* V1/V3/V4: CLIP and SAM have same token count (256 each for 1024×1024 input)
         * Concat: [256, 1024] + [256, 1024] → [256, 2048] along feature dim
         * 【顺序】每个 token 的 2048 维 = 前 1024 CLIP（语义）+ 后 1024 SAM（几何），
         * 与 Python cat(clip_output, sam_output, dim=-1) 一致——CLIP 在前、SAM 在后。
         * （本文件头部示意图写反了顺序，以这里为准。） */
        concat_len = clip_output_len;
        proj_in_dim = clip_dim + cfg->sam_ds2_dim; /* 1024 + 1024 = 2048 */
        concat_features = (float *)malloc(concat_len * proj_in_dim * sizeof(float));
        for (int t = 0; t < concat_len; t++) {
            memcpy(concat_features + t * proj_in_dim,
                   clip_output + t * clip_dim, clip_dim * sizeof(float));
            memcpy(concat_features + t * proj_in_dim + clip_dim,
                   sam_features + t * cfg->sam_ds2_dim, cfg->sam_ds2_dim * sizeof(float));
        }
        free(clip_output);

        if (ds_verbose >= 1)
            fprintf(stderr, "CLIP+SAM concat: [%d, %d] + [%d, %d] → [%d, %d]\n",
                    clip_output_len, clip_dim, n_sam_tokens, cfg->sam_ds2_dim,
                    concat_len, proj_in_dim);
    } else {
        /* Fallback: use CLIP output only (broken for V1, but won't crash)
         * 仅在 CLIP/SAM token 数不一致（如个别 V1 输入）时触发：projector 期望
         * 2048 维输入，这里退化为 1024 维 CLIP-only——输出语义错误但不崩溃 */
        concat_features = clip_output;
        concat_len = clip_output_len;
        proj_in_dim = clip_dim;
        if (ds_verbose >= 1)
            fprintf(stderr, "Warning: CLIP+SAM concat skipped (clip=%d, sam=%d), using CLIP only\n",
                    clip_output_len, n_sam_tokens);
    }

    /* Project to decoder dimension via projector: [concat_len, proj_in_dim] → [concat_len, 1280]
     * projector 是【单层线性】（config: projector_type="linear"，非多层 MLP），
     * 权重实测 [1280, 2048] + bias [1280]；1280 与 MoE 解码器隐藏维一致 */
    int dec_hidden = cfg->dec_hidden; /* 1280 */
    float *projected = (float *)malloc(concat_len * dec_hidden * sizeof(float));
    if (ctx->projector.weight) {
        ds_linear(projected, concat_features, ctx->projector.weight, ctx->projector.bias,
                  concat_len, proj_in_dim, dec_hidden);
    } else {
        memset(projected, 0, concat_len * dec_hidden * sizeof(float));
    }
    free(concat_features);

    if (cfg->model_version == 1 || cfg->model_version == 3 || cfg->model_version == 4) {
        /* =====================================================================
         * Insert image_newline between grid rows + append view_seperator
         * Both V1/V3/V4 need this layout:
         * Python code:
         *   global_features = proj_output.view(h, w, n_dim)
         *   global_features = cat([global_features, image_newline[None,None,:].expand(h,1,n_dim)], dim=1)
         *   global_features = global_features.view(-1, n_dim)  # [272, 1280]
         *   global_local_features = cat([global_features, view_seperator[None,:]], dim=0)  # [273, 1280]
         *
         * Layout: 16 rows × (16 patches + 1 newline) + 1 view_seperator = 273 tokens
         * ===================================================================== */
        int grid_h = (int)sqrtf((float)concat_len); /* 16 */
        int grid_w = concat_len / grid_h;           /* 16 */
        float *image_newline = ctx->vis_tokenizer.image_newline;   /* [1280] */
        float *view_sep = ctx->vis_tokenizer.view_seperator;       /* [1280] */

        if (image_newline && view_sep && grid_h * grid_w == concat_len) {
            /* Build: [row0_patches, newline, row1_patches, newline, ..., view_sep]
             * Each row has grid_w patches + 1 newline = grid_w+1 tokens
             * Total = grid_h * (grid_w + 1) + 1
             * 【为什么要插 newline】projector 输出的一维序列丢掉了二维结构——
             * 解码器分不清"两列并排"和"一行流"。image_newline 是 checkpoint 里
             * 训练出来的可学习向量（源起 LLaVA-1.5），显式给出行边界信号，
             * 对表格/多栏版面是刚需（docs/06 §8）。V2 没有 newline（无此张量），
             * 行结构由因果流查询的 token 顺序隐式携带。 */
            int final_len = grid_h * (grid_w + 1) + 1; /* 16*17+1 = 273 */
            float *final_out = (float *)malloc(final_len * dec_hidden * sizeof(float));
            int dst = 0;
            for (int row = 0; row < grid_h; row++) {
                /* Copy grid_w patches from this row */
                memcpy(final_out + dst * dec_hidden,
                       projected + row * grid_w * dec_hidden,
                       grid_w * dec_hidden * sizeof(float));
                dst += grid_w;
                /* Insert image_newline */
                memcpy(final_out + dst * dec_hidden, image_newline, dec_hidden * sizeof(float));
                dst += 1;
            }
            /* Append view_seperator */
            memcpy(final_out + dst * dec_hidden, view_sep, dec_hidden * sizeof(float));
            dst += 1;

            free(projected);
            projected = final_out;
            *out_seq_len = final_len;

            if (ds_verbose >= 1)
                fprintf(stderr, "CLIP+SAM+Projector: %d patches → %d tokens (with newlines + view_sep, V%d)\n",
                        concat_len, final_len, cfg->model_version);
        } else {
            *out_seq_len = concat_len;
            if (ds_verbose >= 1)
                fprintf(stderr, "Warning: image_newline/view_seperator not loaded or grid mismatch (%d=%d*%d)\n",
                        concat_len, grid_h, grid_w);
        }
    } else {
        *out_seq_len = concat_len;
        if (ds_verbose >= 1)
            fprintf(stderr, "CLIP encoder: %d patches -> %d tokens out (projected to %d)\n",
                    clip_n_patches, concat_len, dec_hidden);
    }

    return projected;
}

/* ========================================================================
 * DeepEncoder V2 Forward Pass (Qwen2-0.5B based)
 * ======================================================================== */

/* enc_linear_f64acc - 编码器专用线性层（float64 累加点积）
 * 【数学】y[s,o] = b[o] + Σ_i x[s,i]·W[o,i]（W 行主序 [out,in]），
 *   累加器用 double，输入/权重仍 F32；每输出元素一次归约。
 * 【为什么不用 BLAS】cblas_sgemm 只做 F32 累加；24 层 DeepEncoder 会把 F32
 *   舍入误差显著放大（docs/14 §6.6: 编码侧 1.5% 的 exp 近似就让 V3 小图 CER
 *   冲到 95%——数值精度在本引擎不可妥协）。double 累加代价约 2× 计算时间，
 *   换取精度漂移大幅收敛；编码器维度小（896/4864），朴素三重循环可接受。
 * 【参数】y: [seq_len,out_dim]；x: [seq_len,in_dim]；W: [out_dim,in_dim]；
 *   b: [out_dim] 或 NULL（无偏置）
 */
/* Encoder-specific linear layers with float64 accumulation.
 * The 24-layer DeepEncoder amplifies float32 rounding errors dramatically.
 * Accumulating dot products in float64 while keeping weights/inputs as float32
 * costs ~2x in compute but dramatically reduces precision drift. */
static void enc_linear_f64acc(float *y, const float *x, const float *W, const float *b,
                               int seq_len, int in_dim, int out_dim) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            /* double 累加器从 bias 起步，逐项乘加——顺序固定保证可复现 */
            double sum = (b != NULL) ? (double)b[o] : 0.0;
            for (int i = 0; i < in_dim; i++) {
                sum += (double)x_row[i] * (double)w_row[i];
            }
            y_row[o] = (float)sum;
        }
    }
}

/* enc_linear_nobias_f64acc - enc_linear_f64acc 的无偏置薄封装（b=NULL）
 * 目前无调用点（__attribute__((unused)) 抑制告警），保留作为调试工具 */
static void __attribute__((unused)) enc_linear_nobias_f64acc(float *y, const float *x, const float *W,
                                       int seq_len, int in_dim, int out_dim) {
    enc_linear_f64acc(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* enc_apply_rope - DeepEncoder V2（Qwen2 风格）的 RoPE，原地旋转 Q 与 K
 * 【数学】NeoX/Llama 拆半式（split-half）：把 head_dim=64 的头拆成两半配对，
 *   x1 = x[i], x2 = x[half+i]，频率 inv_freq[i] = theta^(-2i/head_dim)：
 *   x'[i]       = x1·cos(pos·f) − x2·sin(pos·f)
 *   x'[half+i]  = x2·cos(pos·f) + x1·sin(pos·f)
 *   相对位置由内积自动体现：⟨R_m·q, R_n·k⟩ = ⟨q, R_{n−m}·k⟩。
 * 【为什么 θ=10⁶】V2 的 cfg->enc_rope_theta=1e6（ds_ocr.c），比常规 1e4 大——
 *   长波长的低频对衰减更慢，利于长序列外推（位置数 512 量级虽短，但与官方
 *   checkpoint 训练配置保持一致是硬要求）。
 * 【本代码库没有 freq 缩放（NTK/YaRN）】这里只放大基频 θ 本身（1e4→1e6），
 *   checkpoint 的 rope_scaling 为空。若模型带 rope_scaling，常见两种：
 *   NTK-aware 把 θ 改成 θ·s^(d/(d−2))（整体拉伸频谱，高频维也变慢，
 *   外推长上下文）；YaRN 再按波长分段——低频维纯外推、高频维纯内插
 *   （inv_freq 除以 s）、中频斜坡过渡，并把注意力 softmax 温度乘
 *   0.1·ln(s)+1 补偿。二者都会改变 inv_freq 的取值；本引擎两种都未实现
 *   （V4 解码器的 page2d RoPE 是另一套坐标，同样无缩放），遇到带缩放的
 *   checkpoint 必须在算 freq 处插入对应公式，否则长文位置全部错乱。
 * 【为什么 float64 算频率/角度/cos/sin】24 层逐层应用 RoPE，三角函数的 F32
 *   舍入会逐层累积漂移；f64 只增加这部分小循环的开销，主数据仍是 F32。
 * 【参数】q: [seq_len, n_q_heads×head_dim]；k: [seq_len, n_kv_heads×head_dim]；
 *   二者均原地修改；V 不旋转（无位置信息）。theta: cfg->enc_rope_theta。
 * 【参考】Roformer（Su et al.）；Qwen2 的 rotate_half 实现
 */
/* Apply RoPE to Q/K vectors for encoder (Qwen2 style)
 * Uses float64 for frequency, angle, cos/sin computation to avoid
 * trig precision drift accumulating over 24 layers. */
static void enc_apply_rope(float *q, float *k, int seq_len, int n_q_heads,
                            int n_kv_heads, int head_dim, float rope_theta) {
    /* Apply RoPE in split-half (NeoX/Llama) style:
     * x1 = x[:half], x2 = x[half:]
     * rotated = cat(-x2, x1)
     * result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    double theta_d = (double)rope_theta;
    for (int pos = 0; pos < seq_len; pos++) {
        /* Q heads */
        for (int h = 0; h < n_q_heads; h++) {
            float *qh = q + pos * n_q_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                /* 频率随维号几何衰减 inv_freq[i]=θ^(-2i/64)：i=0 → 1（最高频，
                 * 转最快，区分相邻 token）；i=63 → ≈1/θ（最低频，转最慢，
                 * 编码长程相对位置）——head 内 32 对"时钟"各走各的速档 */
                double freq = 1.0 / pow(theta_d, (double)(2 * i) / (double)head_dim);
                double angle = (double)pos * freq;
                double cos_t = cos(angle);
                double sin_t = sin(angle);
                double x1 = (double)qh[i];
                double x2 = (double)qh[half + i];
                qh[i]        = (float)(x1 * cos_t - x2 * sin_t);
                qh[half + i] = (float)(x2 * cos_t + x1 * sin_t);
            }
        }
        /* K heads */
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = k + pos * n_kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / pow(theta_d, (double)(2 * i) / (double)head_dim);
                double angle = (double)pos * freq;
                double cos_t = cos(angle);
                double sin_t = sin(angle);
                double x1 = (double)kh[i];
                double x2 = (double)kh[half + i];
                kh[i]        = (float)(x1 * cos_t - x2 * sin_t);
                kh[half + i] = (float)(x2 * cos_t + x1 * sin_t);
            }
        }
    }
}

/* ── DeepEncoder 注意力 head 并行门控 ─────────────────────────────────
 * enc_gqa_mixed_attention 按 head 切分：每个 (head, query) 输出元素
 * 独立计算、写区间不相交 → 逐位一致（回归 CER 保持 0.00）。
 * 线程数 DS_ENC_ATTN_THREADS（默认 4）。
 *
 * 全局默认预算：0=关（串行），N=注意力用 N 线程。编码器与 SAM 的
 * 注意力循环不重叠执行，ds_visual_tokenizer.c 会用 ds_enc_thread_gate()
 * 把自己的线程预算"传递"给编码器（钉住 tl_enc_*），避免两层各自开
 * N 线程造成超额订阅。 */
int ds_enc_attn_parallel = 4;   /* 线程预算: 0=关，N=注意力用 N 线程 */

/* 线程局部影子（ds_sam_thread_gates 钉住）；未钉住时用全局默认
 * __thread 存储：每个线程一份，避免多线程下竞争写全局变量 */
static __thread int tl_enc_budget;
static __thread int tl_enc_pinned;
/* ds_enc_thread_gate - 由 SAM 侧调用：钉住本线程后续编码器注意力的线程预算 */
void ds_enc_thread_gate(int budget) { tl_enc_budget = budget; tl_enc_pinned = 1; }
/* ds_enc_gate_budget - 取当前生效预算：被钉住用影子值，否则用全局默认 */
int ds_enc_gate_budget(void) { return tl_enc_pinned ? tl_enc_budget : ds_enc_attn_parallel; }

/* GQA attention with mixed mask (bidirectional for visual, causal for flow)
 * Uses float64 for QK^T dot product, softmax, and V accumulation to mitigate
 * precision drift in the 24-layer DeepEncoder pipeline. */
/* enc_attn_task_t - enc_attn_process_head 的共享任务描述
 * （并行时每个线程持有一份拷贝，仅 tid/nthreads 不同；字段全部只读） */
typedef struct {
    const float *Q, *K, *V;                  /* Q [total,14×64]；K/V [total,2×64] */
    float *out;                              /* [total,14×64] 输出（各头写各的区间） */
    int visual_len, total_len, n_q_heads, n_kv_heads, head_dim;
    int tid, nthreads;                       /* 本 worker 的线程号/总线程数（head 条带划分） */
} enc_attn_task_t;

/* enc_attn_process_head - 计算一个 Q 头在全部 query 位置上的混合掩码注意力
 * 【算法】GQA 分组: kv_h = h / (n_q_heads/n_kv_heads) = h/7，即 14 个 Q 头
 *   共享 2 组 KV，每 7 个 Q 头读同一组 K/V（省 KV cache 与带宽）。
 *   混合掩码（本函数的核心，见下方 vis_end 分支）:
 *     query 位置 qi < visual_len（视觉 token）→ 可见 [0, visual_len)  双向
 *     query 位置 qi ≥ visual_len（因果流查询）→ 可见 [0, qi]          因果
 *   在线 softmax（Float64）：遇更大分数先修正旧累计（corr = e^{m_old−m_new}），
 *   块大小退化为 1（逐 KV 流式），与 docs/13 §2 的 FA 三行公式同构：
 *     m' = max(m_old, s)；l' = l·corr + e^{s−m'}；O' = O·corr + e^{s−m'}·v
 * 【维度】Q [total_len, 14×64]，K/V [total_len, 2×64]，out 同 Q 布局。
 * 【为什么 Float64】24 层编码器对 F32 舍入敏感（docs/14 §6.6：编码侧 1.5%
 *   的 exp 近似就能让 CER 冲破阈值），QK^T / softmax / V 累加全程 double。
 * 【为什么按 head 拆】每个 (h, qi) 输出元素独立 → 多线程切 head 时写区间
 *   不相交，与串行逐位一致（并行化不破坏 CER=0.00 回归的前提）。
 * 【参数】t: 共享任务描述（含 tid/nthreads，由 worker 决定处理哪些头）；h: Q 头号
 */
static void enc_attn_process_head(const enc_attn_task_t *t, int h) {
    const int total_len = t->total_len, visual_len = t->visual_len;
    const int n_q_heads = t->n_q_heads, n_kv_heads = t->n_kv_heads, head_dim = t->head_dim;
    const int kv_group_size = n_q_heads / n_kv_heads;
    const float *Q = t->Q, *K = t->K, *V = t->V;
    float *out = t->out;
    double scale = 1.0 / sqrt((double)head_dim);
    int kv_h = h / kv_group_size;   /* GQA：本 Q 头所属的 KV 组（7 个 Q 头共享 1 组） */
    double *o_row_d = (double *)malloc(head_dim * sizeof(double));
    for (int qi = 0; qi < total_len; qi++) {
        const float *q_vec = Q + qi * n_q_heads * head_dim + h * head_dim;

        /* 混合掩码的关键分支——双向与因果的差别只有可见域上界 vis_end：
         * 视觉 token（qi < visual_len）：上界=visual_len，看到全部视觉 token
         *   （包括"未来"的），即双向注意力，充分交流图像内容；
         * 因果流查询（qi ≥ visual_len）：上界=qi+1，只看自己之前——按序
         *   收集视觉信息，输出天然带顺序性（DeepEncoder V2 的创新点）。
         * 注意因果查询同样覆盖全部视觉段（[0,visual_len) ⊆ [0,qi]）。 */
        int vis_end;
        if (qi < visual_len) {
            vis_end = visual_len;
        } else {
            vis_end = qi + 1;
        }

        /* Float64 online softmax with QK^T and V accumulation
         * 状态: max_score（已见最大分 m）、sum_exp（指数和 l）、o_row_d（未归一
         * 加权值 O）；三者构成 docs/13 §2 的 (m, l, O) 在线三元组 */
        double max_score = -1e30;
        double sum_exp = 0.0;
        for (int d = 0; d < head_dim; d++) o_row_d[d] = 0.0;

        for (int ki = 0; ki < vis_end; ki++) {
            const float *k_vec = K + ki * n_kv_heads * head_dim + kv_h * head_dim;
            const float *v_vec = V + ki * n_kv_heads * head_dim + kv_h * head_dim;

            /* Float64 QK dot product */
            double s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += (double)q_vec[d] * (double)k_vec[d];
            s *= scale;

            if (s > max_score) {
                /* 新最大值出现：旧累计基于 m_old 的指数要整体换基准到 m_new——
                 * corr = e^{m_old−m_new} 同时修正 l 与 O（FA 修正公式） */
                double correction = exp(max_score - s);
                sum_exp = sum_exp * correction + 1.0;
                for (int d = 0; d < head_dim; d++)
                    o_row_d[d] = o_row_d[d] * correction + (double)v_vec[d];
                max_score = s;
            } else {
                /* 无新最大值：直接以当前 m 为基准累加新项 */
                double wt = exp(s - max_score);
                sum_exp += wt;
                for (int d = 0; d < head_dim; d++)
                    o_row_d[d] += wt * (double)v_vec[d];
            }
        }

        /* 归一化输出：O /= l（数学上等价于全量 softmax 加权平均，仅舍入差异） */
        float *out_vec = out + qi * n_q_heads * head_dim + h * head_dim;
        if (sum_exp > 0.0) {
            double inv = 1.0 / sum_exp;
            for (int d = 0; d < head_dim; d++)
                out_vec[d] = (float)(o_row_d[d] * inv);
        } else {
            for (int d = 0; d < head_dim; d++) out_vec[d] = 0.0f;
        }
    }
    free(o_row_d);
}

/* enc_attn_worker - 并行 worker：线程 tid 负责头号 tid, tid+nthreads, ...（条带轮转）
 * 条带（stride）分配比连续块分配负载更均衡（各头计算量相同，效果等价） */
static void *enc_attn_worker(void *arg) {
    enc_attn_task_t *t = (enc_attn_task_t *)arg;
    for (int h = t->tid; h < t->n_q_heads; h += t->nthreads)
        enc_attn_process_head(t, h);
    return NULL;
}
/* enc_gqa_mixed_attention - GQA 混合掩码注意力的并行调度入口
 * 【算法】按 head 切分并行（而非按 query 切分）：每个 (head, query) 输出元素
 *   独立、写区间不相交 → 并行结果与串行逐位一致，CER 回归保持 0.00。
 * 【线程数决策】budget = ds_enc_gate_budget()（SAM 钉住或全局默认 4）；
 *   生效条件: budget>0 且全局线程池 >1 且 Q 头数 ≥4（够切分才值得开线程）；
 *   环境变量 DS_ENC_ATTN_THREADS 可覆盖；上限夹到 n_q_heads（14）。
 * 【参数】out/Q: [total_len, n_q_heads×head_dim]；K/V: [total_len, n_kv_heads×head_dim]；
 *   visual_len: 双向段长度（前 visual_len 行视觉 token，其余因果）
 */
static void enc_gqa_mixed_attention(float *out, const float *Q, const float *K,
                                     const float *V, int visual_len, int total_len,
                                     int n_q_heads, int n_kv_heads, int head_dim) {
    enc_attn_task_t task = {
        .Q = Q, .K = K, .V = V, .out = out,
        .visual_len = visual_len, .total_len = total_len,
        .n_q_heads = n_q_heads, .n_kv_heads = n_kv_heads, .head_dim = head_dim,
        .tid = 0, .nthreads = 1,
    };
    int nth = 1;
    int budget = ds_enc_gate_budget();
    if (budget > 0 && ds_get_threads() > 1 && n_q_heads >= 4) {
        nth = budget;
        /* 环境变量覆盖（调试用）：DS_ENC_ATTN_THREADS=N 强制 N 线程 */
        const char *e = getenv("DS_ENC_ATTN_THREADS");
        if (e && e[0]) { int v = atoi(e); if (v > 0) nth = v; }
        if (nth > n_q_heads) nth = n_q_heads;   /* 线程多于头数没有意义 */
    }
    if (nth > 1) {
        /* 主线程处理头 0 并充当第 0 个 worker，额外起 nth-1 个 pthread */
        pthread_t *th = (pthread_t *)malloc((nth - 1) * sizeof(pthread_t));
        enc_attn_task_t *dt = (enc_attn_task_t *)malloc(nth * sizeof(enc_attn_task_t));
        for (int i = 1; i < nth; i++) {
            dt[i] = task; dt[i].tid = i; dt[i].nthreads = nth;
            pthread_create(&th[i - 1], NULL, enc_attn_worker, &dt[i]);
        }
        task.nthreads = nth;
        enc_attn_worker(&task);
        for (int i = 1; i < nth; i++) pthread_join(th[i - 1], NULL);
        free(th); free(dt);
    } else {
        for (int h = 0; h < n_q_heads; h++) enc_attn_process_head(&task, h);
    }

}


/* enc_v2_layer_forward - DeepEncoder V2 单层前向（Qwen2-0.5B 架构，Pre-Norm）
 * 【流程】x [total,896]
 *   → RMSNorm(ln1, eps=dec_rms_norm_eps)
 *   → Wq/Wk/Wv 投影（带 bias；GQA: Q 896=14头×64，K/V 128=2头×64，7:1 分组）
 *   → enc_apply_rope（θ=1e6，Q/K 旋转，V 不动）
 *   → enc_gqa_mixed_attention（视觉段双向 / 因果查询段因果）
 *   → Wo（无 bias）→ +x 残差
 *   → RMSNorm(ln2) → gate/up 投影（无 bias, 896→4864）
 *   → SwiGLU: silu(gate)⊙up → down 投影（4864→896）→ + 残差 → out
 * 【参数】out/x: [total_len,896]；visual_len: 前 visual_len 行是视觉 token
 *   （双向段），其余为因果流查询；layer_idx: 0..23
 * 【为什么 QKV 带 bias 而 Wo/gate/up/down 无 bias】与 Qwen2-0.5B checkpoint
 *   的张量命名一一对应（q/k/v_proj 有 bias，o_proj/mlp 无）。
 * 【参考】docs/06-视觉编码器篇.md §9（V2 行）；Qwen2 modeling 等价物
 */
static void enc_v2_layer_forward(float *out, const float *x,
                                  ds_ctx_t *ctx, int layer_idx,
                                  int visual_len, int total_len) {
    ds_deep_encoder_t *enc = &ctx->encoder;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->enc_hidden;
    int n_q_heads = cfg->enc_heads;
    int n_kv_heads = cfg->enc_kv_heads;
    int head_dim = cfg->enc_head_dim;
    int intermediate = cfg->enc_intermediate;
    int kv_dim = n_kv_heads * head_dim;   /* K/V 每行宽 = 2 头×64 = 128（Q 行宽 896 的 1/7） */

    /* 本层权重指针速查（checkpoint 前缀 model.qwen2_model.model.model.layers.N.，
     * 加载见 ds_ocr.c 的 ENC_WEIGHT 宏；线性权重形状按 [out,in] 行主序，与 ds_linear 约定一致）:
     *   input_layernorm.weight [896]                        注意力前 RMSNorm γ
     *   self_attn.q_proj.weight [896,896] + bias [896]      Q 投影（14 头×64 维）
     *   self_attn.k_proj.weight [128,896] + bias [128]      K 投影（2 KV 头×64 维，GQA）
     *   self_attn.v_proj.weight [128,896] + bias [128]      V 投影（同上）
     *   self_attn.o_proj.weight [896,896]（无 bias）         注意力输出投影
     *   post_attention_layernorm.weight [896]               FFN 前 RMSNorm γ
     *   mlp.gate_proj.weight [4864,896]（无 bias）           SwiGLU gate 分支
     *   mlp.up_proj.weight   [4864,896]（无 bias）           SwiGLU up 分支
     *   mlp.down_proj.weight [896,4864]（无 bias）           SwiGLU 降维
     * 最终 norm 在 model.qwen2_model.model.model.norm.weight [896]；
     * 因果流查询嵌入在 model.qwen2_model.query_1024.weight [256,896] /
     * query_768.weight [144,896]（加载进 ctx->vis_tokenizer，由调用方传入）。
     */
    float *ln1_w = enc->layers[layer_idx].layer_norm1_weight;
    float *wq_w = enc->layers[layer_idx].wq_weight;
    float *wk_w = enc->layers[layer_idx].wk_weight;
    float *wv_w = enc->layers[layer_idx].wv_weight;
    float *wo_w = enc->layers[layer_idx].wo_weight;
    float *wq_b = enc->layers[layer_idx].wq_bias;
    float *wk_b = enc->layers[layer_idx].wk_bias;
    float *wv_b = enc->layers[layer_idx].wv_bias;
    float *ln2_w = enc->layers[layer_idx].layer_norm2_weight;
    float *gate_w = enc->layers[layer_idx].gate_weight;
    float *up_w = enc->layers[layer_idx].up_weight;
    float *down_w = enc->layers[layer_idx].down_weight;

    /* Pre-attention RMSNorm
     * 复用 dec_rms_norm_eps（编码器与解码器同一 eps 配置项，官方 checkpoint 如此） */
    float *x_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(x_norm, x, ln1_w, total_len, hidden, cfg->dec_rms_norm_eps);

    /* QKV projections with bias (GQA: Q is full, K/V are smaller)
     * Q [*,896]（14 头×64）；K/V [*,128]（2 头×64）——K/V 只有 Q 的 1/7 大小，
     * 这是 GQA 省显存/带宽的直接体现（docs/13 §3.1 的 KV cache 字节账） */
    float *Q = (float *)malloc(total_len * hidden * sizeof(float));
    float *K = (float *)malloc(total_len * kv_dim * sizeof(float));
    float *V = (float *)malloc(total_len * kv_dim * sizeof(float));

    ds_linear(Q, x_norm, wq_w, wq_b, total_len, hidden, hidden);
    ds_linear(K, x_norm, wk_w, wk_b, total_len, hidden, kv_dim);
    ds_linear(V, x_norm, wv_w, wv_b, total_len, hidden, kv_dim);

    /* Debug: dump QKV before RoPE for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_Q_before_rope", Q, total_len * hidden, "[288, 896]");
        ds_dump_tensor("enc0_K_before_rope", K, total_len * kv_dim, "[288, 128]");
        ds_dump_tensor("enc0_V_before_rope", V, total_len * kv_dim, "[288, 128]");
    }

    /* Apply RoPE to Q and K */
    enc_apply_rope(Q, K, total_len, n_q_heads, n_kv_heads, head_dim, cfg->enc_rope_theta);

    /* Debug: dump QKV after RoPE for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_Q_after_rope", Q, total_len * hidden, "[288, 896]");
        ds_dump_tensor("enc0_K_after_rope", K, total_len * kv_dim, "[288, 128]");
    }

    /* GQA mixed attention (bidirectional for visual, causal for flow)
     * 混合掩码在这里生效：visual_len 之前双向、之后因果（详见
     * enc_attn_process_head 的 vis_end 分支注释） */
    float *attn_out = (float *)malloc(total_len * hidden * sizeof(float));
    enc_gqa_mixed_attention(attn_out, Q, K, V, visual_len, total_len,
                            n_q_heads, n_kv_heads, head_dim);

    /* Debug: dump attention output for layer 0 */
    if (layer_idx == 0 && getenv("DS_DUMP_TENSORS")) {
        ds_dump_tensor("enc0_attn_out", attn_out, total_len * hidden, "[288, 896]");
    }

    /* Output projection + residual */
    float *proj_out = (float *)malloc(total_len * hidden * sizeof(float));
    ds_linear_nobias(proj_out, attn_out, wo_w, total_len, hidden, hidden);
    /* 残差一：注意力分支输出加回主路（out 与 x 是不同缓冲，此处首写 out） */
    for (int i = 0; i < total_len * hidden; i++) out[i] = x[i] + proj_out[i];

    /* Pre-FFN RMSNorm + SwiGLU MLP */
    float *ffn_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(ffn_norm, out, ln2_w, total_len, hidden, cfg->dec_rms_norm_eps);

    /* Gate + Up */
    float *gate_buf = (float *)malloc(total_len * intermediate * sizeof(float));
    float *up_buf = (float *)malloc(total_len * intermediate * sizeof(float));
    ds_linear_nobias(gate_buf, ffn_norm, gate_w, total_len, hidden, intermediate);
    ds_linear_nobias(up_buf, ffn_norm, up_w, total_len, hidden, intermediate);

    /* SwiGLU: silu(gate) * up (float64 for numerical stability)
     * silu(g) = g·σ(g)，σ 为 sigmoid；elementwise 乘 up 分支。
     * exp 的 double 版本防 24 层累积漂移（与 enc_linear_f64acc 同一动机） */
    for (int i = 0; i < total_len * intermediate; i++) {
        double g = (double)gate_buf[i];
        double sigmoid_g = 1.0 / (1.0 + exp(-g));
        gate_buf[i] = (float)(g * sigmoid_g * (double)up_buf[i]);
    }
    free(up_buf);

    /* Down projection + residual（FFN 分支输出加回主路，本层结束） */
    float *down_out = (float *)malloc(total_len * hidden * sizeof(float));
    ds_linear_nobias(down_out, gate_buf, down_w, total_len, intermediate, hidden);
    for (int i = 0; i < total_len * hidden; i++) out[i] += down_out[i];

    free(x_norm); free(Q); free(K); free(V);
    free(attn_out); free(proj_out); free(ffn_norm);
    free(gate_buf); free(down_out);
}

/* ds_encoder_forward_v2 - DeepEncoder V2 前向（V2 专用；完整契约见 ds_deep_encoder.h）
 * 【流程】[visual_tokens; causal_queries] 拼接 [total,896]
 *   → 24 层 enc_v2_layer_forward（混合注意力）→ 最终 RMSNorm
 *   → 只取因果查询段（x_norm 后 n_causal_queries 行）→ projector(896→1280)
 * 【为什么只取查询段】视觉 token 是"被阅读的内容"，因果流查询才是"汇报结果的
 *   人"——查询经因果注意力逐步吸收视觉信息，其输出即逐位置压缩的视觉摘要，
 *   直接对接解码器的自回归生成（docs/06 §9）。
 * 【参数】visual_tokens: [n_tokens,896]；causal_queries: [n_causal_queries,896]
 *   （checkpoint 的 query_1024/query_768 嵌入表）；out_seq_len: [输出]=n_causal_queries
 * 【返回】[n_causal_queries, 1280]（调用者 free）
 *
 * 【说人话】这是"读者 + 记者"的协作。前 256 个 token 是"读者"（视觉 token），
 *   它们互相自由交流（双向注意力），把图看透；后 256 个 token 是"记者"
 *   （因果流查询），它们只能按顺序一个个发言，每个记者都能听到前面所有
 *   读者和记者的内容，然后写一段摘要。
 *   为什么只取记者那一段？因为读者只是"看过图"，记者才是"把图讲成一段
 *   可被下游生成模型直接消费的话"。记者的输出天然是因果有序的（第 i 个
 *   记者只依赖前 i-1 个），正好对接解码器的自回归生成。
 * 【边界】"读者"和"记者"在同一个序列里，靠注意力掩码区分角色——前段
 *   双向、后段因果。掩码写错会让记者偷看后面的记者（信息泄漏），或者
 *   让读者看不到全部视觉信息（压缩不充分），两种都是静默错误。
 *   另外 256 这个数字是 checkpoint 里 query_1024/query_768 嵌入表的行数，
 *   不是超参——换模型必须跟着换，写死会越界。 */
float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len,
                               int n_causal_queries, const float *causal_queries) {
    ds_deep_encoder_t *enc = &ctx->encoder;
    ds_config_t *cfg = &ctx->config;

    int visual_len = n_tokens;                    /* 双向段长度（前 visual_len 行视觉 token） */
    int total_len = visual_len + n_causal_queries; /* 1024² 输入 = 256+256 = 512 */
    int hidden = cfg->enc_hidden;                  /* 896 */

    if (ds_verbose >= 1)
        fprintf(stderr, "DeepEncoder V2: %d visual + %d causal flow = %d total\n",
                visual_len, n_causal_queries, total_len);

    /* Concatenate visual tokens and causal flow queries
     * 序列布局: [视觉 token × visual_len | 因果流查询 × n_causal_queries]
     * 前段双向互见、后段因果收集——一个序列同时承载两种注意力角色 */
    float *x = (float *)calloc(total_len * hidden, sizeof(float));
    memcpy(x, visual_tokens, visual_len * hidden * sizeof(float));

    /* 无查询的退化调用（n_causal_queries=0）直接跳过拷贝——序列只剩视觉段，
     * 全员双向注意力，输出段为空 */
    if (n_causal_queries > 0 && causal_queries) {
        memcpy(x + visual_len * hidden,
               causal_queries,
               n_causal_queries * hidden * sizeof(float));
    }

    /* DUMP: encoder input (visual + causal queries concatenated) */
    ds_dump_tensor("enc_input", x, total_len * hidden, "[512, 896]");

    /* Optional: override encoder input with Python reference for debugging.
     * This isolates encoder precision from SAM/resize differences.
     * Usage: DS_LOAD_ENC_INPUT=dump/py_enc_input_crop0.bin */
    {
        const char *load_enc_input = getenv("DS_LOAD_ENC_INPUT");
        if (load_enc_input) {
            FILE *ef = fopen(load_enc_input, "rb");
            if (ef) {
                fread(x, sizeof(float), total_len * hidden, ef);
                fclose(ef);
                fprintf(stderr, "Loaded encoder input from %s (%d x %d)\n",
                        load_enc_input, total_len, hidden);
                /* Re-dump to confirm */
                ds_dump_tensor("enc_input_loaded", x, total_len * hidden, "[512, 896]");
            } else {
                fprintf(stderr, "Warning: DS_LOAD_ENC_INPUT=%s not found\n", load_enc_input);
            }
        }
    }

    /* Transformer 层循环（cfg->enc_layers=24）：每层输出写新缓冲 next——
     * 层内残差要同时读输入 x，不能原地覆写（与 CLIP 循环同一模式） */
    for (int l = 0; l < cfg->enc_layers; l++) {
        float *next = (float *)malloc(total_len * hidden * sizeof(float));
        enc_v2_layer_forward(next, x, ctx, l, visual_len, total_len);
        free(x);
        x = next;

        /* DUMP: each encoder layer output */
        {
            char dump_name[64];
            snprintf(dump_name, sizeof(dump_name), "enc_layer%d", l);
            ds_dump_tensor(dump_name, x, total_len * hidden, "[512, 896]");
        }
    }

    /* Final norm
     * Qwen2 风格：24 层之后、取输出之前的最终 RMSNorm（对全部 total_len 行做，
     * 之后只挑查询段——视觉段结果将被丢弃） */
    float *x_norm = (float *)malloc(total_len * hidden * sizeof(float));
    ds_rms_norm(x_norm, x, enc->final_norm_weight, total_len, hidden, cfg->dec_rms_norm_eps);
    free(x);

    /* DUMP: after final norm (full [512, 896] including visual + causal) */
    ds_dump_tensor("enc_final_norm", x_norm, total_len * hidden, "[512, 896]");

    /* Extract only causal flow query tokens, project to decoder dim
     * 指针偏移 visual_len×hidden 即跳过视觉段；V2 的 projector 是
     * [1280, 896] 单层线性（SAM 维度 896 直接投影，无 CLIP 拼接） */
    int output_len = n_causal_queries;
    float *encoder_output = (float *)malloc(output_len * cfg->dec_hidden * sizeof(float));

    if (ctx->projector.weight) {
        ds_linear(encoder_output, x_norm + visual_len * hidden,
                  ctx->projector.weight, ctx->projector.bias,
                  output_len, hidden, cfg->dec_hidden);
    } else {
        /* 无 projector 权重时直接透传 896 维（仅调试场景，维度与解码器不匹配） */
        memcpy(encoder_output, x_norm + visual_len * hidden,
               output_len * hidden * sizeof(float));
    }

    free(x_norm);

    /* DUMP: projector output [256, 1280] */
    ds_dump_tensor("proj_output", encoder_output, output_len * cfg->dec_hidden, "[256, 1280]");
    *out_seq_len = output_len;

    if (ds_verbose >= 1) {
        fprintf(stderr, "DeepEncoder V2: %d in -> %d out\n", total_len, output_len);
        /* Debug: print first few values of encoder output
         * 输出健康检查：mean_abs/max_abs 异常（全 0 或量级爆炸）通常意味着
         * 上游 SAM/projector 数值坏了，先看这行日志再谈精度 */
        float sum = 0, max_val = 0;
        for (int i = 0; i < output_len * cfg->dec_hidden; i++) {
            float v = encoder_output[i] < 0 ? -encoder_output[i] : encoder_output[i];
            sum += v;
            if (v > max_val) max_val = v;
        }
        fprintf(stderr, "  encoder output: mean_abs=%.4f max_abs=%.4f first=[%.4f, %.4f, %.4f, %.4f]\n",
                sum / (output_len * cfg->dec_hidden), max_val,
                encoder_output[0], encoder_output[1], encoder_output[2], encoder_output[3]);
    }

    return encoder_output;
}

/* ========================================================================
 * Unified Encoder Forward (dispatches to V1 or V2)
 * ======================================================================== */

/* ds_encoder_forward - 统一编码器分发入口（按 enc_type 选 CLIP 或 DeepEncoder V2）
 * enc_type==1 → CLIP 路径：V3/V4 直接把 visual_tokens（SAM 特征）传进去，
 *   像素参数填 0；V1 由 ds_ocr.c 携带原始像素另行直调，不走这里。
 * enc_type==2 → DeepEncoder V2 路径（visual + 因果流查询拼接）。
 * 【参数】与两个被调函数一致（见 .h 契约）；不匹配的多余参数被忽略 */
float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len,
                           int n_causal_queries, const float *causal_queries) {
    if (ctx->config.enc_type == 1) {
        /* CLIP path — for V3, pass SAM features directly.
         * This unified entry point is not used for V1 (V1 goes through ds_ocr.c directly). */
        return ds_clip_encoder_forward(ctx, NULL, 0, 0, 0,
                                       visual_tokens, n_tokens, out_seq_len);
    } else {
        return ds_encoder_forward_v2(ctx, visual_tokens, n_tokens, out_seq_len,
                                      n_causal_queries, causal_queries);
    }
}

/* ds_encoder_load - 占位接口（空实现，恒返回 0）
 * 真实的编码器权重加载在 ds_ocr.c 的 ds_load() 中随整模型统一完成
 * （LOAD_F32 装入 ctx->clip_encoder / ctx->encoder / ctx->projector）。
 * 保留空函数只为公共 API 完整性；清单见 ds_deep_encoder.h 的同名注释。 */
int ds_encoder_load(ds_ctx_t *ctx) {
    (void)ctx;
    return 0;
}
