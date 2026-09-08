/*
 * ds_deep_encoder.h - Encoders for DeepSeek-OCR
 * ds_deep_encoder.h — DeepSeek-OCR 编码器模块
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】编码器是 SAM 和解码器之间的"翻译层"
 * ─────────────────────────────────────────────────────────────────────
 * SAM 提取的视觉特征偏底层（边缘、纹理），编码器进一步精炼为高层语义特征，
 * 使解码器更容易理解和生成 OCR 文本。
 *
 * 【两种编码器架构】
 *
 * V1/V3 — CLIP ViT-L/14（双编码器架构）
 * ┌─────────────────────────────────────────────────────┐
 * │ SAM 特征 [256, 1024]                                │
 * │     ↓ (CLIP绕过自身patch_embed，直接接收SAM特征)     │
 * │ CLIP ViT-L/14 (24层标准ViT, 16头, hidden=1024)      │
 * │     + CLS token → [257, 1024]                       │
 * │     + 位置编码 (可能需要插值)                        │
 * │     24层: LayerNorm → MHA → 残差 → LN → FFN → 残差  │
 * │     ↓ 移除CLS token → [256, 1024]                   │
 * │ Concat(SAM特征, CLIP特征) → [256, 2048]             │
 * │     ↓ Projector(2048→1280)                          │
 * │     + image_newline + view_seperator → 273 tokens   │
 * └─────────────────────────────────────────────────────┘
 *
 * V2 — DeepEncoder V2（Qwen2-0.5B 架构，因果流查询设计）
 * ┌─────────────────────────────────────────────────────┐
 * │ SAM 特征 [256, 896]                                 │
 * │     + 因果流查询 [256, 896] ← 可学习的查询向量       │
 * │     ↓ 24层 Transformer (混合注意力)                  │
 * │     │   视觉token: 双向注意力（充分交流）             │
 * │     │   因果查询: 因果注意力（按序收集信息）          │
 * │     ↓ 最终 RMSNorm → [256, 896]                     │
 * │     ↓ Projector(896→1280)                           │
 * │     + view_seperator → 257 tokens                   │
 * └─────────────────────────────────────────────────────┘
 *
 * 【CLIP接收SAM特征的设计解读】
 * V1/V3 中 CLIP 绕过了自身的 Conv2d patch_embedding (3→1024)，
 * 直接将 SAM 的 256 个视觉 token 作为输入。这是 DeepSeek-OCR 的创新设计：
 * - SAM 已经完成了"patch化 + 投影"工作，且包含多尺度信息
 * - CLIP 的 24 层 Transformer 在此基础上做"二次精炼"
 * - 相当于 SAM 提取底层特征，CLIP 提供高层语义理解
 *
 * 【混合注意力（DeepEncoder V2 独有）】
 * 视觉 token 使用双向注意力（可以互相看到），因果流查询使用因果注意力
 * （只能看到自己和之前的 token）。这种设计实现了从并行理解到序列生成的过渡。
 *
 * 【与 qwen-asr 的对应】
 * qwen-asr 的编码器 = Conv2D stem + Transformer encoder（标准音频编码器）
 * ds-ocr 的编码器 = SAM + (CLIP 或 Qwen2-0.5B)（视觉编码器）
 * 两者功能相同：将原始输入（图像/音频）转换为 token 序列供解码器使用。
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_DEEP_ENCODER_H
#define DS_DEEP_ENCODER_H

#include "ds_ocr.h"

/* ds_clip_encoder_forward — CLIP ViT-L/14 前向传播（V1/V3 专用）
 *
 * 两种调用模式:
 *
 * V1 模式: 传入 rgb_pixels（原始RGB像素），CLIP 使用自身的 patch_embedding
 *   处理流程: RGB像素 → Conv2d(3→1024, k=14, s=14) → CLIP Transformer → 输出
 *   传参: rgb_pixels != NULL, sam_features = NULL
 *
 * V3 模式: 传入 sam_features（SAM视觉特征），CLIP 绕过 patch_embedding
 *   处理流程: SAM特征 → 添加CLS → +位置编码 → CLIP Transformer → 输出
 *   传参: rgb_pixels = NULL, sam_features != NULL
 *   【V3为什么绕过patch_embed?】因为SAM已经完成了patch化+投影，CLIP做二次精炼
 *
 * 输出: 经过 Projector 投影后的 token 序列 [n_output_tokens, dec_hidden]
 *       dec_hidden = 1280（与解码器隐藏维度一致）
 *       输出包含 image_newline 和 view_seperator，可直接送入解码器
 *
 * out_seq_len: 输出序列总长度（含特殊token）
 *   - V1: 273 = 16行×(16patches+1newline) + 1view_sep
 *   - V3: 273 或 111（取决于输入尺寸）
 */
float *ds_clip_encoder_forward(ds_ctx_t *ctx,
                                const unsigned char *rgb_pixels, int width, int height, int channels,
                                const float *sam_features, int n_sam_tokens,
                                int *out_seq_len);

/* ds_encoder_forward_v2 — DeepEncoder V2 前向传播（V2 专用）
 *
 * 输入: SAM视觉token [n_tokens, enc_hidden]（enc_hidden=896）
 *       n_causal_queries: 因果流查询数量（1024×1024 → 256, 768×768 → 144）
 *       causal_queries: 因果流查询嵌入向量
 *
 * 处理流程:
 *   1. 拼接 [visual_tokens, causal_queries] → [n_tokens+n_queries, 896]
 *   2. 24层 Transformer（混合注意力: 视觉token双向, 因果查询因果）
 *      - GQA: 14个Q头共享2组KV头（7:1分组比）
 *      - SwiGLU FFN: intermediate=4864
 *      - RoPE位置编码: theta=1,000,000（支持长序列）
 *   3. 最终 RMSNorm
 *   4. Projector(896→1280) 投影到解码器维度
 *   5. 返回因果查询对应的输出 [n_queries, 1280]
 *
 * 输出: 编码器输出 [n_output_tokens, dec_hidden]（调用者需free）
 *
 * 【因果流查询的创新】
 * 传统编码器用双向注意力产生固定长度的输出。
 * DeepEncoder V2 加入因果流查询，通过因果注意力逐个"收集"视觉信息，
 * 输出天然具有顺序性，更适合解码器的自回归生成。
 * 类比: 双向注意力 = 全员开会讨论; 因果查询 = 逐个汇报总结
 */
float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len,
                               int n_causal_queries, const float *causal_queries);

/* ds_encoder_forward — 统一编码器前向传播（自动分发）
 *
 * 根据 ctx->config.enc_type 选择编码器:
 *   enc_type=1 → ds_clip_encoder_forward()（V1/V3 CLIP编码器）
 *   enc_type=2 → ds_encoder_forward_v2()（V2 DeepEncoder编码器）
 *
 * 这是 ds_ocr.c 中调用的统一入口，避免调用方关心版本差异
 */
float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len,
                           int n_causal_queries, const float *causal_queries);

/* ds_encoder_load — 加载编码器权重
 *
 * V1/V3: 加载 CLIP ViT-L/14 权重 (F32格式，较小，预转换高效)
 *   - class_embedding, patch_embedding, position_embedding
 *   - pre_layernorm
 *   - 24层 Transformer (QKV/投影/FFN/LayerNorm)
 *   - post_layernorm (可选)
 *
 * V2: 加载 DeepEncoder V2 权重 (F32格式)
 *   - 24层 Transformer (Q/K/V/O投影 + SwiGLU FFN + LayerNorm)
 *   - 最终 norm
 *   - 因果流查询嵌入 (query_1024, query_768)
 *
 * 公共: 加载 Projector 权重 (F32: weight[bias])
 *   - V1/V3: [1280, 2048] (CLIP 1024 + SAM 1024 拼接后投影)
 *   - V2: [1280, 896] (DeepEncoder输出直接投影)
 */
int ds_encoder_load(ds_ctx_t *ctx);

#endif /* DS_DEEP_ENCODER_H */
