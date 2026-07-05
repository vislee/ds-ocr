/*
 * ds_visual_tokenizer.h - SAM Vision Tokenizer for DeepSeek-OCR
 * ds_visual_tokenizer.h — SAM 视觉分词器
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】SAM ViT-B 是 ds-ocr 的"眼睛"
 * ─────────────────────────────────────────────────────────────────────
 * 将输入图像的原始像素转换为视觉特征序列（visual tokens），
 * 供后续的编码器（CLIP 或 DeepEncoder V2）进一步精炼。
 *
 * 【SAM ViT-B 架构】
 * 输入: [3, H, W] RGB图像
 *   ↓ Patch Embed: Conv2d(3→768, kernel=16, stride=16)  ← 将图像切为16×16的patch
 *   ↓ + 位置编码 [577, 768]                              ← 1个CLS + 576个patch(24×24)
 *   ↓ 12层 Transformer (window attention + global attention)
 *   │   - 窗口注意力 (window_size=14): 层0,1,3,4,6,7,9,10 — 局部注意力
 *   │   - 全局注意力: 层2,5,8,11 — 所有patch互相可见
 *   │   - 相对位置编码 (rel_pos_h, rel_pos_w): 窗口注意力专用
 *   ↓ SAM Neck: Conv2d(768→256) + LN + Conv2d(256→256) + LN  ← 通道压缩
 *   ↓ SAM Downsample: Conv2d(256→512, k3,s2,p1) + Conv2d(512→1024/896, k3,s2,p1) ← 4×空间压缩
 * 输出: [n_tokens, sam_ds2_dim] 视觉特征
 *   - 1024×1024 输入 → 256 tokens (V1: dim=1024, V2: dim=896)
 *   - 768×768 输入 → 144 tokens (V2多裁剪模式)
 *
 * 【三版本差异】
 * V1: SAM输出 [256, 1024] → 与CLIP特征拼接 → [256, 2048] → Projector
 * V2: SAM输出 [256, 896]  → DeepEncoder V2 → [256, 896] → Projector
 * V3: SAM输出 [256, 1024] → 与CLIP特征拼接 → [256, 2048] → Projector
 *
 * 【与 qwen-asr 的对应】
 * qwen-asr 没有视觉分词器——它用 Conv2D stem + Transformer encoder 处理 Mel 频谱图。
 * 两者都是"将原始输入转换为 token 序列"，但一个是视觉 patch，一个是音频帧。
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_VISUAL_TOKENIZER_H
#define DS_VISUAL_TOKENIZER_H

#include "ds_ocr.h"
#include "ds_image.h"

/* ds_sam_forward — SAM ViT-B 前向传播（核心函数）
 *
 * 输入: RGB像素数据 (width × height × 3 channels, uint8)
 * 输出: SAM空间特征 [n_sam_tokens, sam_ds2_dim] (float32, 调用者需free)
 *
 * 处理流程:
 *   1. 像素归一化: uint8 [0,255] → float32 [0,1]
 *   2. Patch Embed: Conv2d(3→768, k=16, s=16) → [768, H/16, W/16]
 *   3. 添加位置编码 (+ sam_pos_embed)
 *   4. 12层 Transformer (window/global attention + FFN)
 *   5. SAM Neck: Conv2d(768→256) + LN + Conv2d(256→256) + LN
 *   6. SAM Downsample: Conv2d(256→512, s=2) + Conv2d(512→sam_ds2_dim, s=2)
 *      空间维度缩小4倍: H/16/4 × W/16/4 = n_tokens
 *
 * out_n_tokens: 输出token数 (1024×1024 → 256, 768×768 → 144)
 * out_patch_embeds: SAM patch embedding 输出（V1时传给CLIP作为输入）
 *   V1/V3: CLIP不使用自己的patch_embedding Conv2d，而是直接接收SAM的patch_embeds
 *   V2: 不需要（DeepEncoder V2 接收SAM neck+downsample后的特征）
 */
float *ds_sam_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                       int width, int height, int channels,
                       int *out_n_tokens, float **out_patch_embeds);

/* ds_visual_tokenizer_forward — 完整视觉分词器前向传播
 *
 * 封装 ds_sam_forward()，额外返回resize后的像素数据（供V1 CLIP使用）
 *
 * out_resized_pixels: 如果非NULL，返回resize到1024×1024后的RGB像素
 *   V1路径: CLIP需要原始RGB像素（通过自己的Conv2d patch_embedding处理）
 *   V2路径: 不需要（DeepEncoder直接接收SAM特征）
 *   调用者需free返回的像素数组
 */
float *ds_visual_tokenizer_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                                    int width, int height, int channels,
                                    int *out_n_tokens, float **out_patch_embeds,
                                    unsigned char **out_resized_pixels,
                                    int *out_resized_w, int *out_resized_h);

/* ds_sam_forward_image — 从 ds_image_t 前向传播（支持可变输入尺寸）
 *
 * 与 ds_sam_forward() 的区别: 接受 ds_image_t 而非原始像素指针
 * 主要用于 V2 多裁剪编码，每个裁剪可能是 768×768
 *
 * 【位置编码插值】
 * SAM 预训练位置编码为 577 个 (1 CLS + 576 patches = 24×24)
 * 当输入为 768×768 时，产生 48×48=2304 个patch（而非24×24=576）
 * 需要对位置编码做双三次插值(bicubic interpolation)：
 *   从 [24, 24] 插值到 [48, 48] → 2304 个位置编码 + 1 CLS = 2305
 * 这个插值在 SAM 内部自动完成，通过加载预计算的插值文件实现
 * （model_dir/interp_XX.npy 或运行时计算）
 */
float *ds_sam_forward_image(ds_ctx_t *ctx, const ds_image_t *img,
                             int *out_n_tokens, float **out_patch_embeds);

/* ds_visual_tokenizer_load — 加载视觉分词器权重
 *
 * 从 safetensors 文件中加载 SAM ViT-B 的所有权重：
 *   - patch_embed 卷积权重和偏置
 *   - 位置编码
 *   - 12层 Transformer 的 QKV/投影/FFN/归一化/相对位置编码
 *   - SAM Neck 的卷积和归一化权重
 *   - SAM Downsample (net_2, net_3) 的卷积权重
 *   - image_newline 和 view_seperator 可学习token
 *   - V2的因果流查询(causal query)嵌入
 *
 * 权重存储格式: F32（SAM权重较小，预转换为F32更高效）
 * 这与解码器权重的 BF16 策略不同——因为SAM是批量计算，预转换F32减少on-the-fly开销
 */
int ds_visual_tokenizer_load(ds_ctx_t *ctx);

#endif /* DS_VISUAL_TOKENIZER_H */
