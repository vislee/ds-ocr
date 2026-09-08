/*
 * ds_image.h - Image loading for DeepSeek-OCR (PNG/JPEG/WebP/BMP via stb_image)
 * ds_image.h — DeepSeek-OCR 图像加载模块
 *
 * 【模块角色】
 * 在 ds-ocr 推理流水线中，图像加载是第一步——将磁盘上的图像文件解码为内存中的 RGB 像素数组，
 * 供后续的 SAM 视觉编码器处理。
 *
 * 【设计决策】
 * - 使用 stb_image.h（单头文件库）实现零依赖的图像解码
 * - 统一转为 RGB 3通道，简化下游处理逻辑
 * - 提供多种预处理方式：resize（拉伸）、pad（保持宽高比填充）、crop（裁剪）
 * - dynamic_preprocess 为 V2 多裁剪策略服务，将大图像分割为多个 768×768 局部裁剪
 *
 * 【与 qwen-asr 的对应】
 * qwen-asr 没有 ds_image 模块——它处理的是音频（Mel频谱图），而非图像。
 * 这是两个项目最大的输入差异：图像 vs 音频。
 */

#ifndef DS_IMAGE_H
#define DS_IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* ds_image_t — 图像数据结构
 *
 * pixels:  RGB像素数组，行主序排列，每像素3字节(R,G,B)
 *          布局: pixels[y * width * channels + x * channels + c]
 *          其中 c∈{0,1,2} 对应 R/G/B 通道
 *
 * width/height:  图像宽高（像素）
 *
 * channels:  通道数，加载后始终为3（RGB）
 *            即使原始图像是RGBA或灰度图，stb_image也会自动转换
 *
 * owns_stb:  内存所有权标志
 *            1 = pixels 由 stbi_load() 分配，需用 stbi_image_free() 释放
 *            0 = pixels 由 malloc() 分配（如crop/pad后），需用 free() 释放
 *            这个区分很重要：stb_image 使用自己的分配器，不能混用 free()
 */
typedef struct {
    unsigned char *pixels;  /* RGB, 3 channels, row-major */
    int width;
    int height;
    int channels;           /* Always 3 after conversion — 加载后始终为3通道RGB */
    int owns_stb;           /* 1 if pixels from stbi_load (free with stbi_image_free), 0 if malloc'd */
} ds_image_t;

/* ds_image_load — 从文件加载图像
 *
 * 支持格式: PNG, JPEG, WebP, BMP, TIFF, GIF, PSD 等（stb_image 支持的所有格式）
 *
 * 处理流程:
 *   1. stbi_load() 解码图像文件 → RGB像素数组
 *   2. 如果原始图像是RGBA(4通道)或灰度(1通道)，自动转为RGB(3通道)
 *   3. 返回 ds_image_t 结构体
 *
 * 返回: 新分配的 ds_image_t（调用者负责释放），失败返回NULL
 *
 * 【为什么不需要resize?】
 * 此函数只做解码，不做缩放——缩放由调用方根据模型版本选择：
 * - V1/V3: 调用方会 pad 到 1024×1024
 * - V2: dynamic_preprocess 生成多个 768×768 裁剪
 */
ds_image_t *ds_image_load(const char *path);

/* ds_image_resize — 双线性插值缩放图像
 *
 * 将图像缩放到指定尺寸（拉伸模式，不保持宽高比）
 * 使用双线性插值(bilinear interpolation)进行像素重采样
 *
 * 数学原理:
 *   对目标图像的每个像素(x,y)，映射回源图像坐标(x_src, y_src)，
 *   取周围4个像素的加权平均值作为输出值：
 *     out(x,y) = (1-α)(1-β)·src(x₀,y₀) + α(1-β)·src(x₁,y₀)
 *              + (1-α)β·src(x₀,y₁)   + αβ·src(x₁,y₁)
 *   其中 α,β 是小数部分的插值权重
 *
 * 使用场景: V1/V3 的全局图缩放到 1024×1024（stretch模式）
 */
ds_image_t *ds_image_resize(const ds_image_t *img, int target_width, int target_height);

/* ds_image_pad — 保持宽高比填充图像
 *
 * 将图像等比缩放并填充到目标尺寸，不改变原始宽高比。
 * 空白区域用 pad_color 填充（默认127=灰色）。
 *
 * 处理流程:
 *   1. 计算缩放因子 scale = min(target_w/width, target_h/height)
 *   2. 等比缩放图像到 (width*scale, height*scale)
 *   3. 居中放置在 target×target 画布上
 *   4. 空白区域填充 pad_color
 *
 * 这等价于 Python PIL 的 ImageOps.pad()，是 V3 的预处理方式：
 *   - V3 使用 pad（保持宽高比），而非 stretch（拉伸变形）
 *   - 因为 V3 的 CLIP 编码器对图像的空间结构更敏感
 *
 * 使用场景:
 *   - V2 全局图 pad 到 1024×1024
 *   - V3 全局图 pad 到 1024×1024 或 640×640
 *   - V2 局部裁剪 pad 到 768×768
 */
ds_image_t *ds_image_pad(const ds_image_t *img, int target_size, unsigned char pad_color);

/* ds_image_to_float_chw — 图像转F32张量（NCHW格式）
 *
 * 将 uint8 RGB像素 [0,255] 归一化为 float32 [0,1]
 * 输出格式: NCHW（通道在前）= [3, height, width]
 *
 * 输出布局:
 *   [R₀,R₁,...,R_{H*W-1}, G₀,G₁,...,G_{H*W-1}, B₀,B₁,...,B_{H*W-1}]
 *   即3个通道的数据在内存中连续排列，每个通道占 H*W 个float
 *
 * 为什么用 NCHW 而非 NHWC?
 *   - SAM 的 patch_embed 是 Conv2d(3→768, k=16, s=16)
 *   - ds_conv2d() 内核期望 NCHW 输入
 *   - 这与 PyTorch/CUDA 的默认布局一致
 *
 * 归一化: pixel_f32 = pixel_uint8 / 255.0
 * 注意: SAM 不做 ImageNet 标准化(mean/std)，只做 [0,1] 归一化
 */
float *ds_image_to_float_chw(const ds_image_t *img);

/* ds_image_crop_box — 矩形裁剪
 *
 * 从图像中裁剪指定矩形区域 [left, top, right, bottom)
 * 返回新的 ds_image_t（调用者负责释放）
 *
 * 使用场景: V2 的 dynamic_preprocess 将大图裁为多个局部区域
 */
ds_image_t *ds_image_crop_box(const ds_image_t *img, int left, int top, int right, int bottom);

/* ds_dynamic_preprocess — 动态多裁剪预处理（V2/V3 专用）
 *
 * 【这是 V2/V3 处理高分辨率图像的核心策略】
 *
 * 问题: V2 的 DeepEncoder V2 和 V3 的 CLIP 编码器对大图像的处理能力有限
 *       （单张 1024×1024 只能产生 256 个视觉 token，细节丢失严重）
 * 解法: 将大图像分割为多个小块(crop)，每块独立编码后拼接
 *
 * 算法流程:
 *   1. 计算最优裁剪网格 (best_grid)：
 *      遍历所有 (width_factor, height_factor) 组合，
 *      选择 w_f * h_f ∈ [min_num, max_num] 且宽高比最接近原图的组合
 *   2. 将图像裁剪为 w_f × h_f 块，每块 size×size
 *   3. 如果 use_thumbnail=1，额外追加一张缩略图
 *
 * 参数:
 *   image_size — 裁剪块尺寸（V2=768, V3=640）
 *   min_num — 最少裁剪数（Python: V2=2, V3=2；早期C实现V2=1有bug，已修正为2）
 *   max_num — 最多裁剪数（V2=6, V3=32）
 *   use_thumbnail — 是否追加全局缩略图（V2=0, V3=0；缩略图由调用方单独生成）
 *
 * 返回: ds_image_t** 数组（每个元素是一个裁剪图像），*out_count = 裁剪数量
 *       调用者需逐个 ds_image_free() 并 free() 数组本身
 *
 * 【与 Python 的对应】
 * 对应 DeepSeek-OCR Python 代码中的 dynamic_preprocess() 函数
 * 关键参数对齐: min_num=2（不是1！），max_num=6（V2）/ 32（V3）
 */
ds_image_t **ds_dynamic_preprocess(const ds_image_t *img, int image_size,
                                    int min_num, int max_num,
                                    int use_thumbnail, int *out_count);

/* ds_image_free — 释放图像资源
 *
 * 根据 owns_stb 标志选择正确的释放函数:
 *   owns_stb=1 → stbi_image_free() (stb_image 的专用释放器)
 *   owns_stb=0 → free() (标准C释放器)
 *
 * 最后 free(img) 释放结构体本身
 */
void ds_image_free(ds_image_t *img);

#endif /* DS_IMAGE_H */
