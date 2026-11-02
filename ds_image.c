/*
 * ds_image.c - Image loading for DeepSeek-OCR
 * ds_image.c — DeepSeek-OCR 图像加载实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】推理流水线的"入口"
 * ─────────────────────────────────────────────────────────────────────
 * 将磁盘上的图像文件解码为RGB像素数组，提供多种预处理方式。
 *
 * 【关键函数】
 *   ds_image_load()          — 图像解码（PNG/JPEG/WebP等 → RGB uint8）
 *   ds_image_resize()        — 双三次重采样缩放（拉伸模式，PIL风格抗锯齿）
 *   ds_image_pad()           — 保持宽高比填充（V2/V3预处理方式）
 *   ds_image_crop_box()      — 矩形裁剪
 *   ds_dynamic_preprocess()  — V2/V3动态多裁剪（大图分割策略）
 *   ds_image_to_float_chw()  — uint8→float32 + NCHW转换（SAM输入格式）
 *
 * 【预处理策略对比】
 *   V1: resize(1024×1024) — 拉伸，不保持宽高比
 *   V2: pad(1024×1024) + dynamic_preprocess(768×768) — 保持比例+多裁剪
 *   V3: pad(1024×1024 或 640×640) + dynamic_preprocess(640×640) — 保持比例+多裁剪
 *
 * 【stb_image.h】
 * 单头文件图像解码库，只需 #define STB_IMAGE_IMPLEMENTATION 后 #include
 * 支持所有常见格式，零外部依赖，完美符合ds-ocr的设计哲学
 * Uses stb_image.h for decoding (single-header library).
 * ═══════════════════════════════════════════════════════════════════════
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "ds_image.h"
#include "ds_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__APPLE__) && defined(USE_APPLE_VISION)
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include <ImageIO/ImageIO.h>
#endif

#if defined(__APPLE__) && defined(USE_APPLE_VISION)
/* Apply an EXIF orientation (kCGImagePropertyOrientation values 2-8) to an
 * RGBA buffer by pixel remapping; returns a newly allocated buffer and
 * updates *w and *h (dims swap for orientations 5-8). Matches PIL's
 * ImageOps.exif_transpose, which the Python reference applies at load:
 *   2: flip H      3: rotate 180    4: flip V
 *   5: transpose   6: rotate 90 CW  7: transverse   8: rotate 90 CCW
 * Mappings verified against a labelled 3x2 source for all 7 values. */
static unsigned char *ds_exif_transform_rgba(const unsigned char *src, int *w, int *h,
                                             int orientation) {
    int sw = *w, sh = *h;                                  /* 源尺寸：EXIF记录的传感器原始方向 */
    int dw = (orientation >= 5) ? sh : sw;                 /* 5-8号是旋转/转置类，宽高互换 */
    int dh = (orientation >= 5) ? sw : sh;
    unsigned char *dst = (unsigned char *)malloc((size_t)dw * dh * 4);   /* 每像素4字节RGBA */
    if (!dst) return NULL;   /* 失败时调用方保持原始方向继续（比直接报错更宽容） */

    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            /* 反向映射：对目标的每个像素反查它在源图中的位置——
             * 保证输出无空洞、无重复，比正向散射好写且缓存友好 */
            int sx, sy;
            switch (orientation) {
                case 2: sx = sw - 1 - x; sy = y;          break;  /* 水平镜像 */
                case 3: sx = sw - 1 - x; sy = sh - 1 - y; break;  /* 旋转180° */
                case 4: sx = x;          sy = sh - 1 - y; break;  /* 垂直镜像 */
                case 5: sx = y;          sy = x;          break;  /* 转置 */
                case 6: sx = y;          sy = sh - 1 - x; break;  /* 顺时针90°（手机竖拍最常见） */
                case 7: sx = sw - 1 - y; sy = sh - 1 - x; break;  /* 反转置 */
                case 8: sx = sw - 1 - y; sy = x;          break;  /* 逆时针90° */
                default: sx = x;         sy = y;          break;  /* 1=原样（不会到达，调用方已过滤） */
            }
            /* 4字节整块拷贝：RGBA紧挨存放，字节偏移 = (行*宽 + 列) * 4 */
            memcpy(dst + ((size_t)y * dw + x) * 4, src + ((size_t)sy * sw + sx) * 4, 4);
        }
    }
    *w = dw;   /* 回写变换后的尺寸——旋转类变换宽高互换，调用方后续按新尺寸用 */
    *h = dh;
    return dst;
}
#endif

/* ds_image_load — 从磁盘解码图像为 RGB 像素数组（推理流水线第一步）
 * (docs/06 §1)
 *
 * 双路径实现：
 *   macOS 且编译期开启 USE_APPLE_VISION → CoreGraphics/ImageIO 解码，
 *   行为对齐 PIL（含 EXIF 方向自动转正——手机拍照的照片常带方向标签，
 *   不转正会横竖颠倒）；CG 路径任一环节失败都 goto fallback_stb 退回
 *   stb_image 解码，保证任何平台都能出图（代价：EXIF 可能不转正）。
 *
 * 统一输出：RGB 3通道、HWC 行主序（pixels[y*width*3 + x*3 + c]），
 * 灰度/RGBA/调色板图都由解码器统一转成 3 通道。
 *
 * @param path  图像文件路径
 * @return      新分配的 ds_image_t（调用方用 ds_image_free 释放），失败返回NULL */
ds_image_t *ds_image_load(const char *path) {
#if defined(__APPLE__) && defined(USE_APPLE_VISION)
    /* Use CoreGraphics for image loading — matches PIL's JPEG/PNG decoder
     * 路径 → CFURL → CGImageSource 三步拿到解码器句柄，逐个失败点都退回 stb */
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, strlen(path), FALSE);
    if (!url) goto fallback_stb;
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
    CFRelease(url);              /* CF对象引用计数式管理：用完即放，防泄漏 */
    if (!source) goto fallback_stb;

    /* EXIF orientation must be read from the source BEFORE it is released
     * 方向标签存在"源"的属性字典里（而非解码后的位图），必须趁 source 还活着先读 */
    int exif_orientation = 1;    /* 1 = 无旋转的正常方向；2-8 = 各种翻转/旋转 */
    {
        CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
        if (props) {
            CFNumberRef o = (CFNumberRef)CFDictionaryGetValue(props, kCGImagePropertyOrientation);
            int v = 0;
            if (o && CFNumberGetValue(o, kCFNumberIntType, &v) && v >= 1 && v <= 8)
                exif_orientation = v;   /* 超出1-8视为损坏标签，按1处理 */
            CFRelease(props);
        }
    }

    CGImageRef cg_img = CGImageSourceCreateImageAtIndex(source, 0, NULL);   /* 解码第0帧 */
    CFRelease(source);
    if (!cg_img) goto fallback_stb;

    int w = (int)CGImageGetWidth(cg_img);
    int h = (int)CGImageGetHeight(cg_img);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();   /* 设备RGB色彩空间 */
    if (!cs) { CGImageRelease(cg_img); goto fallback_stb; }

    /* Create RGBA bitmap
     * CoreGraphics 不支持每像素3字节的位图格式，只能先落到 RGBA 再紧缩成 RGB。
     * kCGImageAlphaNoneSkipLast = 每像素4字节、前3字节RGB、第4字节是忽略的填充
     * （RGBX）；bitsPerComponent=8，bytesPerRow=w*4 无行对齐填充。 */
    CGContextRef ctx = CGBitmapContextCreate(NULL, w, h, 8, w * 4, cs,
        kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) { CGImageRelease(cg_img); goto fallback_stb; }

    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg_img);   /* 解码像素真正落进位图内存 */
    unsigned char *rgba = (unsigned char *)CGBitmapContextGetData(ctx);   /* 指向ctx内部的像素 */

    /* Apply EXIF orientation (no-op for 1); rotated buffer replaces rgba */
    unsigned char *rotated = NULL;
    if (exif_orientation > 1) {
        rotated = ds_exif_transform_rgba(rgba, &w, &h, exif_orientation);
        if (rotated) {
            rgba = rotated;   /* 换成堆上新缓冲（尺寸可能已随旋转变化） */
            if (ds_verbose >= 2)
                fprintf(stderr, "Applied EXIF orientation %d (%dx%d)\n", exif_orientation, w, h);
        }
    }

    /* Convert RGBA → RGB */
    unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3);
    if (!rgb) {
        free(rotated); /* NULL when no transform happened */
        CGContextRelease(ctx);
        CGImageRelease(cg_img);
        return NULL;
    }
    for (int i = 0; i < w * h; i++) {
        /* 每像素4字节取前3字节：RGBA 紧缩为 RGB，丢弃的第4字节是填充位 */
        rgb[i*3+0] = rgba[i*4+0];
        rgb[i*3+1] = rgba[i*4+1];
        rgb[i*3+2] = rgba[i*4+2];
    }
    free(rotated); /* heap-owned transformed buffer, distinct from ctx-owned rgba */
    CGContextRelease(ctx);   /* frees the original rgba */
    CGImageRelease(cg_img);

    ds_image_t *img = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!img) { free(rgb); return NULL; }
    img->pixels = rgb;
    img->width = w;
    img->height = h;
    img->channels = 3;
    img->owns_stb = 0;   /* CG路径像素来自malloc，释放走 free() 而非 stbi_image_free */
    if (ds_verbose >= 2)
        fprintf(stderr, "Loaded image (CG): %s (%dx%d)\n", path, w, h);
    return img;

fallback_stb:
#endif
    /* stb_image 通用回退路径：跨平台、零依赖，是所有非macOS平台的唯一解码器 */
    { int w, h, channels;
    /* 第4参数 desired_channels=3：无论源是灰度/RGBA/CMYK，stb 一律展开成
     * 3通道RGB——调用方从此不必关心源格式；w/h/channels 收到的是源图信息 */
    unsigned char *data = stbi_load(path, &w, &h, &channels, 3); /* Force 3 channels (RGB) */
    if (!data) {
        /* stbi_failure_reason() 返回线程局部的失败描述（文件不存在/格式不支持/解码错误） */
        fprintf(stderr, "ds_image_load: failed to load '%s': %s\n", path, stbi_failure_reason());
        return NULL;
    }

    ds_image_t *img = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!img) { stbi_image_free(data); return NULL; }

    img->pixels = data;
    img->width = w;
    img->height = h;
    img->channels = 3;
    img->owns_stb = 1;  /* pixels from stbi_load, must use stbi_image_free */

    if (ds_verbose >= 2)
        fprintf(stderr, "Loaded image: %s (%dx%d, %d channels)\n", path, w, h, channels);

    return img;
    }
}

/* Bicubic interpolation kernel (Keys cubic, a=-0.5).
 * Matches PIL/Pillow Image.Resampling.BICUBIC default behavior.
 * This is the W(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1 for |x|<=1
 *              = a|x|^3 - 5a|x|^2 + 8a|x| - 4a          for 1<|x|<=2
 *              = 0                                        for |x|>2
 * with a = -0.5 (Keys/Bicubic). */
static inline float ds_cubic_weight(float x) {
    float ax = fabsf(x);   /* Keys 核只依赖距离|x|，左右对称 */
    if (ax <= 1.0f) {
        /* ((a+2)*|x| - (a+3))*|x|^2 + 1 —— 内段：在 x=0 处取1、x=±1 处平滑过零 */
        return ((-0.5f + 2.0f) * ax - (-0.5f + 3.0f)) * ax * ax + 1.0f;
    } else if (ax <= 2.0f) {
        /* (a*|x| - 5a)*|x|^2 + 8a*|x| - 4a —— 外段：a=-0.5 < 0 产生负权重"负瓣"，
         * 带来轻微锐化（插值结果可以过冲出邻域值范围，调用方负责钳位） */
        return ((-0.5f * ax - 5.0f * (-0.5f)) * ax * ax
                + 8.0f * (-0.5f) * ax - 4.0f * (-0.5f));
    }
    return 0.0f;   /* |x|>2：紧支撑核，4×4 邻域之外不参与加权 */
}

/* PIL-style antialias bicubic interpolation for a single channel value.
 *
 * For DOWNSAMPLING (dst < src), PIL expands the filter kernel:
 *   scale_factor = min(1.0, dst_size / src_size)
 *   filter_support = 2.0 / scale_factor  (wider kernel)
 *   weight = cubic_weight((src_pixel - center) * scale_factor)
 *   normalize so weights sum to 1
 *
 * For UPSAMPLING (dst >= src), this reduces to standard 4x4 bicubic.
 *
 * This matches PIL/Pillow >= 7.0 default BICUBIC behavior (antialias=True). */
static inline float ds_bicuberp_aa(const unsigned char *src, int src_w, int src_h,
                                    float fx, float fy, int c,
                                    float x_scale_factor, float y_scale_factor) {
    /* Compute filter support width (in source pixels)
     * scale_factor = dst/src：下采样时 <1，除以它把核从 ±2 展宽到 ±2/scale——
     * 每个目标像素平均更多源像素，正是抗锯齿（antialias）的来源 */
    float x_support = 2.0f / x_scale_factor;  /* wider for downsampling */
    float y_support = 2.0f / y_scale_factor;

    /* 参与加权的源像素包围盒（support 覆盖到的整数坐标） */
    int ix_min = (int)floorf(fx - x_support);
    int ix_max = (int)ceilf(fx + x_support);
    int iy_min = (int)floorf(fy - y_support);
    int iy_max = (int)ceilf(fy + y_support);

    /* Clamp to image bounds —— 边界外不参与；被截掉的权重靠下面的归一化补偿，
     * 保证边界像素不因邻居缺失而变暗/变亮 */
    if (ix_min < 0) ix_min = 0;
    if (ix_max >= src_w) ix_max = src_w - 1;
    if (iy_min < 0) iy_min = 0;
    if (iy_max >= src_h) iy_max = src_h - 1;

    float result = 0.0f;   /* 加权和累加器 */
    float w_sum = 0.0f;    /* 权重和累加器 */

    for (int iy = iy_min; iy <= iy_max; iy++) {
        /* 二维核可分离：权重 = x方向分量 × y方向分量，两重循环各算一次一维核 */
        float wy = ds_cubic_weight(((float)iy - fy) * y_scale_factor);
        if (wy == 0.0f) continue;   /* 整行为零权重：跳过省掉一整行乘法 */

        for (int ix = ix_min; ix <= ix_max; ix++) {
            float wx = ds_cubic_weight(((float)ix - fx) * x_scale_factor);
            if (wx == 0.0f) continue;

            float w = wx * wy;
            /* src 是 HWC 布局：(行*宽+列)*3 + 通道号 定位到指定通道的字节 */
            result += w * (float)src[(iy * src_w + ix) * 3 + c];
            w_sum += w;
        }
    }
    /* Normalize —— 权重和因边界截断/负瓣不恒为1，除一下恢复"加权平均"语义 */
    if (w_sum > 0.0f) result /= w_sum;
    return result;
}

/* ds_image_resize - 高质量重采样缩放（拉伸模式，不保纵横比）
 * 用于 dynamic_preprocess 第③步：整图 resize 到 (cols·size, rows·size)
 * ——目标尺寸本来就是按原图比例选的，拉伸量可忽略。
 * 实现是 PIL 风格抗锯齿双三次（bicubic）插值，不是双线性：优先走
 * CoreGraphics（输出与 PIL 的 BICUBIC 一致），失败才退回内置
 * ds_bicuberp_aa 逐像素计算；"源坐标 = 目标坐标 × 缩放比"的映射不变。 */
ds_image_t *ds_image_resize(const ds_image_t *img, int target_width, int target_height) {
    if (!img || !img->pixels) return NULL;

#if defined(__APPLE__) && defined(USE_APPLE_VISION)
    /* Use CoreGraphics for high-quality resize matching PIL bicubic.
     * 整条 CG 链路层层嵌套 if：任一环失败就落到函数尾部的内置实现。 */
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (cs) {
        /* Create source image from raw RGB pixels
         * 先把 RGB3字节 展开成 RGBA4字节——CoreGraphics 只认4字节像素格式 */
        unsigned char *rgba_src = (unsigned char *)malloc(img->width * img->height * 4);
        if (rgba_src) {
            for (int i = 0; i < img->width * img->height; i++) {
                rgba_src[i*4+0] = img->pixels[i*3+0];
                rgba_src[i*4+1] = img->pixels[i*3+1];
                rgba_src[i*4+2] = img->pixels[i*3+2];
                rgba_src[i*4+3] = 255;   /* alpha 填不透明，不参与插值语义 */
            }
            CGContextRef src_ctx = CGBitmapContextCreate(NULL, img->width, img->height, 8,
                img->width * 4, cs, kCGImageAlphaNoneSkipLast);
            if (src_ctx) {
                /* 第1参数传NULL=让CG自分配位图内存；这里把展开好的像素灌进去 */
                memcpy(CGBitmapContextGetData(src_ctx), rgba_src, img->width * img->height * 4);
                CGImageRef src_img = CGBitmapContextCreateImage(src_ctx);   /* 位图 → CGImage 快照 */
                CGContextRelease(src_ctx);
                if (src_img) {
                    CGContextRef dst_ctx = CGBitmapContextCreate(NULL, target_width, target_height, 8,
                        target_width * 4, cs, kCGImageAlphaNoneSkipLast);
                    if (dst_ctx) {
                        /* kCGInterpolationHigh：最高插值档（近似双三次），对齐 PIL 输出 */
                        CGContextSetInterpolationQuality(dst_ctx, kCGInterpolationHigh);
                        CGContextDrawImage(dst_ctx, CGRectMake(0, 0, target_width, target_height), src_img);
                        CGImageRef dst_img = CGBitmapContextCreateImage(dst_ctx);
                        CGContextRelease(dst_ctx);
                        CGImageRelease(src_img);
                        if (dst_img) {
                            /* CGImage 的像素不能直接读：经 DataProvider 拷成连续缓冲 */
                            CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(dst_img));
                            const unsigned char *ptr = CFDataGetBytePtr(data);
                            size_t bpr = CGImageGetBytesPerRow(dst_img);   /* 行跨距：
                             * CG 可能做行对齐填充，必须按 bpr 寻址，不能假设 = width*4 */
                            unsigned char *out = (unsigned char *)malloc(target_width * target_height * 3);
                            if (out) {
                                /* 按 bpr 取每行 → 跳过 alpha 紧缩回 RGB3字节 */
                                for (int y = 0; y < target_height; y++)
                                    for (int x = 0; x < target_width; x++) {
                                        int si = (int)(y * bpr + x * 4);
                                        out[(y*target_width+x)*3+0] = ptr[si+0];
                                        out[(y*target_width+x)*3+1] = ptr[si+1];
                                        out[(y*target_width+x)*3+2] = ptr[si+2];
                                    }
                                CFRelease(data); CGImageRelease(dst_img); CGColorSpaceRelease(cs); free(rgba_src);
                                ds_image_t *r = (ds_image_t *)calloc(1, sizeof(ds_image_t));
                                if (!r) { free(out); return NULL; }
                                r->pixels = out; r->width = target_width; r->height = target_height;
                                r->channels = 3; r->owns_stb = 0;
                                return r;   /* CG 成功路径到此返回 */
                            }
                            CFRelease(data); CGImageRelease(dst_img);
                        }
                    }
                    CGImageRelease(src_img);
                }
            }
            free(rgba_src);
        }
        CGColorSpaceRelease(cs);
    }
    /* CG failed, fall through to custom bicubic */
#endif
    /* Fallback: custom bicubic (less accurate vs PIL)
     * 输出缓冲：HWC 布局，每像素3字节 */
    unsigned char *out = (unsigned char *)malloc(target_width * target_height * 3);
    if (!out) return NULL;

    /* 缩放比 = 源/目标：目标图1像素覆盖源图的 x_scale 像素宽度 */
    float x_scale = (float)img->width / target_width;
    float y_scale = (float)img->height / target_height;
    /* PIL 的 scale_factor = 目标/源：>1 是放大、<1 是缩小 */
    float x_aa_factor = (float)target_width / (float)img->width;
    float y_aa_factor = (float)target_height / (float)img->height;
    /* 只有下采样（factor<1）才展宽滤波核；上采样钳回1 = 标准4×4双三次 */
    if (x_aa_factor > 1.0f) x_aa_factor = 1.0f;
    if (y_aa_factor > 1.0f) y_aa_factor = 1.0f;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            /* 像素中心对齐映射：目标第x个像素中心在 x+0.5，按缩放比落到源坐标，
             * 再减0.5换算回"源像素下标坐标系"（源像素i的中心在 i+0.5 处）。
             * 不做中心对齐会产生半像素系统性偏移，缩放后图像发虚。 */
            float fx = (x + 0.5f) * x_scale - 0.5f;
            float fy = (y + 0.5f) * y_scale - 0.5f;
            for (int c = 0; c < 3; c++) {
                float val = ds_bicuberp_aa(img->pixels, img->width, img->height, fx, fy, c,
                                           x_aa_factor, y_aa_factor);
                if (val < 0.0f) val = 0.0f;     /* 双三次负瓣会造成过冲（振铃），钳回合法区间 */
                if (val > 255.0f) val = 255.0f;
                out[(y * target_width + x) * 3 + c] = (unsigned char)(val + 0.5f);   /* +0.5 四舍五入 */
            }
        }
    }

    ds_image_t *resized = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!resized) { free(out); return NULL; }
    resized->pixels = out;
    resized->width = target_width;
    resized->height = target_height;
    resized->channels = 3;
    resized->owns_stb = 0;
    return resized;
}

/* Pad image to target size maintaining aspect ratio (like PIL ImageOps.pad).
 * Fills with pad_color (e.g. 128 for gray). Resizes largest dimension to fit. */
/* ds_image_pad - 等比缩放 + 居中填充（PIL ImageOps.pad 的 C 对应物）
 * (docs/06 §1) 三代模型全局图预处理的统一路径：长边缩到 target_size，
 * 短边居中，四周填 pad_color=127（即 0.5·255，Python mean color）。
 * 【为什么 127】预处理后像素会做 (x/255-0.5)/0.5 归一化，127≈0 恰是
 * 归一化后的均值——填充区域对模型是"无信息的中性灰"，不是黑或白。 */
ds_image_t *ds_image_pad(const ds_image_t *img, int target_size, unsigned char pad_color) {
    if (!img || !img->pixels) return NULL;

    /* Calculate scale to fit within target_size × target_size
     * 用长边算缩放因子：等比缩放后长边恰好 = target_size，短边 ≤ target_size */
    float scale = (float)target_size / (img->width > img->height ? img->width : img->height);
    int new_w = (int)(img->width * scale + 0.5f);   /* +0.5f 四舍五入，纯截断会系统性偏小 */
    int new_h = (int)(img->height * scale + 0.5f);
    if (new_w > target_size) new_w = target_size;   /* 舍入可能超1像素，钳回画布内 */
    if (new_h > target_size) new_h = target_size;

    /* First resize to fit */
    ds_image_t *resized = ds_image_resize(img, new_w, new_h);
    if (!resized) return NULL;

    /* Allocate target and fill with pad color
     * 先整块预填 pad_color：HWC 布局下每像素 RGB 三字节都是同一灰度值 */
    unsigned char *out = (unsigned char *)malloc(target_size * target_size * 3);
    if (!out) { ds_image_free(resized); return NULL; }   /* 失败时回收中间结果 */
    memset(out, pad_color, target_size * target_size * 3);

    /* Center the resized image
     * 居中偏移 = (画布 - 图) / 2，整数除法向下取整 */
    int offset_x = (target_size - new_w) / 2;
    int offset_y = (target_size - new_h) / 2;
    for (int y = 0; y < new_h; y++) {
        /* 逐行整块拷贝：HWC 下一行的 3*new_w 字节连续，一次 memcpy 顶逐像素循环 */
        memcpy(out + ((offset_y + y) * target_size + offset_x) * 3,
               resized->pixels + y * new_w * 3,
               new_w * 3);
    }
    ds_image_free(resized);   /* 中间结果即刻释放，峰值内存只多一份目标图 */

    ds_image_t *padded = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!padded) { free(out); return NULL; }
    padded->pixels = out;
    padded->width = target_size;
    padded->height = target_size;
    padded->channels = 3;
    padded->owns_stb = 0;  /* pixels from malloc, use free() */
    return padded;
}

/* ds_image_to_float_chw - uint8 HWC → float32 CHW + 归一化（SAM 输入格式）
 * 归一化 (pixel/255 - 0.5)/0.5 = pixel/127.5 - 1，匹配 Python
 * BasicImageTransform(mean=0.5, std=0.5)。CHW 平面布局是卷积核
 * ds_conv2d 期望的输入；HWC→CHW 的转置在此一次完成。 */
float *ds_image_to_float_chw(const ds_image_t *img) {
    if (!img || !img->pixels) return NULL;

    int n = img->width * img->height;   /* 单通道像素数：每个通道平面占 n 个float */
    float *out = (float *)malloc(3 * n * sizeof(float));   /* 3个通道平面 = 3n 个float */
    if (!out) return NULL;

    /* Convert HWC (interleaved RGB) to CHW (planar), normalize to [-1, 1]
     * Using mean=0.5, std=0.5: (pixel/255 - 0.5) / 0.5 = pixel/255*2 - 1
     * This matches Python's BasicImageTransform(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) */
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int hwc_idx = (y * img->width + x) * 3;   /* HWC：同一像素的R,G,B连续存放 */
            int chw_idx = y * img->width + x;         /* CHW：同一通道内按行主序连续 */
            for (int c = 0; c < 3; c++) {
                /* 目标下标 = 通道平面偏移 c*n + 平面内下标；归一化合并成一步
                 * pixel/255*2 - 1，数值上等价 (pixel/255-0.5)/0.5（mean=std=0.5） */
                out[c * n + chw_idx] = (float)img->pixels[hwc_idx + c] / 255.0f * 2.0f - 1.0f;
            }
        }
    }

    return out;
}

/* ds_image_crop_box — 裁剪矩形区域 [left, right) × [top, bottom)（半开区间）
 * dynamic_preprocess 第④步的底层操作：从 resize 后的整图上切出 size² 的 tile。
 * 像素坐标以左上角为原点，right/bottom 不包含在结果内（与 Python 切片一致）。
 *
 * @param img            源图像
 * @param left,top       裁剪框左上角（含）
 * @param right,bottom   裁剪框右下角（不含）
 * @return               新分配的裁剪图（调用方 ds_image_free），失败返回NULL */
ds_image_t *ds_image_crop_box(const ds_image_t *img, int left, int top, int right, int bottom) {
    if (!img || !img->pixels) return NULL;

    int crop_w = right - left;   /* 半开区间：宽 = right-left，高 = bottom-top */
    int crop_h = bottom - top;
    if (crop_w <= 0 || crop_h <= 0) return NULL;   /* 空框/参数非法 */

    unsigned char *out = (unsigned char *)malloc(crop_w * crop_h * 3);   /* HWC RGB */
    if (!out) return NULL;

    for (int y = 0; y < crop_h; y++) {
        /* 逐行 memcpy：源行起点 = 源图第 (top+y) 行内偏移 left 处，
         * 该处连续 crop_w*3 字节正是裁剪结果的一行（HWC 下行内像素连续） */
        memcpy(out + y * crop_w * 3,
               img->pixels + ((top + y) * img->width + left) * 3,
               crop_w * 3);
    }

    ds_image_t *cropped = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!cropped) { free(out); return NULL; }
    cropped->pixels = out;
    cropped->width = crop_w;
    cropped->height = crop_h;
    cropped->channels = 3;
    cropped->owns_stb = 0;   /* 像素来自malloc，释放走 free() */
    return cropped;
}

/* Find closest aspect ratio — matches Python find_closest_aspect_ratio() */
/* find_closest_aspect_ratio - 在候选网格里选最接近原图比例的块组合
 * (docs/06 §7)
 *
 * 候选集 = {(i,j) : min_num ≤ i·j ≤ max_num}，i=宽方向块数、j=高方向。
 * V2 调用 (768, 2, 6)：最多 6 块；V3/V4 调用 (640, 2, 32/16)：大图最多 32/16 块。
 *
 * 【tie-break 护栏】(docs/06 §7 真实细节) 2/3 与 4/6 的浮点比例完全相等，
 * 此时规则是"图够大才选块数更多的组合"：
 *   area > 0.5·image_size²·i·j 才升级
 * 1600×2264（面积 3.62M < 0.5·640²·24=4.92M）→ 不升级到 24 块。没有这条
 * 护栏，中等图片会被切成远超必要的 tile 数，token 成本翻四倍。 */
static int find_closest_aspect_ratio(float aspect_ratio, int width, int height,
                                       int image_size, int *out_rows, int *out_cols,
                                       int min_num, int max_num) {
    /* Generate target ratios: (i, j) where min_num <= i*j <= max_num
     * Python: target_ratios = set((i, j) ...) where i=width_blocks, j=height_blocks.
     * ratio[0]=i is width direction, ratio[1]=j is height direction.
     * V2: min_num=2, max_num=6 → at most 6 crops
     * V3: min_num=2, max_num=32 → up to 32 crops for large images */
    int best_width_blocks = 1, best_height_blocks = 1;   /* 兜底解：1×1 */
    float best_diff = 1e30f;   /* "正无穷"初值：保证第一个合法候选必定胜出 */
    long long area = (long long)width * height;   /* 64位乘积：极端长图下int会溢出 */

    /* 三重循环与 Python 的集合推导式同构：外层 n=候选块数，内层枚举 (i,j)
     * 分解；虽然允许 i·j≠n 的组合被重复枚举，但"更优才替换"的比较语义下无害 */
    for (int n = min_num; n <= max_num; n++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i * j < min_num || i * j > max_num) continue;   /* 块数约束 */
                float target_ar = (float)i / (float)j;  /* i=width, j=height */
                float diff = fabsf(aspect_ratio - target_ar);   /* 与原图比例的偏差 */
                if (diff < best_diff) {
                    best_diff = diff;
                    best_width_blocks = i;   /* width direction */
                    best_height_blocks = j;  /* height direction */
                } else if (diff == best_diff) {
                    /* Tie-break: prefer larger ratio if image area is large
                     * 护栏：图不够大就不升级到块数更多的组合（省token，见上文件头） */
                    if (area > (long long)(0.5f * image_size * image_size * i * j)) {
                        best_width_blocks = i;
                        best_height_blocks = j;
                    }
                }
            }
        }
    }

    *out_rows = best_height_blocks;  /* rows = height direction */
    *out_cols = best_width_blocks;   /* cols = width direction */
    return best_width_blocks * best_height_blocks;
}

/* ds_dynamic_preprocess - 动态多裁剪：NaViT"保比例+分块"的工程化
 * (docs/06 §7，五步逐步算例见文档)
 *
 * 【流程】① 算原图纵横比 ② find_closest_aspect_ratio 选 i×j 组合
 * ③ 整图 resize 到 (i·size, j·size)（双线性）④ 行主序等分成 i×j 个
 * size² tile（先左到右、再上到下）⑤ thumbnail 由调用方生成
 * （本函数 use_thumbnail=0，避免重复 pad 全局图）。
 *
 * 【1600×2264 对 V3 的完整算例】比例 0.7067 → 2:3 (差0.0400) 胜 3:4
 * (差0.0433) → resize 1280×1920 → 6 个 640² tile → 每个 tile SAM 100 tok
 * → CLIP 111 tok（含换行）→ 剥换行 100 tok → 6 块重排 30×20 网格
 * → 插换行 630 tok。总 token = 630+272+1 = 903（不含全局图路径）。
 *
 * 【成本上限】V3 允许 32 块，极端长图 1:32 → 1×32 网格 → 3793 token；
 * token 数随块数线性增长（V3 大图 43s 编码的来源）。V4 把上限收到 16：
 * 官方 vLLM 最多 16 crop，超出训练分布的 token 数会让模型把多视图
 * 误读成多页（<PAGE>×N）。
 * 【小图捷径】≤image_size 的图直接 pad 成 1 块，不走分块路径。 */
ds_image_t **ds_dynamic_preprocess(const ds_image_t *img, int image_size,
                                    int min_num, int max_num,
                                    int use_thumbnail, int *out_count) {
    if (!img || !img->pixels) { *out_count = 0; return NULL; }

    int orig_w = img->width, orig_h = img->height;

    /* Small images: no splitting needed, just pad to image_size
     * 小图捷径：两维都不超过 image_size → 不切块，只 pad 成单张
     * （与 Python 版提前 return 一致；结果数组只装1个指针） */
    if (orig_w <= image_size && orig_h <= image_size) {
        ds_image_t **result = (ds_image_t **)malloc(sizeof(ds_image_t *));
        if (!result) { *out_count = 0; return NULL; }
        result[0] = ds_image_pad(img, image_size, 127);
        if (!result[0]) { free(result); *out_count = 0; return NULL; }
        *out_count = 1;
        if (ds_verbose >= 1)
            fprintf(stderr, "dynamic_preprocess: small image %dx%d, 1 padded image\n", orig_w, orig_h);
        return result;
    }

    /* ① 原图纵横比：先转float再除——整数除法会把非整除结果截成0或1 */
    float aspect_ratio = (float)orig_w / (float)orig_h;

    /* ② 选最接近原图比例的 (rows, cols) 块组合 */
    int block_rows, block_cols;
    int blocks = find_closest_aspect_ratio(aspect_ratio, orig_w, orig_h, image_size,
                                            &block_rows, &block_cols, min_num, max_num);

    /* ③ 目标画布 = 每块 size × 块数，恰好整除无余数，切分时不会有碎边 */
    int target_w = image_size * block_cols;
    int target_h = image_size * block_rows;

    if (ds_verbose >= 1)
        fprintf(stderr, "dynamic_preprocess: %dx%d -> ratio %d:%d, resize to %dx%d, %d blocks\n",
                orig_w, orig_h, block_cols, block_rows, target_w, target_h, blocks);

    /* Resize full image to target_w × target_h */
    ds_image_t *resized = ds_image_resize(img, target_w, target_h);
    if (!resized) { *out_count = 0; return NULL; }

    /* Split into blocks and crop each
     * 结果指针数组：块数 + 可选的全局缩略图1张 */
    int n_patches = blocks + (use_thumbnail ? 1 : 0);
    ds_image_t **result = (ds_image_t **)malloc(n_patches * sizeof(ds_image_t *));
    if (!result) { ds_image_free(resized); *out_count = 0; return NULL; }

    /* ④ 行主序切分：先左到右、再上到下——顺序与 Python 一致，
     * 决定了后续 token 拼接时 tile 的排列 */
    int idx = 0;
    for (int r = 0; r < block_rows; r++) {
        for (int c = 0; c < block_cols; c++) {
            /* 每块在画布上的半开矩形 [left,right) × [top,bottom) */
            int left = c * image_size;
            int top = r * image_size;
            int right = left + image_size;
            int bottom = top + image_size;
            result[idx] = ds_image_crop_box(resized, left, top, right, bottom);
            if (!result[idx]) {
                /* 任一块失败：回滚已生成的块与中间整图，避免部分泄漏 */
                for (int k = 0; k < idx; k++) ds_image_free(result[k]);
                free(result); ds_image_free(resized);
                *out_count = 0; return NULL;
            }
            idx++;
        }
    }

    ds_image_free(resized);   /* 整图已切完，中间结果不再需要 */

    /* Append thumbnail (original resized to image_size × image_size)
     * 缩略图由【原图】pad 而来（保比例），与 tile 的等分切分是两条来源 */
    if (use_thumbnail) {
        ds_image_t *thumb = ds_image_pad(img, image_size, 127);
        if (!thumb) {
            for (int k = 0; k < idx; k++) ds_image_free(result[k]);
            free(result);
            *out_count = 0; return NULL;
        }
        result[idx++] = thumb;   /* 缩略图固定排在所有tile之后 */
    }

    *out_count = idx;
    if (ds_verbose >= 1)
        fprintf(stderr, "dynamic_preprocess: %d patches + %d thumbnail = %d images\n",
                blocks, use_thumbnail ? 1 : 0, idx);
    return result;
}

/* ds_image_free — 释放图像（按 owns_stb 标志选择匹配的释放器）
 * stb_image 内部用自己的分配器（可能带对齐/统计头），与 free() 混用是
 * 未定义行为——这正是 owns_stb 标志存在的意义：加载方式决定释放方式。
 *
 * @param img  待释放图像（可为NULL） */
void ds_image_free(ds_image_t *img) {
    if (!img) return;   /* NULL安全 */
    if (img->pixels) {
        if (img->owns_stb)
            stbi_image_free(img->pixels);   /* stbi_load 的内存必须用它自己的释放器 */
        else
            free(img->pixels);              /* malloc 分配（pad/crop/CG路径）走 free() */
    }
    free(img);   /* 最后释放结构体本身 */
}
