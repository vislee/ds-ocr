/*
 * ds_platform_ocr.m - macOS Vision OCR backend.
 * ds_platform_ocr.m — macOS Vision 框架 OCR 后端实现
 *
 * 【模块角色】与主神经网络流水线并行的"第二条 OCR 通路"：
 *   - 主通路：图片 → SAM/DeepEncoder → MoE 解码器 → token 序列（本项目自研引擎）
 *   - 本通路：图片 → Apple Vision 系统框架 → 直接返回 UTF-8 文本
 *   由 main.c 的 --vision / --vision-fast 旗标选择，用于与引擎输出对照评测。
 *   非 macOS 或未定义 USE_APPLE_VISION 时，链接 ds_platform_ocr.c 的桩（恒 NULL）。
 *
 * 【数据流】文件路径 → CGImageSource 解码 → 缩放至 ≤1280px →
 *   VNRecognizeTextRequest 识别 → 按阅读顺序（先上后下、同行先左后右）拼接行 → C 字符串。
 *
 * 【坐标系要点】Vision 的 boundingBox 是归一化 [0,1] 坐标，原点在【左下角】——
 *   与 CoreGraphics/屏幕坐标（左上原点）相反，所以"行顶"= origin.y + height，
 *   排序时 top 越大越靠前。见 docs/09-量化与部署篇.md 平台相关章节。
 */

#include "ds_platform_ocr.h"
#include <stddef.h>

#if defined(__APPLE__) && defined(USE_APPLE_VISION)

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* dup_nsstring_utf8 - NSString → malloc 的 UTF-8 C 字符串（所有权移交 C 世界）
 *
 * 【为什么要拷贝】-[NSString UTF8String] 返回的指针由 NSString 对象（或其
 *   autorelease 池）拥有，离开 @autoreleasepool 后即失效；而本模块的 API 契约
 *   要求返回"调用者 free"的 C 堆内存（见 ds_platform_ocr.h），必须做一次深拷贝。
 *
 * @param text 源字符串（可为 nil → 返回 NULL）
 * @return malloc(len+1) 的 NUL 结尾 UTF-8 缓冲区；分配失败返回 NULL */
static char *dup_nsstring_utf8(NSString *text) {
    const char *utf8 = [text UTF8String];
    if (!utf8) return NULL;
    size_t len = strlen(utf8);
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, utf8, len + 1);
    return out;
}

/* observation_top - 取文本框"上边缘"的归一化 y 值
 *
 * Vision 坐标系原点在左下角、y 向上（与 CG 的左上原点相反），所以一个框的
 * "图面上边缘"= origin.y + height——值越大代表在图里越靠上。
 * 这是阅读顺序排序的主键。 */
static CGFloat observation_top(VNRecognizedTextObservation *obs) {
    return obs.boundingBox.origin.y + obs.boundingBox.size.height;
}

/* create_vision_image - 把任意 CGImage 规整成 Vision 友好的 RGB 位图
 *
 * 【为什么限制 1280px】Apple Vision 在长边 ≈1280 附近的速度/精度拐点最佳；
 *   更大的输入只增加识别耗时，小字精度收益递减。只缩不放（scale ≤ 1），
 *   避免放大引入插值伪影干扰 OCR。
 * 【像素格式】kCGImageAlphaNoneSkipLast：每像素 4 字节但忽略 alpha（RGBX），
 *   kCGBitmapByteOrder32Host = 按主机字节序解释 32 位字——x86/ARM64 小端下
 *   内存布局即 R,G,B,填充，与 Vision 期望一致。不支持 24bpp 直接渲染，
 *   与 ds_pdf.m 中的 RGBA 中转是同一约束（见该文件头注释）。
 * 【返回】新 CGImage（调用者 CGImageRelease）；失败返回 NULL。 */
static CGImageRef create_vision_image(CGImageRef src) {
    size_t src_w = CGImageGetWidth(src);
    size_t src_h = CGImageGetHeight(src);
    if (src_w == 0 || src_h == 0) return NULL;

    const CGFloat max_dim = 1280.0;
    CGFloat scale = MIN(1.0, max_dim / (CGFloat)MAX(src_w, src_h));
    size_t dst_w = (size_t)lrint((CGFloat)src_w * scale);
    size_t dst_h = (size_t)lrint((CGFloat)src_h * scale);
    if (dst_w < 1) dst_w = 1;
    if (dst_h < 1) dst_h = 1;

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    if (!color_space) return NULL;
    /* bytesPerRow 传 0：让 CoreGraphics 自己选择对齐步长，取数据前用 GetBytesPerRow 询问 */
    CGContextRef ctx = CGBitmapContextCreate(NULL, dst_w, dst_h, 8, 0,
                                             color_space,
                                             kCGImageAlphaNoneSkipLast |
                                             kCGBitmapByteOrder32Host);
    CGColorSpaceRelease(color_space);
    if (!ctx) return NULL;

    /* 高质量插值缩放：OCR 对字符边缘锐度敏感，kCGInterpolationHigh 保留最多细节 */
    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextDrawImage(ctx, CGRectMake(0, 0, dst_w, dst_h), src);
    CGImageRef out = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    return out;
}

/* sorted_observations - 把 Vision 识别结果整理成人类阅读顺序
 *
 * 【排序规则】两阶段比较：
 *   1. 行分组：两框上边缘相差 ≤ 0.015（归一化坐标，≈图高的 1.5%）视为同一行，
 *      否则 top 大（图面偏上）者排前 → 实现先上后下；
 *   2. 同行内按 origin.x 升序（x_delta = left.x − right.x，>0 说明 left 更靠右，
 *      返回 Descending 让 left 排后）→ 实现先左后右。
 * 【为什么不用 Vision 默认顺序】VNRecognizeTextRequest 结果顺序不保证阅读序，
 *   直接拼接会打乱段落，必须按几何位置重排。
 *
 * @param observations VNRecognizeTextRequest.results（可为 nil）
 * @return 排序后的不可变数组（autorelease，调用者不持有） */
static NSArray<VNRecognizedTextObservation *> *sorted_observations(NSArray *observations) {
    return [observations sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        VNRecognizedTextObservation *left = (VNRecognizedTextObservation *)a;
        VNRecognizedTextObservation *right = (VNRecognizedTextObservation *)b;
        CGFloat left_top = observation_top(left);
        CGFloat right_top = observation_top(right);
        CGFloat top_delta = right_top - left_top;
        if (fabs(top_delta) > 0.015) {
            return top_delta > 0 ? NSOrderedDescending : NSOrderedAscending;
        }
        CGFloat x_delta = left.boundingBox.origin.x - right.boundingBox.origin.x;
        if (fabs(x_delta) > 0.001) {
            return x_delta > 0 ? NSOrderedDescending : NSOrderedAscending;
        }
        return NSOrderedSame;
    }];
}

/* ds_platform_ocr_file - Apple Vision 识别一张图片（实现侧入口）
 *
 * 【流程】解码 → 规整(≤1280px RGB) → VNRecognizeTextRequest 同步识别 →
 *   行排序 → 拼接为"\n"分隔的 UTF-8 文本。API 契约（参数/返回/所有权）见
 *   ds_platform_ocr.h；此处只讲实现取舍。
 * 【与主引擎的差异】本函数直接产出文本，跳过 token 化与后处理（无温度/无
 *   n-gram 阻断），因此可当"参照答案"用：README 的 --vision 对照评测即走此路径。
 * @param accurate  1=VNRequestTextRecognitionLevelAccurate（慢而准）
 *                  0=Fast（快而糙）——对应 --vision / --vision-fast
 * @param verbosity >0 时向 stderr 打印耗时与行数（正文走 stdout 由 main.c 控制）
 * @return malloc 文本（调用者 free）；任一环节失败返回 NULL */
char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    if (!image_path) return NULL;

    /* @autoreleasepool：下面所有 Foundation 临时对象（NSURL/CGImageSource 包装/
     * request.results 等）在块尾统一释放。本函数会被 main.c 的 PDF 循环逐页调用，
     * 不套池会导致内存随页数线性增长。 */
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:image_path];
        if (!path) return NULL;

        NSURL *url = [NSURL fileURLWithPath:path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot open image\n");
            return NULL;
        }
        /* 只取第 0 张图像：PNG/JPEG 的 EXIF 缩略图等次级表示全部忽略 */
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot decode image\n");
            return NULL;
        }

        CGImageRef vision_image = create_vision_image(image);
        CGImageRelease(image);
        if (!vision_image) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot prepare image\n");
            return NULL;
        }

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:vision_image options:@{}];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        /* Fast/Accurate 是 Vision 内部两套模型：Fast 走轻量网络毫秒级返回，
         * Accurate 走更大网络 + 语言模型重排序，耗时可达数倍 */
        request.recognitionLevel = accurate ? VNRequestTextRecognitionLevelAccurate
                                            : VNRequestTextRecognitionLevelFast;
        /* 打开系统词典纠错（如把 "teh" 修成 "the"）——OCR 场景对印刷体是净收益 */
        request.usesLanguageCorrection = YES;
        /* 识别语言按优先级排列：中文简体在前，英文兜底。macOS 11 才开放该属性 */
        if (@available(macOS 11.0, *)) {
            request.recognitionLanguages = @[@"zh-Hans", @"en-US"];
        }

        NSError *error = nil;
        double start = CFAbsoluteTimeGetCurrent();
        BOOL ok = [handler performRequests:@[request] error:&error];
        double elapsed_ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0;
        if (!ok) {
            if (verbosity > 0) {
                fprintf(stderr, "Apple Vision OCR failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
            CGImageRelease(vision_image);
            return NULL;
        }
        CGImageRelease(vision_image);

        NSMutableString *text = [NSMutableString string];
        NSArray *sorted = sorted_observations(request.results ?: @[]);
        NSCharacterSet *trim_set = [NSCharacterSet whitespaceAndNewlineCharacterSet];
        for (VNRecognizedTextObservation *obs in sorted) {
            /* topCandidates:1 只取该框置信度最高的候选文本行 */
            VNRecognizedText *candidate = [[obs topCandidates:1] firstObject];
            if (!candidate || candidate.string.length == 0) continue;
            NSString *line = [candidate.string stringByTrimmingCharactersInSet:trim_set];
            if (line.length == 0) continue;
            if (text.length > 0) [text appendString:@"\n"];
            [text appendString:line];
        }

        if (verbosity > 0) {
            fprintf(stderr, "Apple Vision OCR: %.0f ms, %lu lines\n",
                    elapsed_ms, (unsigned long)sorted.count);
        }
        return dup_nsstring_utf8(text);
    }
}

#else

/* ds_platform_ocr_file - 非 macOS / 未启用 USE_APPLE_VISION 的桩实现
 * 恒返回 NULL，调用方（main.c）据此回退到自研引擎流水线。
 * 与 macOS 真实实现（上方）同签名，链接期二选一，调用方无平台分支。 */
char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    (void)image_path;
    (void)accurate;
    (void)verbosity;
    return NULL;
}

#endif
