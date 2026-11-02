/*
 * ds_platform_ocr.h - Optional platform OCR backends.
 * ds_platform_ocr.h — 平台 OCR 后端的公共接口（一份签名，两份实现）
 *
 * 实现选择由构建系统决定，调用方（main.c）完全不感知：
 *   ds_platform_ocr.m — macOS 且定义 USE_APPLE_VISION 时编入，
 *                       用 Apple Vision 框架真正识别图片；
 *   ds_platform_ocr.c — 其他平台编入的桩，运行时恒返回 NULL。
 * 入口: main.c 的 --vision（高精度）/ --vision-fast（快速）旗标。
 */

#ifndef DS_PLATFORM_OCR_H
#define DS_PLATFORM_OCR_H

/* ds_platform_ocr_file — 对一张图片做平台 OCR。
 * @param image_path 图片文件路径
 * @param accurate   1=高精度识别，0=快速识别
 * @param verbosity  日志级别（越大输出越多，0=安静）
 * @return 识别出的 UTF-8 文本（malloc 分配，调用者 free）；
 *         平台不支持或识别失败返回 NULL。 */
char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity);

#endif
