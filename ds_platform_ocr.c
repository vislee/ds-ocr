/*
 * ds_platform_ocr.c - Non-Apple platform OCR stub.
 * ds_platform_ocr.c — 非 Apple 平台的"平台 OCR"桩实现
 *
 * 【模块角色】"平台 OCR"指用操作系统自带的 OCR 引擎代替本项目的
 * DeepSeek 模型：macOS 上由 ds_platform_ocr.m 用 Apple Vision 框架
 * 实现（编译开关 USE_APPLE_VISION 打开时才编入），main.c 的
 * --vision / --vision-fast 旗标走它——不加载任何模型权重、
 * 直接对图片出文字的快速模式。
 *
 * 【本文件的存在意义】非 Apple 平台没有 Vision 框架，但仍要保证
 * main.c 不加 #ifdef 就能编译链接：这里提供签名完全一致的桩函数，
 * 运行时返回 NULL 表示"平台不支持"。与 ds_pdf.c 的桩是同一个分层
 * 思路——平台能力下沉到平台文件（.c/.m 各一份），公共 API 保持一致。
 */

#include "ds_platform_ocr.h"

/* ds_platform_ocr_file — 识别单张图片（桩：恒不支持）。
 * @param image_path 图片文件路径（桩不使用，(void) 消除未用参数警告）
 * @param accurate   1=高精度模式（Vision 的 .accurate 请求级），0=快速模式
 * @param verbosity  日志级别（真实实现用它控制 stderr 输出量）
 * @return 识别文本（malloc 分配的 UTF-8 串，调用者负责 free）；
 *         桩恒返回 0（即 NULL）表示平台不支持。
 * 真实实现见 ds_platform_ocr.m 的同名函数。 */
char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    (void)image_path;   /* 三个参数只为保持签名一致 */
    (void)accurate;
    (void)verbosity;
    return 0;           /* NULL：调用方（main.c）据此提示并退出 */
}
