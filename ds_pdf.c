/*
 * ds_pdf.c - PDF rendering stub for non-macOS platforms.
 * ds_pdf.c — 非 macOS 平台的 PDF 渲染桩（保持零外部依赖，返回失败）
 *
 * 【设计意图】PDF 输入的真实实现在 ds_pdf.m（macOS 专属，CoreGraphics/
 * PDFKit），使 main.c 的 -i file.pdf 在所有平台上都能编译通过——非
 * macOS 平台链接本桩，运行时返回 NULL/-1 并提示。这是"平台能力下沉到
 * 平台文件、公共 API 保持一致"的分层：调用方不需要 #ifdef。
 * 【调用流程】main.c 的典型用法：ds_pdf_open → ds_pdf_page_count →
 * 循环 { ds_pdf_render_page → ds_recognize_image } → ds_pdf_close。
 * 本桩的四个函数与 ds_pdf.m 一一同名同签名，行为差异只在返回值。
 */

#include "ds_pdf.h"

#include <stdio.h>

/* ds_pdf_open — 打开 PDF（桩：恒失败）。
 * (void)path 消除"参数未使用"警告——桩必须保持与真实实现完全相同的
 * 签名。行为: stderr 提示当前构建不含 PDF 支持，返回 NULL 表示打开
 * 失败。main.c 的 -i file.pdf 路径据此向用户报告"此平台不支持"。 */
ds_pdf_doc_t *ds_pdf_open(const char *path) {
    (void)path;  /* 桩不消费参数，仅占签名 */
    fprintf(stderr, "PDF support is only available on macOS in this build\n");
    return NULL; /* 失败句柄：调用方拿到 NULL 即停止，不会继续调下面的接口 */
}

/* ds_pdf_page_count — 返回总页数（桩：无文档可数，恒 0）。
 * 与头文件约定一致：doc 无效时返回 0，调用方以此判断"没有可处理页"。 */
int ds_pdf_page_count(ds_pdf_doc_t *doc) {
    (void)doc;
    return 0;
}

/* ds_pdf_render_page — 渲染一页为 RGB 像素（桩：恒失败）。
 * 真实实现（ds_pdf.m）会 malloc 像素缓冲并通过 out_* 四个出参交还；
 * 桩只把所有出参 (void) 掉并返回 -1——调用方检查返回值后即跳过该页。 */
int ds_pdf_render_page(ds_pdf_doc_t *doc, int page_index,
                       unsigned char **out_pixels, int *out_w, int *out_h) {
    (void)doc; (void)page_index; (void)out_pixels; (void)out_w; (void)out_h;
    return -1; /* 失败：非 macOS 平台无 CoreGraphics 可渲染 */
}

/* ds_pdf_close — 关闭文档（桩：无事可做，NULL 安全）。
 * 桩也保持与真实实现相同的释放约定，调用方代码无需任何平台分支。 */
void ds_pdf_close(ds_pdf_doc_t *doc) {
    (void)doc;
}
