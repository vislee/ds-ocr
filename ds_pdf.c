/*
 * ds_pdf.c - PDF rendering stub for non-macOS platforms.
 * ds_pdf.c — 非 macOS 平台的 PDF 渲染桩（保持零外部依赖，返回失败）
 *
 * 【设计意图】PDF 输入的真实实现在 ds_pdf.m（macOS 专属，CoreGraphics/
 * PDFKit），使 main.c 的 -i file.pdf 在所有平台上都能编译通过——非
 * macOS 平台链接本桩，运行时返回 NULL/-1 并提示。这是"平台能力下沉到
 * 平台文件、公共 API 保持一致"的分层：调用方不需要 #ifdef。
 */

#include "ds_pdf.h"

#include <stdio.h>

ds_pdf_doc_t *ds_pdf_open(const char *path) {
    (void)path;
    fprintf(stderr, "PDF support is only available on macOS in this build\n");
    return NULL;
}

int ds_pdf_page_count(ds_pdf_doc_t *doc) {
    (void)doc;
    return 0;
}

int ds_pdf_render_page(ds_pdf_doc_t *doc, int page_index,
                       unsigned char **out_pixels, int *out_w, int *out_h) {
    (void)doc; (void)page_index; (void)out_pixels; (void)out_w; (void)out_h;
    return -1;
}

void ds_pdf_close(ds_pdf_doc_t *doc) {
    (void)doc;
}
