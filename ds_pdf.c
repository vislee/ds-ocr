/*
 * ds_pdf.c - PDF rendering stub for non-macOS platforms.
 * ds_pdf.c — 非 macOS 平台的 PDF 渲染桩（保持零外部依赖，返回失败）
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
