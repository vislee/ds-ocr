/*
 * ds_pdf.h - PDF page rendering for DeepSeek-OCR
 * ds_pdf.h — PDF 逐页渲染模块
 *
 * 【模块角色】
 * 将 PDF 文档逐页渲染为 RGB 像素数组，交给 ds_recognize_image() 识别。
 * 引擎本身只认像素，PDF 是输入端的"翻译层"——与 ds_image.c（图片解码）同级。
 *
 * 【设计决策】
 * - macOS: CoreGraphics CGPDFDocument（系统框架，零外部依赖）
 * - 非 macOS: 桩实现（返回失败），保持项目零依赖原则
 * - 渲染分辨率: 2.0×页面点阵（≈144 DPI），长边上限 2048px
 *   A4 页面 (612×792pt) → 约 1224×1584px，小字号文档仍保留足够细节
 * - 输出统一为 3 通道 RGB 行主序，与 ds_image_t 的布局约定一致
 */

#ifndef DS_PDF_H
#define DS_PDF_H

#include <stddef.h>

/* ds_pdf_doc_t — 打开的 PDF 文档句柄（不透明结构体） */
typedef struct ds_pdf_doc ds_pdf_doc_t;

/* ds_pdf_open — 打开 PDF 文件
 *
 * 返回: 文档句柄（调用者用 ds_pdf_close 释放），失败返回 NULL
 */
ds_pdf_doc_t *ds_pdf_open(const char *path);

/* ds_pdf_page_count — 返回 PDF 总页数（doc 无效时返回 0） */
int ds_pdf_page_count(ds_pdf_doc_t *doc);

/* ds_pdf_render_page — 渲染指定页为 RGB 像素
 *
 * page_index: 0 起始页码
 * out_pixels: 输出 malloc 分配的 RGB 缓冲区（3字节/像素，行主序，白底不透明）
 *             调用者负责 free(*out_pixels)
 * out_w/out_h: 输出像素宽高
 *
 * 返回: 0 成功，-1 失败
 */
int ds_pdf_render_page(ds_pdf_doc_t *doc, int page_index,
                       unsigned char **out_pixels, int *out_w, int *out_h);

/* ds_pdf_close — 关闭文档并释放资源（NULL 安全） */
void ds_pdf_close(ds_pdf_doc_t *doc);

#endif /* DS_PDF_H */
