/*
 * ds_pdf.m - PDF page rendering via macOS CoreGraphics.
 * ds_pdf.m — macOS CoreGraphics PDF 逐页渲染实现
 *
 * CGPDFDocument 加载文档，CGBitmapContext 渲染页面位图。
 * CoreGraphics 位图不支持 24bpp RGB，因此渲染为 RGBA 再抽取出 RGB。
 */

#include "ds_pdf.h"

#if defined(__APPLE__)

#import <CoreGraphics/CoreGraphics.h>
#import <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>

/* 渲染缩放: 2.0×页面点阵 ≈ 144 DPI；长边超过 2048px 时按比例压回 */
#define DS_PDF_SCALE        2.0
#define DS_PDF_MAX_LONG_SIDE 2048

struct ds_pdf_doc {
    CGPDFDocumentRef doc;
    int n_pages;
};

/* ds_pdf_open - 打开 PDF 文档（CoreGraphics 实现侧）
 *
 * 【CF 桥接链】路径 C 字符串 → CFString（文件系统表示）→ CFURL → CGPDFDocument。
 *   CFStringCreateWithFileSystemRepresentation 而非 NSString，是为了在纯 C 风格
 *   路径上避免引入 Foundation 依赖——本文件只依赖 CoreGraphics/CoreFoundation。
 * 【所有权】每个中间对象用完立即 CFRelease；doc 的引用计数移交给 ds_pdf_doc_t，
 *   由 ds_pdf_close 统一释放。
 * @return malloc 的文档句柄（调用者 ds_pdf_close），页数 ≤0 或解析失败返回 NULL */
ds_pdf_doc_t *ds_pdf_open(const char *path) {
    if (!path) return NULL;

    CFStringRef str = CFStringCreateWithFileSystemRepresentation(kCFAllocatorDefault, path);
    if (!str) return NULL;
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, str, kCFURLPOSIXPathStyle, false);
    CFRelease(str);
    if (!url) return NULL;

    CGPDFDocumentRef doc = CGPDFDocumentCreateWithURL(url);
    CFRelease(url);
    if (!doc) return NULL;

    int n_pages = (int)CGPDFDocumentGetNumberOfPages(doc);
    if (n_pages <= 0) {
        CGPDFDocumentRelease(doc);
        return NULL;
    }

    ds_pdf_doc_t *d = (ds_pdf_doc_t *)malloc(sizeof(*d));
    if (!d) {
        CGPDFDocumentRelease(doc);
        return NULL;
    }
    d->doc = doc;
    d->n_pages = n_pages;
    return d;
}

/* ds_pdf_page_count - 返回总页数（NULL 安全，主调用方 main.c 用它做循环上界） */
int ds_pdf_page_count(ds_pdf_doc_t *doc) {
    return doc ? doc->n_pages : 0;
}

/* ds_pdf_render_page - 渲染一页为 RGB 像素（实现侧）
 *
 * 【管线】MediaBox → 计算缩放（2.0×≈144 DPI，长边 ≤2048px 再压回）→
 *   RGBA 位图上下文（白底）→ DrawPDFPage → 逐行抽包 RGBA→RGB。
 * 【为什么渲染成 RGBA 再抽 RGB】CoreGraphics 位图上下文不支持 24bpp RGB，
 *   只能带 alpha 通道渲染；而引擎侧 ds_image_t 约定 3 字节/像素，故最后做
 *   内存级抽取（每像素跳过第 4 字节）。stride 用 GetBytesPerRow 取——上下文
 *   可能做行对齐，不能假设等于 w*4。
 * 【预乘 alpha 说明】kCGImageAlphaPremultipliedLast 表示 RGB 已被 alpha 预乘；
 *   因先填白底（alpha=1 处不透明），整页像素 alpha 恒为 255，预乘无失真。
 * 【CTM 顺序】先 Scale 后 Translate（矩阵右乘依次作用），先把页面点阵缩放
 *   到设备像素，再平移掉 MediaBox 原点——见下方 DrawPDFPage 处的坐标注释。
 * @return 0 成功（*out_pixels 归调用者 free），-1 失败 */
int ds_pdf_render_page(ds_pdf_doc_t *d, int page_index,
                       unsigned char **out_pixels, int *out_w, int *out_h) {
    if (!d || page_index < 0 || page_index >= d->n_pages ||
        !out_pixels || !out_w || !out_h)
        return -1;

    /* CGPDF 页码 1 起始 */
    CGPDFPageRef page = CGPDFDocumentGetPage(d->doc, page_index + 1);
    if (!page) return -1;

    CGRect box = CGPDFPageGetBoxRect(page, kCGPDFMediaBox);
    if (box.size.width <= 0.0 || box.size.height <= 0.0) return -1;

    CGFloat scale = DS_PDF_SCALE;
    CGFloat long_side = (box.size.width > box.size.height ? box.size.width : box.size.height) * scale;
    if (long_side > DS_PDF_MAX_LONG_SIDE) scale *= DS_PDF_MAX_LONG_SIDE / long_side;

    size_t w = (size_t)(box.size.width * scale + 0.5);
    size_t h = (size_t)(box.size.height * scale + 0.5);
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (!cs) return -1;
    /* RGBA 大端 = 内存字节序 R,G,B,A */
    CGContextRef bmp = CGBitmapContextCreate(NULL, w, h, 8, w * 4, cs,
                                             kCGImageAlphaPremultipliedLast |
                                             kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!bmp) return -1;

    /* 白底: PDF 页面可能带透明通道 */
    CGContextSetRGBFillColor(bmp, 1.0, 1.0, 1.0, 1.0);
    CGContextFillRect(bmp, CGRectMake(0.0, 0.0, (CGFloat)w, (CGFloat)h));

    /* CGContextDrawPDFPage 自带"PDF坐标(左下原点)→位图坐标(左上原点)"的翻转，
     * 只需叠加缩放与 MediaBox 原点平移，不要手工翻转（实测会得到上下颠倒的页面）。
     * /Rotate 旋转属性由 DrawPDFPage 内部处理。 */
    CGContextScaleCTM(bmp, scale, scale);
    CGContextTranslateCTM(bmp, -box.origin.x, -box.origin.y);
    CGContextDrawPDFPage(bmp, page);

    const unsigned char *rgba = (const unsigned char *)CGBitmapContextGetData(bmp);
    size_t stride = CGBitmapContextGetBytesPerRow(bmp);
    unsigned char *rgb = (unsigned char *)malloc(w * h * 3);
    if (!rgb) {
        CGContextRelease(bmp);
        return -1;
    }
    for (size_t y = 0; y < h; y++) {
        const unsigned char *src = rgba + y * stride;
        unsigned char *dst = rgb + y * w * 3;
        for (size_t x = 0; x < w; x++) {
            dst[x * 3 + 0] = src[x * 4 + 0];
            dst[x * 3 + 1] = src[x * 4 + 1];
            dst[x * 3 + 2] = src[x * 4 + 2];
        }
    }

    CGContextRelease(bmp);
    *out_pixels = rgb;
    *out_w = (int)w;
    *out_h = (int)h;
    return 0;
}

/* ds_pdf_close - 释放 CGPDFDocument 与句柄本身（NULL 安全）
 * 调用时机见 main.c：渲染完所有页后再关文档，避免 CGPDFDocument 在
 * CGContextDrawPDFPage 期间被释放。 */
void ds_pdf_close(ds_pdf_doc_t *doc) {
    if (!doc) return;
    CGPDFDocumentRelease(doc->doc);
    free(doc);
}

#endif /* __APPLE__ */
