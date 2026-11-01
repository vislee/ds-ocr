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

int ds_pdf_page_count(ds_pdf_doc_t *doc) {
    return doc ? doc->n_pages : 0;
}

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

void ds_pdf_close(ds_pdf_doc_t *doc) {
    if (!doc) return;
    CGPDFDocumentRelease(doc->doc);
    free(doc);
}

#endif /* __APPLE__ */
