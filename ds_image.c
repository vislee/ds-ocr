/*
 * ds_image.c - Image loading for DeepSeek-OCR
 * Uses stb_image.h for decoding (single-header library).
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

ds_image_t *ds_image_load(const char *path) {
#if defined(__APPLE__) && defined(USE_APPLE_VISION)
    /* Use CoreGraphics for image loading — matches PIL's JPEG/PNG decoder */
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, strlen(path), FALSE);
    if (!url) goto fallback_stb;
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
    CFRelease(url);
    if (!source) goto fallback_stb;
    CGImageRef cg_img = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!cg_img) goto fallback_stb;

    int w = (int)CGImageGetWidth(cg_img);
    int h = (int)CGImageGetHeight(cg_img);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (!cs) { CGImageRelease(cg_img); goto fallback_stb; }

    /* Create RGBA bitmap */
    CGContextRef ctx = CGBitmapContextCreate(NULL, w, h, 8, w * 4, cs,
        kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) { CGImageRelease(cg_img); goto fallback_stb; }

    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg_img);
    unsigned char *rgba = (unsigned char *)CGBitmapContextGetData(ctx);

    /* Convert RGBA → RGB */
    unsigned char *rgb = (unsigned char *)malloc(w * h * 3);
    if (!rgb) { CGContextRelease(ctx); CGImageRelease(cg_img); return NULL; }
    for (int i = 0; i < w * h; i++) {
        rgb[i*3+0] = rgba[i*4+0];
        rgb[i*3+1] = rgba[i*4+1];
        rgb[i*3+2] = rgba[i*4+2];
    }
    CGContextRelease(ctx);
    CGImageRelease(cg_img);

    ds_image_t *img = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!img) { free(rgb); return NULL; }
    img->pixels = rgb;
    img->width = w;
    img->height = h;
    img->channels = 3;
    img->owns_stb = 0;
    if (ds_verbose >= 2)
        fprintf(stderr, "Loaded image (CG): %s (%dx%d)\n", path, w, h);
    return img;

fallback_stb:
#endif
    { int w, h, channels;
    unsigned char *data = stbi_load(path, &w, &h, &channels, 3); /* Force 3 channels (RGB) */
    if (!data) {
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
    float ax = fabsf(x);
    if (ax <= 1.0f) {
        /* ((a+2)*|x| - (a+3))*|x|^2 + 1 */
        return ((-0.5f + 2.0f) * ax - (-0.5f + 3.0f)) * ax * ax + 1.0f;
    } else if (ax <= 2.0f) {
        /* (a*|x| - 5a)*|x|^2 + 8a*|x| - 4a */
        return ((-0.5f * ax - 5.0f * (-0.5f)) * ax * ax
                + 8.0f * (-0.5f) * ax - 4.0f * (-0.5f));
    }
    return 0.0f;
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
    /* Compute filter support width (in source pixels) */
    float x_support = 2.0f / x_scale_factor;  /* wider for downsampling */
    float y_support = 2.0f / y_scale_factor;

    int ix_min = (int)floorf(fx - x_support);
    int ix_max = (int)ceilf(fx + x_support);
    int iy_min = (int)floorf(fy - y_support);
    int iy_max = (int)ceilf(fy + y_support);

    /* Clamp to image bounds */
    if (ix_min < 0) ix_min = 0;
    if (ix_max >= src_w) ix_max = src_w - 1;
    if (iy_min < 0) iy_min = 0;
    if (iy_max >= src_h) iy_max = src_h - 1;

    float result = 0.0f;
    float w_sum = 0.0f;

    for (int iy = iy_min; iy <= iy_max; iy++) {
        float wy = ds_cubic_weight(((float)iy - fy) * y_scale_factor);
        if (wy == 0.0f) continue;

        for (int ix = ix_min; ix <= ix_max; ix++) {
            float wx = ds_cubic_weight(((float)ix - fx) * x_scale_factor);
            if (wx == 0.0f) continue;

            float w = wx * wy;
            result += w * (float)src[(iy * src_w + ix) * 3 + c];
            w_sum += w;
        }
    }
    /* Normalize */
    if (w_sum > 0.0f) result /= w_sum;
    return result;
}

ds_image_t *ds_image_resize(const ds_image_t *img, int target_width, int target_height) {
    if (!img || !img->pixels) return NULL;

#if defined(__APPLE__) && defined(USE_APPLE_VISION)
    /* Use CoreGraphics for high-quality resize matching PIL bicubic. */
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (cs) {
        /* Create source image from raw RGB pixels */
        unsigned char *rgba_src = (unsigned char *)malloc(img->width * img->height * 4);
        if (rgba_src) {
            for (int i = 0; i < img->width * img->height; i++) {
                rgba_src[i*4+0] = img->pixels[i*3+0];
                rgba_src[i*4+1] = img->pixels[i*3+1];
                rgba_src[i*4+2] = img->pixels[i*3+2];
                rgba_src[i*4+3] = 255;
            }
            CGContextRef src_ctx = CGBitmapContextCreate(NULL, img->width, img->height, 8,
                img->width * 4, cs, kCGImageAlphaNoneSkipLast);
            if (src_ctx) {
                memcpy(CGBitmapContextGetData(src_ctx), rgba_src, img->width * img->height * 4);
                CGImageRef src_img = CGBitmapContextCreateImage(src_ctx);
                CGContextRelease(src_ctx);
                if (src_img) {
                    CGContextRef dst_ctx = CGBitmapContextCreate(NULL, target_width, target_height, 8,
                        target_width * 4, cs, kCGImageAlphaNoneSkipLast);
                    if (dst_ctx) {
                        CGContextSetInterpolationQuality(dst_ctx, kCGInterpolationHigh);
                        CGContextDrawImage(dst_ctx, CGRectMake(0, 0, target_width, target_height), src_img);
                        CGImageRef dst_img = CGBitmapContextCreateImage(dst_ctx);
                        CGContextRelease(dst_ctx);
                        CGImageRelease(src_img);
                        if (dst_img) {
                            CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(dst_img));
                            const unsigned char *ptr = CFDataGetBytePtr(data);
                            size_t bpr = CGImageGetBytesPerRow(dst_img);
                            unsigned char *out = (unsigned char *)malloc(target_width * target_height * 3);
                            if (out) {
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
                                return r;
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
    /* Fallback: custom bicubic (less accurate vs PIL) */
    unsigned char *out = (unsigned char *)malloc(target_width * target_height * 3);
    if (!out) return NULL;

    float x_scale = (float)img->width / target_width;
    float y_scale = (float)img->height / target_height;
    float x_aa_factor = (float)target_width / (float)img->width;
    float y_aa_factor = (float)target_height / (float)img->height;
    if (x_aa_factor > 1.0f) x_aa_factor = 1.0f;
    if (y_aa_factor > 1.0f) y_aa_factor = 1.0f;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float fx = (x + 0.5f) * x_scale - 0.5f;
            float fy = (y + 0.5f) * y_scale - 0.5f;
            for (int c = 0; c < 3; c++) {
                float val = ds_bicuberp_aa(img->pixels, img->width, img->height, fx, fy, c,
                                           x_aa_factor, y_aa_factor);
                if (val < 0.0f) val = 0.0f;
                if (val > 255.0f) val = 255.0f;
                out[(y * target_width + x) * 3 + c] = (unsigned char)(val + 0.5f);
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
ds_image_t *ds_image_pad(const ds_image_t *img, int target_size, unsigned char pad_color) {
    if (!img || !img->pixels) return NULL;

    /* Calculate scale to fit within target_size × target_size */
    float scale = (float)target_size / (img->width > img->height ? img->width : img->height);
    int new_w = (int)(img->width * scale + 0.5f);
    int new_h = (int)(img->height * scale + 0.5f);
    if (new_w > target_size) new_w = target_size;
    if (new_h > target_size) new_h = target_size;

    /* First resize to fit */
    ds_image_t *resized = ds_image_resize(img, new_w, new_h);
    if (!resized) return NULL;

    /* Allocate target and fill with pad color */
    unsigned char *out = (unsigned char *)malloc(target_size * target_size * 3);
    if (!out) { ds_image_free(resized); return NULL; }
    memset(out, pad_color, target_size * target_size * 3);

    /* Center the resized image */
    int offset_x = (target_size - new_w) / 2;
    int offset_y = (target_size - new_h) / 2;
    for (int y = 0; y < new_h; y++) {
        memcpy(out + ((offset_y + y) * target_size + offset_x) * 3,
               resized->pixels + y * new_w * 3,
               new_w * 3);
    }
    ds_image_free(resized);

    ds_image_t *padded = (ds_image_t *)calloc(1, sizeof(ds_image_t));
    if (!padded) { free(out); return NULL; }
    padded->pixels = out;
    padded->width = target_size;
    padded->height = target_size;
    padded->channels = 3;
    padded->owns_stb = 0;  /* pixels from malloc, use free() */
    return padded;
}

float *ds_image_to_float_chw(const ds_image_t *img) {
    if (!img || !img->pixels) return NULL;

    int n = img->width * img->height;
    float *out = (float *)malloc(3 * n * sizeof(float));
    if (!out) return NULL;

    /* Convert HWC (interleaved RGB) to CHW (planar), normalize to [-1, 1]
     * Using mean=0.5, std=0.5: (pixel/255 - 0.5) / 0.5 = pixel/255*2 - 1
     * This matches Python's BasicImageTransform(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) */
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int hwc_idx = (y * img->width + x) * 3;
            int chw_idx = y * img->width + x;
            for (int c = 0; c < 3; c++) {
                out[c * n + chw_idx] = (float)img->pixels[hwc_idx + c] / 255.0f * 2.0f - 1.0f;
            }
        }
    }

    return out;
}

ds_image_t *ds_image_crop_box(const ds_image_t *img, int left, int top, int right, int bottom) {
    if (!img || !img->pixels) return NULL;

    int crop_w = right - left;
    int crop_h = bottom - top;
    if (crop_w <= 0 || crop_h <= 0) return NULL;

    unsigned char *out = (unsigned char *)malloc(crop_w * crop_h * 3);
    if (!out) return NULL;

    for (int y = 0; y < crop_h; y++) {
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
    cropped->owns_stb = 0;
    return cropped;
}

/* Find closest aspect ratio — matches Python find_closest_aspect_ratio() */
static int find_closest_aspect_ratio(float aspect_ratio, int width, int height,
                                       int image_size, int *out_rows, int *out_cols,
                                       int min_num, int max_num) {
    /* Generate target ratios: (i, j) where min_num <= i*j <= max_num
     * Python: target_ratios = set((i, j) ...) where i=width_blocks, j=height_blocks.
     * ratio[0]=i is width direction, ratio[1]=j is height direction.
     * V2: min_num=2, max_num=6 → at most 6 crops
     * V3: min_num=2, max_num=32 → up to 32 crops for large images */
    int best_width_blocks = 1, best_height_blocks = 1;
    float best_diff = 1e30f;
    long long area = (long long)width * height;

    for (int n = min_num; n <= max_num; n++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i * j < min_num || i * j > max_num) continue;
                float target_ar = (float)i / (float)j;  /* i=width, j=height */
                float diff = fabsf(aspect_ratio - target_ar);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_width_blocks = i;   /* width direction */
                    best_height_blocks = j;  /* height direction */
                } else if (diff == best_diff) {
                    /* Tie-break: prefer larger ratio if image area is large */
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

ds_image_t **ds_dynamic_preprocess(const ds_image_t *img, int image_size,
                                    int min_num, int max_num,
                                    int use_thumbnail, int *out_count) {
    if (!img || !img->pixels) { *out_count = 0; return NULL; }

    int orig_w = img->width, orig_h = img->height;

    /* Small images: no splitting needed, just pad to image_size */
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

    float aspect_ratio = (float)orig_w / (float)orig_h;

    int block_rows, block_cols;
    int blocks = find_closest_aspect_ratio(aspect_ratio, orig_w, orig_h, image_size,
                                            &block_rows, &block_cols, min_num, max_num);

    int target_w = image_size * block_cols;
    int target_h = image_size * block_rows;

    if (ds_verbose >= 1)
        fprintf(stderr, "dynamic_preprocess: %dx%d -> ratio %d:%d, resize to %dx%d, %d blocks\n",
                orig_w, orig_h, block_cols, block_rows, target_w, target_h, blocks);

    /* Resize full image to target_w × target_h */
    ds_image_t *resized = ds_image_resize(img, target_w, target_h);
    if (!resized) { *out_count = 0; return NULL; }

    /* Split into blocks and crop each */
    int n_patches = blocks + (use_thumbnail ? 1 : 0);
    ds_image_t **result = (ds_image_t **)malloc(n_patches * sizeof(ds_image_t *));
    if (!result) { ds_image_free(resized); *out_count = 0; return NULL; }

    int idx = 0;
    for (int r = 0; r < block_rows; r++) {
        for (int c = 0; c < block_cols; c++) {
            int left = c * image_size;
            int top = r * image_size;
            int right = left + image_size;
            int bottom = top + image_size;
            result[idx] = ds_image_crop_box(resized, left, top, right, bottom);
            if (!result[idx]) {
                for (int k = 0; k < idx; k++) ds_image_free(result[k]);
                free(result); ds_image_free(resized);
                *out_count = 0; return NULL;
            }
            idx++;
        }
    }

    ds_image_free(resized);

    /* Append thumbnail (original resized to image_size × image_size) */
    if (use_thumbnail) {
        ds_image_t *thumb = ds_image_pad(img, image_size, 127);
        if (!thumb) {
            for (int k = 0; k < idx; k++) ds_image_free(result[k]);
            free(result);
            *out_count = 0; return NULL;
        }
        result[idx++] = thumb;
    }

    *out_count = idx;
    if (ds_verbose >= 1)
        fprintf(stderr, "dynamic_preprocess: %d patches + %d thumbnail = %d images\n",
                blocks, use_thumbnail ? 1 : 0, idx);
    return result;
}

void ds_image_free(ds_image_t *img) {
    if (!img) return;
    if (img->pixels) {
        if (img->owns_stb)
            stbi_image_free(img->pixels);
        else
            free(img->pixels);
    }
    free(img);
}
