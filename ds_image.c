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

ds_image_t *ds_image_load(const char *path) {
    int w, h, channels;
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

    if (ds_verbose >= 2)
        fprintf(stderr, "Loaded image: %s (%dx%d, %d channels)\n", path, w, h, channels);

    return img;
}

/* Bilinear interpolation helper */
static inline float ds_bilerp(const unsigned char *src, int src_w, int src_h,
                               float fx, float fy, int c) {
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1, y1 = y0 + 1;
    float dx = fx - x0, dy = fy - y0;

    /* Clamp to image bounds */
    if (x0 < 0) x0 = 0; if (x1 >= src_w) x1 = src_w - 1;
    if (y0 < 0) y0 = 0; if (y1 >= src_h) y1 = src_h - 1;

    float v00 = src[(y0 * src_w + x0) * 3 + c];
    float v10 = src[(y0 * src_w + x1) * 3 + c];
    float v01 = src[(y1 * src_w + x0) * 3 + c];
    float v11 = src[(y1 * src_w + x1) * 3 + c];

    float top = v00 * (1 - dx) + v10 * dx;
    float bot = v01 * (1 - dx) + v11 * dx;
    return top * (1 - dy) + bot * dy;
}

ds_image_t *ds_image_resize(const ds_image_t *img, int target_width, int target_height) {
    if (!img || !img->pixels) return NULL;

    unsigned char *out = (unsigned char *)malloc(target_width * target_height * 3);
    if (!out) return NULL;

    float x_scale = (float)img->width / target_width;
    float y_scale = (float)img->height / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float fx = (x + 0.5f) * x_scale - 0.5f;
            float fy = (y + 0.5f) * y_scale - 0.5f;

            for (int c = 0; c < 3; c++) {
                float val = ds_bilerp(img->pixels, img->width, img->height, fx, fy, c);
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
    return padded;
}

float *ds_image_to_float_chw(const ds_image_t *img) {
    if (!img || !img->pixels) return NULL;

    int n = img->width * img->height;
    float *out = (float *)malloc(3 * n * sizeof(float));
    if (!out) return NULL;

    /* Convert HWC (interleaved RGB) to CHW (planar), normalize to [-1, 1]
     * Using mean=0.5, std=0.5: (pixel/255 - 0.5) / 0.5 = pixel/255*2 - 1 */
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

void ds_image_free(ds_image_t *img) {
    if (!img) return;
    if (img->pixels) stbi_image_free(img->pixels);
    free(img);
}
