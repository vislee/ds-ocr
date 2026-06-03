/*
 * ds_image.h - Image loading for DeepSeek-OCR (PNG/JPEG/WebP/BMP via stb_image)
 */

#ifndef DS_IMAGE_H
#define DS_IMAGE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    unsigned char *pixels;  /* RGB, 3 channels, row-major */
    int width;
    int height;
    int channels;           /* Always 3 after conversion */
} ds_image_t;

/* Load image from file (supports PNG, JPEG, WebP, BMP, TIFF, GIF, PSD, etc.)
 * Automatically converts to RGB (3 channels) and resizes to target size if needed.
 * Returns NULL on error. */
ds_image_t *ds_image_load(const char *path);

/* Resize image to target dimensions using bilinear interpolation */
ds_image_t *ds_image_resize(const ds_image_t *img, int target_width, int target_height);

/* Pad image to target size maintaining aspect ratio (like PIL ImageOps.pad).
 * Fills unused area with pad_color (128=gray). */
ds_image_t *ds_image_pad(const ds_image_t *img, int target_size, unsigned char pad_color);

/* Convert image to float32 tensor normalized to [0, 1]
 * Output: [3, height, width] in NCHW format (channel-first) */
float *ds_image_to_float_chw(const ds_image_t *img);

/* Free image resources */
void ds_image_free(ds_image_t *img);

#endif /* DS_IMAGE_H */
