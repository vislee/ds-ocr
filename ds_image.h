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
    int owns_stb;           /* 1 if pixels from stbi_load (free with stbi_image_free), 0 if malloc'd */
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

/* Crop a rectangular region from the image (returns new ds_image_t, caller must free) */
ds_image_t *ds_image_crop_box(const ds_image_t *img, int left, int top, int right, int bottom);

/* Dynamic preprocess: split large images into patches for V2 encoder.
 * Returns array of ds_image_t* (caller must free each image and the array).
 * *out_count = number of images returned (patches + optional thumbnail).
 * For images <= 768x768, returns 1 image (just the original, padded).
 * For larger images, splits into patches and optionally appends thumbnail. */
ds_image_t **ds_dynamic_preprocess(const ds_image_t *img, int image_size,
                                    int min_num, int max_num,
                                    int use_thumbnail, int *out_count);

/* Free image resources */
void ds_image_free(ds_image_t *img);

#endif /* DS_IMAGE_H */
