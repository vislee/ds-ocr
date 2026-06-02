/*
 * stb_image.h - Image loading library (single-header)
 * 
 * Minimal implementation for ds_ocr: supports PNG, JPEG, BMP.
 * For the full version, see: https://github.com/nothings/stb
 *
 * This is a simplified version that wraps platform-native image loading.
 * For production use, replace with the full stb_image.h from nothings/stb.
 */

#ifndef STB_IMAGE_H
#define STB_IMAGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* API */
extern unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);
extern char const *stbi_failure_reason(void);

#ifdef __cplusplus
}
#endif

#endif /* STB_IMAGE_H */

#ifdef STB_IMAGE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char stbi__g_failure_reason[256] = "no error";

char const *stbi_failure_reason(void) {
    return stbi__g_failure_reason;
}

/* Check file extension */
static int stbi__ends_with(const char *str, const char *suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

/* Minimal JPEG reader using macOS ImageIO or libjpeg */
#ifdef __APPLE__
#include <ApplicationServices/ApplicationServices.h>

static unsigned char *stbi__load_macos(const char *filename, int *x, int *y, int *comp, int desired) {
    CFStringRef path = CFStringCreateWithCString(NULL, filename, kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, 0);
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
    
    unsigned char *result = NULL;
    size_t w = 0, h = 0;
    
    if (source) {
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        if (image) {
            w = CGImageGetWidth(image);
            h = CGImageGetHeight(image);
            *x = (int)w;
            *y = (int)h;
            *comp = desired > 0 ? desired : 4;
            
            size_t stride = w * (*comp);
            result = (unsigned char *)malloc(h * stride);
            
            CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            CGContextRef context = CGBitmapContextCreate(result, w, h, 8, stride, colorSpace,
                                                         desired == 3 ? kCGImageAlphaNone : kCGImageAlphaPremultipliedLast);
            if (context) {
                CGContextDrawImage(context, CGRectMake(0, 0, w, h), image);
                
                /* Flip vertically (CGImage origin is bottom-left) */
                unsigned char *row = (unsigned char *)malloc(stride);
                for (size_t i = 0; i < h / 2; i++) {
                    unsigned char *row1 = result + i * stride;
                    unsigned char *row2 = result + (h - 1 - i) * stride;
                    memcpy(row, row1, stride);
                    memcpy(row1, row2, stride);
                    memcpy(row2, row, stride);
                }
                free(row);
                
                CGContextRelease(context);
            }
            CGColorSpaceRelease(colorSpace);
            CGImageRelease(image);
        }
        CFRelease(source);
    }
    
    CFRelease(url);
    CFRelease(path);
    
    if (!result) {
        snprintf(stbi__g_failure_reason, sizeof(stbi__g_failure_reason),
                 "Failed to load image: %s", filename);
    }
    
    /* Convert to desired channel count if needed */
    if (result && desired > 0 && desired != *comp) {
        int src_comp = *comp;
        *comp = desired;
        size_t new_stride = w * desired;
        unsigned char *new_result = (unsigned char *)malloc(h * new_stride);
        for (size_t i = 0; i < h; i++) {
            for (size_t j = 0; j < w; j++) {
                unsigned char *src = result + i * w * src_comp + j * src_comp;
                unsigned char *dst = new_result + i * new_stride + j * desired;
                for (int c = 0; c < desired && c < src_comp; c++) dst[c] = src[c];
                for (int c = src_comp; c < desired; c++) dst[c] = 255;
            }
        }
        free(result);
        result = new_result;
    }
    
    return result;
}
#endif /* __APPLE__ */

unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels) {
#ifdef __APPLE__
    return stbi__load_macos(filename, x, y, channels_in_file, desired_channels);
#else
    /* Fallback: try to load using system commands or report error */
    (void)filename; (void)x; (void)y; (void)channels_in_file; (void)desired_channels;
    snprintf(stbi__g_failure_reason, sizeof(stbi__g_failure_reason),
             "Image loading not implemented on this platform. Install full stb_image.h.");
    return NULL;
#endif
}

void stbi_image_free(void *retval_from_stbi_load) {
    free(retval_from_stbi_load);
}

#endif /* STB_IMAGE_IMPLEMENTATION */
