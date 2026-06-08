/*
 * ds_platform_ocr.m - macOS Vision OCR backend.
 */

#include "ds_platform_ocr.h"

#if defined(__APPLE__) && defined(USE_APPLE_VISION)

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static char *dup_nsstring_utf8(NSString *text) {
    const char *utf8 = [text UTF8String];
    if (!utf8) return NULL;
    size_t len = strlen(utf8);
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, utf8, len + 1);
    return out;
}

static CGFloat observation_top(VNRecognizedTextObservation *obs) {
    return obs.boundingBox.origin.y + obs.boundingBox.size.height;
}

static CGImageRef create_vision_image(CGImageRef src) {
    size_t src_w = CGImageGetWidth(src);
    size_t src_h = CGImageGetHeight(src);
    if (src_w == 0 || src_h == 0) return NULL;

    const CGFloat max_dim = 1280.0;
    CGFloat scale = MIN(1.0, max_dim / (CGFloat)MAX(src_w, src_h));
    size_t dst_w = (size_t)lrint((CGFloat)src_w * scale);
    size_t dst_h = (size_t)lrint((CGFloat)src_h * scale);
    if (dst_w < 1) dst_w = 1;
    if (dst_h < 1) dst_h = 1;

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    if (!color_space) return NULL;
    CGContextRef ctx = CGBitmapContextCreate(NULL, dst_w, dst_h, 8, 0,
                                             color_space,
                                             kCGImageAlphaNoneSkipLast |
                                             kCGBitmapByteOrder32Host);
    CGColorSpaceRelease(color_space);
    if (!ctx) return NULL;

    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextDrawImage(ctx, CGRectMake(0, 0, dst_w, dst_h), src);
    CGImageRef out = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    return out;
}

static NSArray<VNRecognizedTextObservation *> *sorted_observations(NSArray *observations) {
    return [observations sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        VNRecognizedTextObservation *left = (VNRecognizedTextObservation *)a;
        VNRecognizedTextObservation *right = (VNRecognizedTextObservation *)b;
        CGFloat left_top = observation_top(left);
        CGFloat right_top = observation_top(right);
        CGFloat top_delta = right_top - left_top;
        if (fabs(top_delta) > 0.015) {
            return top_delta > 0 ? NSOrderedDescending : NSOrderedAscending;
        }
        CGFloat x_delta = left.boundingBox.origin.x - right.boundingBox.origin.x;
        if (fabs(x_delta) > 0.001) {
            return x_delta > 0 ? NSOrderedDescending : NSOrderedAscending;
        }
        return NSOrderedSame;
    }];
}

char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    if (!image_path) return NULL;

    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:image_path];
        if (!path) return NULL;

        NSURL *url = [NSURL fileURLWithPath:path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot open image\n");
            return NULL;
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot decode image\n");
            return NULL;
        }

        CGImageRef vision_image = create_vision_image(image);
        CGImageRelease(image);
        if (!vision_image) {
            if (verbosity > 0) fprintf(stderr, "Apple Vision OCR failed: cannot prepare image\n");
            return NULL;
        }

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:vision_image options:@{}];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = accurate ? VNRequestTextRecognitionLevelAccurate
                                            : VNRequestTextRecognitionLevelFast;
        request.usesLanguageCorrection = YES;
        if (@available(macOS 11.0, *)) {
            request.recognitionLanguages = @[@"zh-Hans", @"en-US"];
        }

        NSError *error = nil;
        double start = CFAbsoluteTimeGetCurrent();
        BOOL ok = [handler performRequests:@[request] error:&error];
        double elapsed_ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0;
        if (!ok) {
            if (verbosity > 0) {
                fprintf(stderr, "Apple Vision OCR failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
            CGImageRelease(vision_image);
            return NULL;
        }
        CGImageRelease(vision_image);

        NSMutableString *text = [NSMutableString string];
        NSArray *sorted = sorted_observations(request.results ?: @[]);
        NSCharacterSet *trim_set = [NSCharacterSet whitespaceAndNewlineCharacterSet];
        for (VNRecognizedTextObservation *obs in sorted) {
            VNRecognizedText *candidate = [[obs topCandidates:1] firstObject];
            if (!candidate || candidate.string.length == 0) continue;
            NSString *line = [candidate.string stringByTrimmingCharactersInSet:trim_set];
            if (line.length == 0) continue;
            if (text.length > 0) [text appendString:@"\n"];
            [text appendString:line];
        }

        if (verbosity > 0) {
            fprintf(stderr, "Apple Vision OCR: %.0f ms, %lu lines\n",
                    elapsed_ms, (unsigned long)sorted.count);
        }
        return dup_nsstring_utf8(text);
    }
}

#else

char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    (void)image_path;
    (void)accurate;
    (void)verbosity;
    return NULL;
}

#endif
