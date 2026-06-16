/*
 * ds_dump.h - Tensor dump utilities for debugging C vs Python divergence
 *
 * Enable with: DS_DUMP_TENSORS=1 environment variable
 * Dumps matching tensors that can be compared with Python dump_tensors.py output.
 */

#ifndef DS_DUMP_H
#define DS_DUMP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

/* Global dump flag, checked once at startup */
static int g_dump_tensors = -1;  /* -1 = not yet checked */
extern int g_dump_crop_id;       /* Crop ID for per-crop dumps, -1 = no crop prefix. Defined in ds_ocr.c */

static int ds_dump_enabled(void) {
    if (g_dump_tensors == -1) {
        g_dump_tensors = (getenv("DS_DUMP_TENSORS") != NULL) ? 1 : 0;
    }
    return g_dump_tensors;
}

/* Dump tensor stats + first N values to stderr, matching Python dump format.
 * name: tensor name (must match Python hook name)
 * data: float32 data
 * n: total number of elements
 * shape_info: human-readable shape string e.g. "[256, 896]"
 */
static void ds_dump_tensor(const char *name, const float *data, int n,
                           const char *shape_info) {
    if (!ds_dump_enabled()) return;

    float sum = 0, max_val = -1e30f, min_val = 1e30f;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum += v;
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }
    float mean = sum / n;
    float var = 0;
    for (int i = 0; i < n; i++) {
        float d = data[i] - mean;
        var += d * d;
    }
    float std_val = sqrtf(var / n);
    float abs_sum = 0;
    for (int i = 0; i < n; i++) abs_sum += fabsf(data[i]);
    float abs_mean = abs_sum / n;

    fprintf(stderr, "[DUMP] %s shape=%s n=%d mean=%.6f std=%.6f abs_mean=%.6f min=%.6f max=%.6f\n",
            name, shape_info, n, mean, std_val, abs_mean, min_val, max_val);

    /* Print first 20 values */
    int show = n < 20 ? n : 20;
    fprintf(stderr, "[DUMP] %s first20=[", name);
    for (int i = 0; i < show; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%.6f", data[i]);
    }
    fprintf(stderr, "]\n");

    /* Also write binary .bin file for exact comparison */
    char fname[256];
    if (g_dump_crop_id >= 0)
        snprintf(fname, sizeof(fname), "dump/crop%d_%s.bin", g_dump_crop_id, name);
    else
        snprintf(fname, sizeof(fname), "dump/%s.bin", name);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

/* Dump 2D tensor in [rows, cols] layout */
static void ds_dump_tensor2d(const char *name, const float *data,
                              int rows, int cols, const char *shape_info) {
    if (!ds_dump_enabled()) return;
    ds_dump_tensor(name, data, rows * cols, shape_info);
}

/* Create dump directory */
static void ds_dump_init(void) {
    if (!ds_dump_enabled()) return;
#ifdef _WIN32
    _mkdir("dump");
#else
    mkdir("dump", 0755);
#endif
}

#endif /* DS_DUMP_H */
