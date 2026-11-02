/*
 * ds_dump.h - Tensor dump utilities for debugging C vs Python divergence
 *
 * Enable with: DS_DUMP_TENSORS=1 environment variable
 * Dumps matching tensors that can be compared with Python dump_tensors.py output.
 *
 * 【设计动机】(docs/06 §6.3/6.4 两次位置编码事故的验证手段)
 * C 实现与 Python 参考的分叉靠肉眼和日志根本看不出来——幻觉/孤儿标签
 * 只是最末端的症状。本工具把 C 端关键中间量（patch_embed、QKV、neck、
 * 位置表插值结果）dump 成统计行 + 二进制 .bin，与 Python 端同名 hook
 * 的输出逐位对拍：mean/std/first20 一眼定差异在哪个层引入。
 *
 * 【使用模式】
 *   DS_DUMP_TENSORS=1 ./ds_ocr -d models/... -i img.png  → stderr 统计
 *                                                        + dump 目录下的 .bin
 *   tools/regress.sh / dump_tensors.py 消费 .bin 做精确比较
 *   g_dump_crop_id：多裁剪时给 dump 文件名加 crop 前缀，避免互踩
 */

#ifndef DS_DUMP_H
#define DS_DUMP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

/* Global dump flag, checked once at startup */
/* g_dump_tensors 三态: -1=还没查过环境变量, 1=开, 0=关（惰性初始化） */
static int g_dump_tensors = -1;  /* -1 = not yet checked */
extern int g_dump_crop_id;       /* Crop ID for per-crop dumps, -1 = no crop prefix. Defined in ds_ocr.c */

/* ds_dump_enabled — dump 总开关（惰性读取环境变量 DS_DUMP_TENSORS）。
 * 首次调用时 getenv 一次并缓存结果，之后 O(1) 返回——dump 关闭时
 * 对主流程的开销就是一次整数比较。变量"存在即开"（哪怕值为空串）。
 * @return 1 = 开启 dump，0 = 关闭。 */
static int ds_dump_enabled(void) {
    if (g_dump_tensors == -1) {   /* 还没检查过环境变量 */
        g_dump_tensors = (getenv("DS_DUMP_TENSORS") != NULL) ? 1 : 0;
    }                             /* 之后缓存住，整个进程只 getenv 一次 */
    return g_dump_tensors;
}

/* Dump tensor stats + first N values to stderr, matching Python dump format.
 * name: tensor name (must match Python hook name)
 * data: float32 data
 * n: total number of elements
 * shape_info: human-readable shape string e.g. "[256, 896]"
 */
/* ds_dump_tensor - 统计 + first20 + 二进制三件套。
 * name 必须与 Python hook 同名——对拍脚本靠名字配对。统计行放 stderr
 * （不污染 stdout 的 OCR 结果），二进制放 dump/ 目录做逐位比较。 */
static void ds_dump_tensor(const char *name, const float *data, int n,
                           const char *shape_info) {
    if (!ds_dump_enabled()) return;   /* 开关关闭：一次比较即返回，零侵入 */

    /* 第一遍扫描: sum / max / min——C 端与 Python 端都按同一口径统计 */
    float sum = 0, max_val = -1e30f, min_val = 1e30f;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum += v;
        if (v > max_val) max_val = v;   /* 初始值取 ±1e30，任意有限值都能被替换 */
        if (v < min_val) min_val = v;
    }
    float mean = sum / n;
    /* 第二遍扫描: 方差（两遍式最直白，避免单遍公式的数值稳定性坑） */
    float var = 0;
    for (int i = 0; i < n; i++) {
        float d = data[i] - mean;
        var += d * d;
    }
    float std_val = sqrtf(var / n);       /* 总体标准差（除 n 不除 n-1） */
    float abs_sum = 0;
    for (int i = 0; i < n; i++) abs_sum += fabsf(data[i]);
    float abs_mean = abs_sum / n;         /* 平均绝对幅度：mean 正负抵消时的量级参照 */

    /* 统计行打 stderr：与 Python hook 同格式，肉眼/脚本 diff 一眼定位分叉层 */
    fprintf(stderr, "[DUMP] %s shape=%s n=%d mean=%.6f std=%.6f abs_mean=%.6f min=%.6f max=%.6f\n",
            name, shape_info, n, mean, std_val, abs_mean, min_val, max_val);

    /* Print first 20 values */
    /* first20：n<20 时全打——统计量接近还看不出差异时，逐值对拍用 */
    int show = n < 20 ? n : 20;
    fprintf(stderr, "[DUMP] %s first20=[", name);
    for (int i = 0; i < show; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%.6f", data[i]);
    }
    fprintf(stderr, "]\n");

    /* Also write binary .bin file for exact comparison */
    /* 第三件套：全量 F32 原样写 .bin，供 regress/对拍脚本逐位比较——
     * 文件名带 tensor 名；多裁剪场景加 crop 前缀避免互相覆盖 */
    char fname[256];
    if (g_dump_crop_id >= 0)
        snprintf(fname, sizeof(fname), "dump/crop%d_%s.bin", g_dump_crop_id, name);
    else
        snprintf(fname, sizeof(fname), "dump/%s.bin", name);
    FILE *f = fopen(fname, "wb");      /* 打不开就静默放弃：dump 不能影响主流程 */
    if (f) {
        fwrite(data, sizeof(float), n, f);  /* 原始 F32 位型，不做任何变换 */
        fclose(f);
    }
}

/* Debug helpers — not every TU calls all of them, so mark them __unused
 * to keep them available for debugging without -Wunused-function noise. */
#define DS_DUMP_UNUSED __attribute__((unused))

/* Dump 2D tensor in [rows, cols] layout */
/* ds_dump_tensor2d — 二维张量便捷入口：内部展开成 rows*cols 调一维版。
 * shape_info 仍由调用方给（如 "[256, 896]"），便于对拍时人眼核对。 */
static void DS_DUMP_UNUSED ds_dump_tensor2d(const char *name, const float *data,
                              int rows, int cols, const char *shape_info) {
    if (!ds_dump_enabled()) return;
    ds_dump_tensor(name, data, rows * cols, shape_info);
}

/* Create dump directory */
/* ds_dump_init — 建 dump/ 输出目录（程序启动时调一次）。
 * 目录已存在时 mkdir 返回失败也无妨（errno=EEXIST，忽略）。
 * 不开启 dump 时是空操作。Windows 走 _mkdir（无权限位概念）。 */
static void DS_DUMP_UNUSED ds_dump_init(void) {
    if (!ds_dump_enabled()) return;
#ifdef _WIN32
    _mkdir("dump");
#else
    mkdir("dump", 0755);   /* 0755: 属主可写，其他人可读 */
#endif
}

#endif /* DS_DUMP_H */
