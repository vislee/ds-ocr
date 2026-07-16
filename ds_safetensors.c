/*
 * ds_safetensors.c - Safetensors reader with multi-shard support
 * ds_safetensors.c — Safetensors 权重文件读取器（多分片支持）
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】权重加载的"基础设施"
 * ─────────────────────────────────────────────────────────────────────
 * 所有模型权重都通过本模块加载——它是推理引擎的"基石"。
 *
 * 【Safetensors 格式】(详见 03-推理引擎实战篇.md)
 *   ┌─────────────────────────────────────┐
 *   │ 8 bytes: N = header JSON 长度       │  ← 固定8字节，小端序uint64
 *   ├─────────────────────────────────────┤
 *   │ N bytes: header JSON                │  ← {"tensor_name": {"dtype":"BF16","shape":[1280,896],"data_offsets":[start,end]}}
 *   ├─────────────────────────────────────┤
 *   │ 数据区: 原始二进制 (BF16/F32/...)    │  ← 按data_offsets定位每个tensor
 *   └─────────────────────────────────────┘
 *
 * 【核心设计: mmap零拷贝】
 *   传统: open → read到内存 → 解析 → 占用完整内存
 *   mmap:  open → mmap映射 → 直接用指针访问 → 按需加载(页式调度)
 *
 *   优势:
 *   1. 零拷贝: 不需要将数据从内核空间复制到用户空间
 *   2. 惰性加载: 只有实际访问的页才从磁盘读入(MoE稀疏激活时大量权重不访问)
 *   3. 内存效率: 3B参数模型BF16约6GB，mmap后物理内存只占用实际访问的部分
 *
 * 【多分片支持】
 * 大模型权重通常分成多个文件(model-00001-of-00003.safetensors等)
 * 本模块自动检测目录下的safetensors文件，按文件名排序打开所有分片
 * multi_safetensors_find() 在所有分片中查找tensor，对调用者透明
 *
 * 【BF16零拷贝 vs F32转换】
 *   safetensors_get_bf16_direct() — 返回mmap区域内的BF16指针(零拷贝，O(1))
 *   safetensors_get_f32()         — 分配新内存，逐元素BF16→F32转换(O(n))
 *   解码器权重用BF16零拷贝（巨大，逐行转换开销可忽略）
 *   编码器权重用F32预转换（较小，批量计算更高效）
 *
 * Adapted from antirez/qwen-asr project.
 * ═══════════════════════════════════════════════════════════════════════
 */

#include "ds_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

/* ========================================================================
 * Minimal JSON parser for safetensors header
 * ======================================================================== */

/* 跳过JSON空白字符（空格、换行、回车、制表符）
 * 工具函数：将指针前进到第一个非空白字符位置 */
static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* parse_string — 解析JSON字符串值
 *
 * 从指针p当前位置解析一个带双引号的JSON字符串。
 * 支持标准JSON转义序列（\n, \t, \\, \", \/ 等）。
 *
 * @param p        [输入/输出] 指向当前解析位置的指针，解析后前进到结束引号之后
 * @param out      [输出] 解析出的字符串缓冲区
 * @param max_len  缓冲区最大长度（含NUL终止符）
 * @return         成功返回0，失败返回-1
 */
static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

/* parse_int — 解析JSON整数（含负数）
 *
 * 从指针p当前位置解析一个可选的负号后跟一串数字。
 * 注意：此函数不处理小数或科学计数法——safetensors的shape和offset都是纯整数。
 *
 * @param p  [输入/输出] 指向当前解析位置的指针，解析后前进到数字末尾
 * @return   解析出的整数（int64_t可覆盖BF16 offset等大数值）
 */
static int64_t parse_int(const char **p) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/* parse_dtype — 将JSON中的dtype字符串转换为枚举值
 *
 * safetensors规范支持的dtype: F32, F16, BF16, I32, I64, BOOL
 * 遇到不支持的类型返回DTYPE_UNKNOWN，由调用方自行处理。
 *
 * @param s  dtype字符串（如 "BF16"）
 * @return   对应的枚举值，未知返回DTYPE_UNKNOWN
 */
static safetensor_dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_F32;
    if (strcmp(s, "F16") == 0) return DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "I32") == 0) return DTYPE_I32;
    if (strcmp(s, "I64") == 0) return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

/* parse_tensor_entry — 解析单个张量的JSON元数据
 *
 * 解析JSON对象：{"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,33554432]}
 * 将解析结果填入safetensor_t结构体。
 *
 * @param p  [输入/输出] 指向JSON对象起始位置（'{'之后），解析后前进到'}'之后
 * @param t  [输出] 解析结果写入此张量结构体
 * @return   成功返回0，失败返回-1
 */
static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_whitespace(p);
    if (**p != '{') return -1;
    (*p)++;

    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') { (*p)++; continue; }

        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        skip_whitespace(p);
        if (**p != ':') return -1;
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                skip_whitespace(p);
                if (**p == ',') { (*p)++; continue; }
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            skip_whitespace(p);
            size_t start = (size_t)parse_int(p);
            skip_whitespace(p);
            if (**p == ',') (*p)++;
            skip_whitespace(p);
            size_t end = (size_t)parse_int(p);
            t->data_offset = start;
            t->data_size = end - start;
            skip_whitespace(p);
            if (**p == ']') (*p)++;
        } else {
            /* Skip unknown value */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '[') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '[') depth++;
                    else if (**p == ']') depth--;
                    (*p)++;
                }
            } else if (**p == '{') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '{') depth++;
                    else if (**p == '}') depth--;
                    (*p)++;
                }
            } else {
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }
    if (**p == '}') (*p)++;
    return 0;
}

/* parse_header — 解析完整的JSON header
 *
 * JSON header格式：
 *   {
 *     "tensor_name_1": { "dtype":"BF16","shape":[1280,896],"data_offsets":[0,2293760] },
 *     "tensor_name_2": { ... },
 *     "__metadata__": { ... }   ← 元数据对象，跳过不处理
 *   }
 *
 * 遍历JSON对象的每个键值对：
 *   - 键(key) = 张量名称（如 "vision_encoder.patch_embed.weight"）
 *   - 值(value) = 张量元数据对象（dtype, shape, data_offsets）
 *   - "__metadata__" 特殊键跳过（包含格式版本等元信息）
 *
 * @param sf  [输出] 解析结果写入sf->tensors[]数组，sf->num_tensors为解析数量
 * @return   成功返回0，失败返回-1
 */
static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;
    skip_whitespace(&p);
    if (*p != '{') return -1;
    p++;

    sf->num_tensors = 0;

    while (*p && *p != '}' && sf->num_tensors < SAFETENSORS_MAX_TENSORS) {
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0) return -1;
        skip_whitespace(&p);
        if (*p != ':') return -1;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        safetensor_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (parse_tensor_entry(&p, t) != 0) return -1;
        sf->num_tensors++;
    }
    return 0;
}

/* ========================================================================
 * Single file operations
 * ======================================================================== */

/* safetensors_open — 打开单个safetensors文件（mmap映射 + JSON解析）
 *
 * 处理流程：
 *   1. open() 打开文件 → 获取文件描述符 fd
 *   2. fstat() 获取文件大小 file_size
 *   3. mmap() 将整个文件映射到内存 → 返回基地址 data
 *   4. 读取前8字节 → 得到 header_size（little-endian uint64）
 *   5. 从 data+8 提取 header_size 字节的JSON字符串
 *   6. parse_header() 解析JSON → 填充 tensors[] 数组
 *
 * @param path  safetensors文件路径
 * @return      成功返回safetensors_file_t指针（调用者需safetensors_close释放），失败返回NULL
 */
safetensors_file_t *safetensors_open(const char *path) {
    /* 第1步: 以只读方式打开文件，获取文件描述符 */
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    /* 第2步: 获取文件大小，验证文件至少能容纳8字节的header长度 */
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) { close(fd); return NULL; }

    /* 第3步: mmap将整个文件映射到虚拟地址空间（零拷贝关键） */
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  /* 映射完成后可立即关闭fd */
    if (data == MAP_FAILED) return NULL;

    /* 第4步: 读取前8字节（little-endian uint64），这是JSON header的长度 */
    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    if (header_size > file_size - 8) { munmap(data, file_size); return NULL; }

    /* 第5步: 分配safetensors_file_t结构体，存储文件元信息 */
    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) { munmap(data, file_size); return NULL; }

    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = (size_t)header_size;

    /* 第6步: 复制JSON header字符串（从mmap区域复制到堆内存），并解析 */
    sf->header_json = malloc(header_size + 1);
    if (!sf->header_json) { safetensors_close(sf); return NULL; }
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';

    if (parse_header(sf) != 0) { safetensors_close(sf); return NULL; }

    return sf;
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    if (sf->data) munmap(sf->data, sf->file_size);
    free(sf->path);
    free(sf->header_json);
    free(sf);
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

int64_t safetensor_numel(const safetensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    int64_t n = safetensor_numel(t);
    if (n <= 0) return NULL;

    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;

    const void *data = safetensors_data(sf, t);

    switch (t->dtype) {
        case DTYPE_F32:
            memcpy(out, data, n * sizeof(float));
            break;
        case DTYPE_BF16: {
            const uint16_t *src = (const uint16_t *)data;
            for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            break;
        }
        default:
            free(out);
            return NULL;
    }
    return out;
}

int safetensor_is_bf16(const safetensor_t *t) {
    return t && t->dtype == DTYPE_BF16;
}

uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_BF16) return NULL;
    return (uint16_t *)safetensors_data(sf, t);
}

void safetensor_print(const safetensor_t *t) {
    const char *dtype_names[] = {"F32", "F16", "BF16", "I32", "I64", "BOOL"};
    const char *dtype_name = t->dtype >= 0 && t->dtype <= 5 ?
                             dtype_names[t->dtype] : "UNKNOWN";
    printf("%s: dtype=%s, shape=[", t->name, dtype_name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", (long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("]\n");
}

void safetensors_print_all(const safetensors_file_t *sf) {
    printf("File: %s (%d tensors)\n", sf->path, sf->num_tensors);
    for (int i = 0; i < sf->num_tensors; i++) safetensor_print(&sf->tensors[i]);
}

/* ========================================================================
 * Multi-shard operations
 * ======================================================================== */

multi_safetensors_t *multi_safetensors_open(const char *model_dir) {
    multi_safetensors_t *ms = calloc(1, sizeof(multi_safetensors_t));
    if (!ms) return NULL;

    char path[4096];

    /* Try single file first */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        ms->shards[0] = sf;
        ms->num_shards = 1;
        return ms;
    }

    /* Try multi-shard: model-00001-of-NNNNN.safetensors */
    for (int i = 1; i <= SAFETENSORS_MAX_SHARDS; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                 model_dir, i, i); /* placeholder - need to detect count */
        /* We don't know the total count yet, try common patterns */
        break;
    }

    /* Scan directory for shard files */
    DIR *dir = opendir(model_dir);
    if (!dir) { free(ms); return NULL; }

    struct dirent *entry;
    char shard_names[SAFETENSORS_MAX_SHARDS][256];
    int n_shards = 0;

    while ((entry = readdir(dir)) != NULL && n_shards < SAFETENSORS_MAX_SHARDS) {
        if (strncmp(entry->d_name, "model-", 6) == 0 &&
            strstr(entry->d_name, ".safetensors") != NULL) {
            snprintf(shard_names[n_shards], sizeof(shard_names[n_shards]),
                     "%s", entry->d_name);
            n_shards++;
        }
    }
    closedir(dir);

    if (n_shards == 0) {
        fprintf(stderr, "multi_safetensors_open: no safetensors files in %s\n", model_dir);
        free(ms);
        return NULL;
    }

    /* Sort shard names to ensure consistent ordering */
    qsort(shard_names, n_shards, 256, (int(*)(const void*,const void*))strcmp);

    /* Open each shard */
    for (int i = 0; i < n_shards; i++) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
        ms->shards[i] = safetensors_open(path);
        if (!ms->shards[i]) {
            fprintf(stderr, "multi_safetensors_open: failed to open %s\n", path);
            multi_safetensors_close(ms);
            return NULL;
        }
    }
    ms->num_shards = n_shards;
    return ms;
}

void multi_safetensors_close(multi_safetensors_t *ms) {
    if (!ms) return;
    for (int i = 0; i < ms->num_shards; i++) {
        safetensors_close(ms->shards[i]);
    }
    free(ms);
}

const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            safetensors_file_t **out_sf) {
    for (int s = 0; s < ms->num_shards; s++) {
        safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            if (strcmp(sf->tensors[i].name, name) == 0) {
                if (out_sf) *out_sf = sf;
                return &sf->tensors[i];
            }
        }
    }
    return NULL;
}
