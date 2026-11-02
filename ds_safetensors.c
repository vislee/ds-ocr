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
    if (**p != '"') return -1;   /* JSON字符串必须以双引号开头，否则不是合法的字符串值 */
    (*p)++;                      /* 跳过开引号，指向第一个内容字符 */
    size_t i = 0;                /* out 中已写入的字符计数 */
    while (**p && **p != '"' && i < max_len - 1) {
        /* 三重循环条件：\0 = JSON提前截断（文件损坏）；" = 正常结束；
         * max_len-1 = 给NUL终止符预留1字节，写满即停，绝不越界 */
        if (**p == '\\') {
            /* 转义序列：源文件占2字节（\ + 转义符），输出只产1字节 */
            (*p)++;              /* 前进到转义符本身 */
            if (**p == 'n') out[i++] = '\n';        /* \n → 换行 */
            else if (**p == 't') out[i++] = '\t';   /* \t → 制表符 */
            else if (**p == '"') out[i++] = '"';    /* \" → 引号本身 */
            else if (**p == '\\') out[i++] = '\\';  /* \\ → 反斜杠本身 */
            else out[i++] = **p; /* \/ 等未列举情形：宽松处理，取转义字符原样 */
        } else {
            out[i++] = **p;      /* 普通字符原样拷贝（张量名是ASCII标识符） */
        }
        (*p)++;                  /* 消费掉这个源字符 */
    }
    out[i] = '\0';               /* NUL终止成C字符串 */
    if (**p != '"') return -1;   /* 停下的位置不是结束引号 → JSON被截断/缓冲区写满，按失败处理 */
    (*p)++;                      /* 跳过结束引号——解析位置总是落在"已消费部分"之后，调用方无缝续读 */
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
    if (**p == '-') { neg = 1; (*p)++; }  /* 负号：shape/offset实际恒为非负，防御性支持 */
    while (**p >= '0' && **p <= '9') {
        /* 逐位累积："4096" → (((4*10+0)*10+9)*10+6)，即十进制Horner展开；
         * (**p - '0') 利用ASCII连续编码，字符差值就是数字面值 */
        val = val * 10 + (**p - '0');
        (*p)++;                           /* 消费这一位数字 */
    }
    return neg ? -val : val;   /* 停在首个非数字字符（','、']'、空白）上——
                                * 指针推进式解析器靠"停在边界"让调用方继续识别分隔符 */
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
    if (**p != '{') return -1;   /* 每个张量的元数据是一个JSON对象 */
    (*p)++;                      /* 跳过'{'，进入键值对序列 */

    /* 先置默认值：不规范的header缺某个键时，字段仍有确定的回退值 */
    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') { (*p)++; continue; }   /* 键值对之间的逗号，直接消费 */

        char key[64];   /* 键只会是 dtype/shape/data_offsets 这类短词，栈上64B足够 */
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        skip_whitespace(p);
        if (**p != ':') return -1;   /* JSON语法：键后必须跟冒号 */
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            /* dtype：字符串值（如 "BF16"）→ 转成枚举，后续用switch整型判断 */
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            /* shape：整数数组（如 [4096, 4096]）→ 逐维填入 shape[] */
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                /* 上限8维：与 safetensor_t.shape[8] 对应，超出即停（防数组溢出） */
                skip_whitespace(p);
                if (**p == ',') { (*p)++; continue; }   /* 维度间的逗号 */
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;   /* 消费闭括号（缺失也容忍——宽松解析） */
        } else if (strcmp(key, "data_offsets") == 0) {
            /* data_offsets：[start, end) 半开字节区间，相对【数据区】起点
             * （不是文件起点！）。end-start 直接得到字节数，免算 numel×元素宽 */
            if (**p != '[') return -1;
            (*p)++;
            skip_whitespace(p);
            size_t start = (size_t)parse_int(p);   /* 张量首字节在数据区内的偏移 */
            skip_whitespace(p);
            if (**p == ',') (*p)++;
            skip_whitespace(p);
            size_t end = (size_t)parse_int(p);     /* 张量末字节的下一位置（不含） */
            t->data_offset = start;
            t->data_size = end - start;
            skip_whitespace(p);
            if (**p == ']') (*p)++;
        } else {
            /* 未知键：按值的类型整段跳过——对将来新增字段向前兼容 */
            if (**p == '"') {
                /* 字符串值：扫到未转义的结束引号 */
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;    /* 先看转义符，避免把\"误判成结束引号 */
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '[') {
                /* 数组值：括号深度计数，嵌套多少层都能正确配对 */
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '[') depth++;
                    else if (**p == ']') depth--;
                    (*p)++;
                }
            } else if (**p == '{') {
                /* 对象值：同上，用花括号深度计数跳过整段 */
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '{') depth++;
                    else if (**p == '}') depth--;
                    (*p)++;
                }
            } else {
                /* 数字/true/false/null：扫到分隔符（',' 或 '}')为止 */
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }
    if (**p == '}') (*p)++;   /* 消费对象闭括号，解析位置交还调用方 */
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
/* parse_header - 解析 header JSON → tensors[] 数组
 * header 是 {"tensor_name": {dtype,shape,data_offsets}, ...} 的扁平对象。
 * 【__metadata__ 特殊键】部分导出工具会带一个字符串值的元数据对象，
 * 对本工程无用——用括号深度计数整段跳过（里面可能有嵌套结构）。
 * 张量条目由 parse_tensor_entry 解析（dtype/shape/data_offsets 三键）。 */
static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;   /* 解析游标：始终指向JSON文本的当前读取位置 */
    skip_whitespace(&p);
    if (*p != '{') return -1;          /* 整个header必须是一个大JSON对象 */
    p++;

    sf->num_tensors = 0;

    while (*p && *p != '}' && sf->num_tensors < SAFETENSORS_MAX_TENSORS) {
        /* 上限检查：超出 SAFETENSORS_MAX_TENSORS 的张量被静默丢弃而非崩溃
         * （教学引擎的取舍；生产实现应动态扩容或直接报错） */
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }   /* 键值对之间的逗号 */
        if (*p == '}') break;               /* 对象提前结束 */

        char name[256];   /* 张量名缓冲区，与 safetensor_t.name 同宽 */
        if (parse_string(&p, name, sizeof(name)) != 0) return -1;
        skip_whitespace(&p);
        if (*p != ':') return -1;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            /* __metadata__ 是导出工具附加的字符串值元信息（非张量）：
             * 用深度计数整段跳过，内部嵌套多少层都能正确配对 */
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;   /* 不占用 tensors[] 名额 */
        }

        safetensor_t *t = &sf->tensors[sf->num_tensors];   /* 追加到数组尾部 */
        snprintf(t->name, sizeof(t->name), "%s", name);    /* snprintf保证超长名NUL截断 */
        if (parse_tensor_entry(&p, t) != 0) return -1;
        sf->num_tensors++;
    }
    return 0;   /* 游标停在顶层'}'处即视为成功（不要求恰好消费到结尾） */
}

/* ========================================================================
 * Single file operations
 * ======================================================================== */

/* safetensors_open — 打开单个safetensors文件（mmap映射 + JSON解析）
 * (docs/03 §2.1-2.2；docs/09 §5.1-5.2)
 *
 * 处理流程：
 *   1. open() 打开文件 → 获取文件描述符 fd
 *   2. fstat() 获取文件大小 file_size
 *   3. mmap() 将整个文件映射到内存 → 返回基地址 data
 *   4. 读取前8字节 → 得到 header_size（little-endian uint64）
 *   5. 从 data+8 提取 header_size 字节的JSON字符串
 *   6. parse_header() 解析JSON → 填充 tensors[] 数组
 *
 * 【文件布局】(docs/03 §2.1)
 *   [8B header长度][header JSON][数据区: tensor1 | tensor2 | ...]
 *   data_offsets 是相对数据区起点（即文件偏移 8+header_size）的。
 *
 * 【为什么 mmap 而非 read】(docs/03 §2.2)
 *   零拷贝（不经过内核→用户缓冲复制）+ 惰性加载（只访问的页才读盘）。
 *   3B 参数模型 BF16 ≈6GB，MoE 稀疏激活意味着大量专家页永远不会被碰。
 *   header JSON 单独拷到堆：解析器要 '\0' 结尾，且 header 只读一次。
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

    /* 第3步: mmap建立【私有只读文件映射】——零拷贝的关键。
     *   PROT_READ   : 只读保护，权重页不可写（符合权重只读语义，写即SIGSEGV）
     *   MAP_PRIVATE : 私有写时复制映射；文件页直接共享内核page cache，无用户态副本
     *   offset=0    : 从文件首字节开始映射整个文件
     * 页对齐细节：内核按页（通常4KB）粒度映射，file_size 无需页对齐——映射
     * 长度向上取整到页边界，文件末尾不足一页的"部分页"由内核补零；但访问
     * 超出文件结尾的页会收到SIGBUS，所以一切越界检查都以 file_size 为准。 */
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  /* 映射建立后fd的使命完成，可立即关闭（已建立的映射不受影响） */
    if (data == MAP_FAILED) return NULL;

    /* 第4步: 读取前8字节（little-endian uint64），这是JSON header的长度。
     * 用 memcpy 而非 *(uint64_t*)data 强转：不假设mmap基址8字节对齐，
     * 规避未对齐读取的未定义行为；x86/ARM均为小端，内存字节序恰是
     * safetensors规范要求的LE，无需字节交换。 */
    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    if (header_size > file_size - 8) { munmap(data, file_size); return NULL; }
    /* ↑ 合法性校验：声称的header长度不得超过文件剩余部分——抵御损坏/恶意
     * 文件，否则下一步 memcpy 读header时会越过映射结尾触发SIGSEGV/SIGBUS */

    /* 第5步: 分配safetensors_file_t结构体，存储文件元信息 */
    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) { munmap(data, file_size); return NULL; }

    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = (size_t)header_size;

    /* 第6步: 复制JSON header字符串（mmap区 → 堆），并解析。
     * 为什么拷贝：mmap区内的JSON后随二进制数据而非NUL，而下面的指针
     * 推进式解析器按C字符串扫描（依赖'\0'兜底）；header只读一次，
     * 几MB的拷贝成本可忽略，却换来解析器实现简单+可放心越界读一个字节。 */
    sf->header_json = malloc(header_size + 1);
    if (!sf->header_json) { safetensors_close(sf); return NULL; }   /* 失败路径统一走close回收 */
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';   /* 补NUL终止符（+1字节的用途） */

    if (parse_header(sf) != 0) { safetensors_close(sf); return NULL; }

    return sf;
}

/* safetensors_close — 释放 safetensors_open 占用的全部资源
 * 顺序与 open 的每一步分配一一对应：munmap 文件映射 → free 路径副本
 * → free JSON 副本 → free 结构体。NULL 安全，可直接用于失败清理路径。
 *
 * @param sf  safetensors_open 返回的句柄（可为NULL） */
void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;                                /* NULL安全：调用方可无条件调用 */
    if (sf->data) munmap(sf->data, sf->file_size);  /* 解除mmap，权重页归还内核page cache */
    free(sf->path);                                 /* strdup 出来的路径副本 */
    free(sf->header_json);                          /* 堆上的JSON副本（注意：不是mmap内指针） */
    free(sf);                                       /* 最后释放结构体本身 */
}

/* safetensors_data — 计算张量数据在 mmap 区域内的起始地址
 * 指针算术：mmap基地址 + 8字节长度字段 + header JSON + 张量在数据区内的偏移，
 * 三段"跳过"正好对应文件布局的三层结构（见 03-推理引擎实战篇.md §2.1）。
 *
 * @param sf  张量所在的分片（multi_safetensors_find 的 out_sf 给出）
 * @param t   张量元数据（data_offset 是相对数据区起点的，不含header长度）
 * @return    只读数据指针——指向mmap内部，零拷贝，调用方不得free/写入 */
const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

/* safetensor_numel — 张量元素总数 = shape 各维的乘积
 * 0维标量返回1（循环不执行，初值1）。用于输出缓冲区分配、大小校验等。
 *
 * @param t  目标张量
 * @return   元素总数 */
int64_t safetensor_numel(const safetensor_t *t) {
    int64_t n = 1;   /* 用int64：4096²矩阵就有1600万元素，int在大4维张量下会溢出 */
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* bf16_to_f32 — 单个 BF16 值转 F32（纯位操作，零舍入损失）
 *
 * BF16 本质是"截断的F32"：1符号 + 8指数 + 7尾数（F32尾数23位）。
 * 转换只需把16位整体左移16位，对齐到F32的高半段：
 *   BF16: seee eeee emmm mmmm
 *   F32 : seee eeee emmm mmmm 0000 0000 0000 0000
 * 符号/指数原样保留，尾数低位补零——数值完全相等（精度不增也不损）。
 *
 * 用 memcpy 写回 float 而非 *(float*)&f32 指针强转：避免违反严格别名规则
 * （uint32_t* 与 float* 互访是UB）；编译器将其优化为一条寄存器移动，零开销。
 *
 * @param bf16  16位BF16位模式
 * @return      等值的float */
static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;   /* 高位对齐：BF16占据F32的符号+指数+高7位尾数 */
    float result;
    memcpy(&result, &f32, sizeof(float));    /* 位模式重解释，运行期零成本 */
    return result;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    /* F32 预转换路径 (docs/03 §2.4)：分配新内存 + 逐元素转换 O(n)。
     * 【选型策略】解码器权重（95%+）走 bf16_direct 零拷贝——解码是
     * 逐 token 推理，matvec 里 BF16→F32 在寄存器内 on-the-fly 做，
     * 权重不翻倍；编码器权重较小（SAM/CLIP ≈ 0.5GB），批量计算
     * （prefill 的 sgemm 需要 F32 输入），加载时一次预转换更划算。 */
    int64_t n = safetensor_numel(t);
    if (n <= 0) return NULL;

    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;

    const void *data = safetensors_data(sf, t);

    switch (t->dtype) {
        case DTYPE_F32:
            /* 源已是F32：整块memcpy最快（仍是复制一份，维持"返回新缓冲区"的接口约定） */
            memcpy(out, data, n * sizeof(float));
            break;
        case DTYPE_BF16: {
            /* BF16 → F32：逐元素左移16位展开，O(n)；n 可达数千万（4096²矩阵），
             * 所以编码器只对小权重用这条路，大权重走 bf16_direct 零拷贝 */
            const uint16_t *src = (const uint16_t *)data;
            for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            break;
        }
        default:
            /* F16/I32/I64/BOOL 不支持：本引擎的权重只出现 F32/BF16 两种 */
            free(out);
            return NULL;
    }
    return out;
}

/* safetensor_is_bf16 — 判断张量是否为BF16类型（NULL安全）
 * 权重加载的分流开关：BF16 → safetensors_get_bf16_direct() 零拷贝；
 * F32 → safetensors_get_f32() 预转换。
 *
 * @param t  目标张量（可为NULL）
 * @return   1=是BF16；0=不是或t为NULL */
int safetensor_is_bf16(const safetensor_t *t) {
    return t && t->dtype == DTYPE_BF16;
}

/* safetensors_get_bf16_direct - BF16 权重的零拷贝访问 (docs/03 §2.4)
 * 返回 mmap 区域内的原始 uint16 指针：O(1)、不复制、不转换。
 * 指针生命周期与 sf 相同——调用方绝不能 free 或写这个区域
 * （MAP_PRIVATE 只读映射，写会 SIGSEGV）。
 * matvec 内核按需把 uint16 左移 16 位还原成 F32（寄存器内完成），
 * 3GB 级解码器权重不翻倍——这是整个引擎内存效率的基石。 */
uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_BF16) return NULL;
    return (uint16_t *)safetensors_data(sf, t);
}

/* safetensor_print — 打印单个张量的元数据（调试用）
 * 输出形如：model.layers.0.mlp.gate.weight: dtype=BF16, shape=[64, 2048]
 * dtype_names[] 按枚举值顺序索引；UNKNOWN=-1 越界时回退打印 "UNKNOWN"。
 *
 * @param t  目标张量 */
void safetensor_print(const safetensor_t *t) {
    const char *dtype_names[] = {"F32", "F16", "BF16", "I32", "I64", "BOOL"};
    const char *dtype_name = t->dtype >= 0 && t->dtype <= 5 ?
                             dtype_names[t->dtype] : "UNKNOWN";
    printf("%s: dtype=%s, shape=[", t->name, dtype_name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", (long)t->shape[i], i < t->ndim - 1 ? ", " : "");   /* 逗号分隔，末维不加 */
    }
    printf("]\n");
}

/* safetensors_print_all — 打印文件内全部张量的元数据（调试用）
 * 加载后调用一次可核对：张量数量/名称是否齐全、dtype 是否符合预期。
 *
 * @param sf  已打开的safetensors文件 */
void safetensors_print_all(const safetensors_file_t *sf) {
    printf("File: %s (%d tensors)\n", sf->path, sf->num_tensors);
    for (int i = 0; i < sf->num_tensors; i++) safetensor_print(&sf->tensors[i]);
}

/* ========================================================================
 * Multi-shard operations
 * ======================================================================== */

multi_safetensors_t *multi_safetensors_open(const char *model_dir) {
    /* 多分片打开 (docs/03 §2.3)：优先单文件 model.safetensors；
     * 否则扫目录收集 model-*.safetensors 分片名（sorted 后逐个 open）。
     * 分片对调用方透明：multi_safetensors_find 在所有分片中统一找。 */
    multi_safetensors_t *ms = calloc(1, sizeof(multi_safetensors_t));
    if (!ms) return NULL;   /* calloc已清零：半初始化状态传给close也安全 */

    char path[4096];   /* 路径拼接缓冲区：4096 是多数系统的 PATH_MAX */

    /* Try single file first
     * 单文件是常见情形（小模型/单分片导出），先探测一次可省去目录扫描 */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        ms->shards[0] = sf;      /* 单文件也放进shards[0]：统一按分片访问 */
        ms->num_shards = 1;
        return ms;
    }

    /* Try multi-shard: model-00001-of-NNNNN.safetensors
     * 历史遗留的占位循环：本想按编号逐个探测分片，但不知道总分片数N就拼不出
     * 完整文件名（model-0000i-of-0000N），循环体第一轮即 break，实际不生效——
     * 真正的分片发现靠下面的目录扫描。 */
    for (int i = 1; i <= SAFETENSORS_MAX_SHARDS; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                 model_dir, i, i); /* placeholder - need to detect count */
        /* We don't know the total count yet, try common patterns */
        break;
    }

    /* Scan directory for shard files
     * opendir/readdir 是 POSIX 目录扫描三件套：opendir拿句柄，
     * readdir逐项返回 struct dirent（含 d_name 文件名），closedir归还 */
    DIR *dir = opendir(model_dir);
    if (!dir) { free(ms); return NULL; }   /* 目录打不开（不存在/无权限）：整体失败 */

    struct dirent *entry;
    char shard_names[SAFETENSORS_MAX_SHARDS][256];   /* 只存文件名，完整路径稍后拼 */
    int n_shards = 0;

    while ((entry = readdir(dir)) != NULL && n_shards < SAFETENSORS_MAX_SHARDS) {
        /* readdir 的返回顺序依文件系统而定（不保证字母序），必须收集后自行排序；
         * n_shards 同时充当上限护栏，防止异常目录塞爆数组 */
        if (strncmp(entry->d_name, "model-", 6) == 0 &&
            strstr(entry->d_name, ".safetensors") != NULL) {
            /* 匹配规则恰好命中 HF 分片命名 model-00001-of-00002.safetensors：
             * 前缀 "model-" 排除 model.safetensors（无连字符），
             * 后缀 ".safetensors" 排除 tokenizer.json 等其他文件 */
            snprintf(shard_names[n_shards], sizeof(shard_names[n_shards]),
                     "%s", entry->d_name);
            n_shards++;
        }
    }
    closedir(dir);   /* 目录流也是资源，扫描完立即归还 */

    if (n_shards == 0) {
        fprintf(stderr, "multi_safetensors_open: no safetensors files in %s\n", model_dir);
        free(ms);
        return NULL;
    }

    /* Sort shard names to ensure consistent ordering
     * 字典序 == 数值序：分片编号被 %05d 零填充成固定位宽（00001 < 00002），
     * 纯 strcmp 即得正确顺序，无需解析数字；分片内张量互不重叠，
     * 但查找时仍需确定性顺序以保证结果可复现。
     * （把 strcmp 强转成 qsort 比较函数指针是常见惯用法——参数布局恰好兼容。） */
    qsort(shard_names, n_shards, 256, (int(*)(const void*,const void*))strcmp);

    /* Open each shard */
    for (int i = 0; i < n_shards; i++) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
        ms->shards[i] = safetensors_open(path);
        if (!ms->shards[i]) {
            fprintf(stderr, "multi_safetensors_open: failed to open %s\n", path);
            multi_safetensors_close(ms);   /* 错误路径：回收已打开的部分。
             * 注意此刻 ms->num_shards 仍为 calloc 的 0，close 只会 free(ms)——
             * 前面已mmap的分片要等进程退出才由内核回收（教学实现的已知简化，
             * 生产代码应先 ms->num_shards = i 再调用close）。 */
            return NULL;
        }
    }
    ms->num_shards = n_shards;   /* 全部打开成功才记录数量——close 依赖此计数逐个释放 */
    return ms;
}

/* multi_safetensors_close — 逐个关闭所有分片并释放包装器
 * 与 multi_safetensors_open 的成功路径一一对应：每个分片各自
 * munmap + free，最后 free 包装器本身。
 *
 * @param ms  multi_safetensors_open 返回的句柄（可为NULL） */
void multi_safetensors_close(multi_safetensors_t *ms) {
    if (!ms) return;
    for (int i = 0; i < ms->num_shards; i++) {
        safetensors_close(ms->shards[i]);   /* 每个分片独立 munmap，互不影响 */
    }
    free(ms);   /* 最后释放包装器本身 */
}

const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            safetensors_file_t **out_sf) {
    /* 线性扫所有分片的所有 tensor。调用方每次权重访问都会经过这里
     * （ds_ocr.c 的 LOAD_* 宏），O(张量数) 的 strcmp 在加载期一次性
     * 场景下足够——不做索引表，直白优先。out_sf 告诉调用方 tensor
     * 属于哪个分片（取 bf16 直指针需要 sf->data + header_size）。 */
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

/* multi_safetensors_madvise_unused — 加载完成后丢弃映射页的常驻副本
 *
 * 权重加载会把「整份 mmap」读一遍（每个 tensor 都摸过），清洁文件页因此
 * 全部驻留；其中专家权重/lm_head 等已被拷贝进连续 malloc 块或 F32 化，
 * mmap 里的源页此后不再被读。对这些页做 MADV_DONTNEED 可立即归还物理页
 * （~4-5GB）：它们是 MAP_PRIVATE 文件映射的干净页，此后任何指针触碰都会
 * 从 page cache 廉价重入（无磁盘 I/O），正确性不受影响。
 * DS_NO_MADVISE=1 可关闭（调试对照用）。
 * 【与 docs/09 §5.5 冷/热态的关系】madvise 只影响 RSS 常驻，不影响跑分——
 * 热态基准测的是内存带宽，被访问的页会自然回到常驻集。 */
void multi_safetensors_madvise_unused(multi_safetensors_t *ms) {
    if (!ms) return;
    if (getenv("DS_NO_MADVISE")) return;   /* 环境变量开关：调试对照 RSS/性能时禁用本优化 */
    for (int i = 0; i < ms->num_shards; i++) {
        safetensors_file_t *sf = ms->shards[i];
        if (sf && sf->data && sf->file_size > 0)
            madvise(sf->data, sf->file_size, MADV_DONTNEED);   /* 按页丢弃整个映射的常驻副本：
             * 内核把这些干净文件页从进程驻留集中摘除；此后任何指针触碰都会
             * 触发缺页、从 page cache 重新映射（无磁盘I/O），正确性不受影响 */
    }
}
