/*
 * ds_tokenizer.c - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 * ds_tokenizer.c — Qwen2/Qwen3 BPE 分词器实现
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【BPE (Byte-Pair Encoding) 编码算法详解】
 * ─────────────────────────────────────────────────────────────────────
 *
 * 【Encode 流程】(text → token IDs)
 *   1. 将输入文本转为UTF-8字节序列
 *   2. 每个字节查基础词表得到初始token ID序列
 *   3. 迭代合并: 按合并规则优先级从高到低
 *      扫描相邻token对 (tokens[i], tokens[i+1])
 *      如果该对在merges中有记录且优先级最高 → 合并为新token
 *      重复直到无法合并
 *   4. 返回最终token ID序列
 *
 *   示例: "Hello world"
 *     UTF-8: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
 *     初始tokens: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
 *     合并1: (108,108) → 256  → [72, 101, 256, 111, 32, 119, 111, 114, 108, 100]
 *     合并2: (256,111) → 257  → [72, 101, 257, 32, 119, 111, 114, 108, 100]
 *     ...最终: [15496, 993]  ("Hello"=15496, " world"=993)
 *
 * 【Decode 流程】(token ID → text)
 *   简单查表: id_to_text[token_id] → 文本字符串
 *   多个token的文本拼接即得最终结果
 *
 * 【GPT-2字节级BPE的特殊编码】
 *   空格前缀: 在GPT-2 BPE中，词首空格编码为 Ġ (U+0120)
 *   例如: " Hello" → "ĠHello" → token 15496
 *   换行符: Ċ (U+010A)
 *   这些特殊字符在vocab.json中出现，decode时自动转换为实际字符
 *
 * 【DeepSeek-OCR词表结构】(vocab_size = 129280)
 *   0:      BOS (Beginning of Sequence)
 *   1:      EOS (End of Sequence)
 *   2:      PAD (Padding)
 *   3-255:  字节token (原始UTF-8字节)
 *   256+:   BPE合并token
 *   151655: IMAGE_START (V1/V2图像开始标记)
 *   151656: IMAGE_END (V1/V2图像结束标记)
 *   151657: IMAGE_NEWLINE (V1/V3图像行分隔符)
 *   128815: IMAGE_PLACEHOLDER (V3图像占位符)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Supports:
 *   - decode token IDs to UTF-8 text (for ASR output assembly)
 *   - encode UTF-8 text to token IDs using vocab.json + merges.txt
 */

#include "ds_tokenizer.h"
#include "ds_kernels.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * GPT-2 Bytes-to-Unicode Mapping
 * ======================================================================== */

static int gpt2_byte_to_unicode[256];
static int gpt2_unicode_to_byte[512]; /* codepoints up to ~384 */
static int gpt2_mapping_initialized = 0;

/* init_gpt2_mapping - 构建 GPT-2 byte-level 的"字节↔载体字符"双向映射表
 *
 * 【为什么需要这张表】(docs/07 §4)
 * BPE 在"可见字符"上工作，但文本本质是任意字节（含空格 0x20、控制字节、
 * UTF-8 续字节 0x80-0xBF 等不可打印值）。GPT-2 的解法：把 256 个字节值
 * 一一映射到 256 个"可打印载体码位"，之后的切分/合并全在载体字符上进行，
 * 彻底绕开"按字节操作会撕裂多字节字符"的问题——一个中文字"中"
 * (0xE4 0xB8 0xAD) 的 3 个字节各自变成载体字符，3 个 token 拼回
 * 仍输出合法 UTF-8，只是浪费了 3 个 token 额度。
 *
 * 【映射规则】"看得见的字节保持原样，看不见的移到 256+N 区"
 *   - 可打印区：b ∈ ['!','~'] ∪ ['¡','¬'] ∪ ['®','ÿ'] → 载体=自身；
 *     空格(0x20)、换行(0x0A)、全部控制字节与续字节被排除在外
 *   - 其余 68 个字节（空格/换行/控制/续字节）→ 256+n（n 递增）。
 *     著名成员：空格 0x20 → U+0120 'Ġ'；换行 0x0A → U+010A 'Ċ'。
 *     这就是词表里 'Ġpage' 的 Ġ 前缀、回车 token id 201 = 'Ċ' 的由来。
 *
 * 【实现要点】两张表一次构建、进程内复用（gpt2_mapping_initialized 门闩）：
 *   - gpt2_byte_to_unicode：字节 → 载体码位（encode 方向）
 *   - gpt2_unicode_to_byte：载体码位 → 字节（decode 方向），未知码位=-1
 * 反表只开 512 格：载体码位最大 = 256+67 = 323 < 512，查不到即 -1，
 * decode 遇 -1 写 '?'（理论不可达——词表 key 全部来自同一映射）。 */
static void init_gpt2_mapping(void) {
    if (gpt2_mapping_initialized) return;

    memset(gpt2_unicode_to_byte, -1, sizeof(gpt2_unicode_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        int is_normal = 0;
        if (b >= 33 && b <= 126) is_normal = 1;   /* '!'..'~' */
        if (b >= 161 && b <= 172) is_normal = 1;  /* '¡'..'¬' */
        if (b >= 174 && b <= 255) is_normal = 1;  /* '®'..'ÿ' */

        /* 三个"可打印"区间共 188+2 个字节保持自身；其余 68 个移入
         * 256+n 区。跳过 173 (­软连字符) 是官方设定，非遗漏。 */
        if (is_normal) gpt2_byte_to_unicode[b] = b;
        else gpt2_byte_to_unicode[b] = 256 + n++;
    }

    /* 反表：只记录 0-511 内的载体码位。所有载体码位 ≤323，
     * 因此 256 个字节全部能进反表。 */
    for (int b = 0; b < 256; b++) {
        int cp = gpt2_byte_to_unicode[b];
        if (cp < 512) gpt2_unicode_to_byte[cp] = b;
    }

    gpt2_mapping_initialized = 1;
}

/* Convert one Unicode codepoint to UTF-8 bytes. Returns bytes written. */
static int utf8_encode_cp(int cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/*
 * Decode a GPT-2 encoded token string (vocab key) to raw bytes/UTF-8 text.
 * Returns allocated string (caller must free).
 */
static char *decode_gpt2_token(const char *token_str) {
    init_gpt2_mapping();

    size_t len = strlen(token_str);
    unsigned char *bytes = (unsigned char *)malloc(len + 1);
    if (!bytes) return NULL;

    int byte_count = 0;
    const unsigned char *p = (const unsigned char *)token_str;
    const unsigned char *end = p + len;

    while (p < end) {
        int cp;
        int nbytes;

        if ((*p & 0x80) == 0) {
            cp = *p;
            nbytes = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p + 1 < end) cp |= (p[1] & 0x3F);
            nbytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 6;
            if (p + 2 < end) cp |= (p[2] & 0x3F);
            nbytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 12;
            if (p + 2 < end) cp |= (p[2] & 0x3F) << 6;
            if (p + 3 < end) cp |= (p[3] & 0x3F);
            nbytes = 4;
        } else {
            cp = *p;
            nbytes = 1;
        }
        p += nbytes;

        if (cp < 512 && gpt2_unicode_to_byte[cp] >= 0) {
            bytes[byte_count++] = (unsigned char)gpt2_unicode_to_byte[cp];
        } else {
            bytes[byte_count++] = '?';
        }
    }
    bytes[byte_count] = '\0';
    return (char *)bytes;
}

/*
 * Decode a DeepSeek BPE token string to UTF-8 text.
 * DeepSeek uses a mix:
 *   - Special tokens use ▁ (U+2581) for space (e.g., <｜begin▁of▁sentence｜>)
 *   - Regular BPE tokens use Ġ (U+0120) for space (GPT-2 byte-level encoding)
 * We handle both: replace ▁ with space first, then apply GPT-2 decoding.
 * Returns allocated string (caller must free).
 */
static char *decode_deepseek_token(const char *token_str) {
    init_gpt2_mapping();

    /* First pass: replace ▁ (U+2581 = 0xE2 0x96 0x81) with space,
     * keeping everything else as-is for GPT-2 decoding */
    size_t len = strlen(token_str);
    char *cleaned = (char *)malloc(len + 1);
    if (!cleaned) return NULL;

    size_t cw = 0;
    const unsigned char *sp = (const unsigned char *)token_str;
    while (*sp) {
        if (sp[0] == 0xE2 && sp[1] == 0x96 && sp[2] == 0x81) {
            cleaned[cw++] = ' ';  /* ▁ → space */
            sp += 3;
        } else {
            cleaned[cw++] = *sp++;
        }
    }
    cleaned[cw] = '\0';

    /* Second pass: apply GPT-2 byte-level decoding */
    size_t clen = strlen(cleaned);
    unsigned char *bytes = (unsigned char *)malloc(clen + 1);
    if (!bytes) { free(cleaned); return NULL; }

    int byte_count = 0;
    const unsigned char *p = (const unsigned char *)cleaned;
    const unsigned char *end = p + clen;

    while (p < end) {
        int cp, nbytes;
        if ((*p & 0x80) == 0) { cp = *p; nbytes = 1; }
        else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p + 1 < end) cp |= (p[1] & 0x3F);
            nbytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 6;
            if (p + 2 < end) cp |= (p[2] & 0x3F);
            nbytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 12;
            if (p + 2 < end) cp |= (p[2] & 0x3F) << 6;
            if (p + 3 < end) cp |= (p[3] & 0x3F);
            nbytes = 4;
        } else { cp = *p; nbytes = 1; }
        p += nbytes;

        if (cp < 512 && gpt2_unicode_to_byte[cp] >= 0) {
            bytes[byte_count++] = (unsigned char)gpt2_unicode_to_byte[cp];
        } else {
            bytes[byte_count++] = '?';
        }
    }
    bytes[byte_count] = '\0';
    free(cleaned);
    return (char *)bytes;
}

/* ========================================================================
 * Simple JSON parser for vocab.json
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* Skip a JSON value (string, number, object, array, bool, null) at *p.
 * Correctly handles nested strings containing { } [ ] characters. */
static void skip_json_value(const char **p) {
    skip_ws(p);
    if (**p == '"') {
        /* String: skip past closing quote, handling escapes */
        (*p)++;
        while (**p && **p != '"') {
            if (**p == '\\') (*p)++;  /* skip escaped char */
            if (**p) (*p)++;
        }
        if (**p == '"') (*p)++;
    } else if (**p == '{' || **p == '[') {
        /* Object or array: track nesting depth, skip strings inside */
        char open = **p, close = (**p == '{') ? '}' : ']';
        int depth = 1;
        (*p)++;
        while (**p && depth > 0) {
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else {
                if (**p == open) depth++;
                else if (**p == close) depth--;
                (*p)++;
            }
        }
    } else {
        /* Number, bool, null */
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

static int parse_json_string(const char **p, char *out, size_t max_len) {
    skip_ws(p);
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
            else if (**p == '/') out[i++] = '/';
            else if (**p == 'u') {
                (*p)++;
                unsigned int cp = 0;
                for (int j = 0; j < 4 && **p; j++, (*p)++) {
                    cp <<= 4;
                    char c = **p;
                    if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
                }
                if (cp < 0x80 && i + 1 < max_len) {
                    out[i++] = (char)cp;
                } else if (cp < 0x800 && i + 2 < max_len) {
                    out[i++] = (char)(0xC0 | (cp >> 6));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                } else if (i + 3 < max_len) {
                    out[i++] = (char)(0xE0 | (cp >> 12));
                    out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                }
                continue;
            } else {
                out[i++] = **p;
            }
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

static int64_t parse_json_int(const char **p) {
    skip_ws(p);
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    int64_t val = 0;
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/* ========================================================================
 * String->int hash map (open addressing)
 * docs/07 §10.2: FNV-1a + 开放寻址——词表/合并表共用的查找底座
 * ======================================================================== */

typedef struct {
    char *key; /* NULL means empty slot */
    int value;
} str_int_entry_t;

/* fnv1a_hash - FNV-1a 64 位哈希
 * 偏移基数 1469598103934665603、素数 1099511628211（FNV 官方常数）。
 * 选它因为：逐字节处理对"载体字符"UTF-8 串天然友好；实现 6 行；
 * 分布均匀性对本场景足够——负载因子压在 ≤0.5 时线性探查期望 O(1)。 */
static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

/* next_pow2 - 向上取 2 的幂。容量恒为 2ᵏ ⇒ 取模可以换成 &mask(cap-1)，
 * 省掉除法；配合容量=条目数×2，负载因子永远 ≤0.5。 */
static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* map_insert - 开放寻址 + 线性探查插入
 * 探查序列 (hash+i)&mask；遇空槽插入，遇同 key 就地更新 value。
 * 【所有权约定】(docs/07 §10.3) key 的内存归调用方：
 *   - vocab_map 传 tok->id_to_bpe[i] —— 借用，free 时绝不碰 key；
 *   - merge_map 传 strdup 出来的串 —— 自持，free 时逐槽 free(key)。
 * "谁拥有 key"在插入点决定，释放路径必须与之一一对应（bug 2 教训）。 */
static int map_insert(str_int_entry_t *map, int cap, char *key, int value) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) {              /* 空槽：占位 */
            map[idx].key = key;
            map[idx].value = value;
            return 0;
        }
        if (strcmp(map[idx].key, key) == 0) {  /* 已存在：更新值，key 不动 */
            map[idx].value = value;
            return 0;
        }
    }
    return -1;  /* 表满（正常不会发生：负载 ≤0.5） */
}

/* map_get - 查找。命中空槽即停（开放寻址不变量：无空洞截断探查链）。
 * 返回 value；未找到返回 -1（id 从非负，-1 可安全作哨兵）。 */
static int map_get(const str_int_entry_t *map, int cap, const char *key) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) return -1;     /* 探查链断在空洞 → 不存在 */
        if (strcmp(map[idx].key, key) == 0) return map[idx].value;
    }
    return -1;
}

/* ========================================================================
 * BPE helpers
 * docs/07 §3: 贪心合并——每轮找 rank 最小的相邻对合并，直到无可合并
 * ======================================================================== */

/* utf8_char_len - 由首字节判断 UTF-8 序列长度（1-4 字节）。
 * BPE 符号 = 一个完整码位，切分时绝不把多字节字符撕开。 */
static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  /* 非法首字节按 1 字节兜底（容错，理论不可达） */
}

/* split_utf8_symbols - 把载体字符文本按码位切成符号数组
 * encode 的第一步素材：每个符号是 malloc 的小串，后续合并按
 * "两符号拼串查表"进行。失败时释放已分配部分再返回 -1。 */
static int split_utf8_symbols(const char *s, char ***out_syms, int *out_n) {
    *out_syms = NULL;
    *out_n = 0;
    if (!s || !*s) return 0;

    /* 第一遍只数符号个数 → 精确 calloc，避免反复 realloc */
    int n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        p += l;
        n++;
    }
    if (n <= 0) return 0;

    char **syms = (char **)calloc((size_t)n, sizeof(char *));
    if (!syms) return -1;

    /* 第二遍逐符号拷贝。中途 malloc 失败要回收前面的符号（半初始化
     * 状态不能泄漏给调用方）。 */
    int i = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        syms[i] = (char *)malloc((size_t)l + 1);
        if (!syms[i]) {
            for (int j = 0; j < i; j++) free(syms[j]);
            free(syms);
            return -1;
        }
        memcpy(syms[i], p, (size_t)l);
        syms[i][l] = '\0';
        p += l;
        i++;
    }

    *out_syms = syms;
    *out_n = n;
    return 0;
}

static char *str_concat2(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *r = (char *)malloc(la + lb + 1);
    if (!r) return NULL;
    memcpy(r, a, la);
    memcpy(r + la, b, lb);
    r[la + lb] = '\0';
    return r;
}

/* merge_rank - 查相邻符号对 (a,b) 的合并优先级
 * rank = 训练时该对被学出的轮次（merges.txt 行号），越小越先合并。
 * 拼成 "a b"（空格分隔）查 merge_map；不在表中返回 INT_MAX。
 * docs/07 §3: 贪心正确性的根据——合并表按频率从高到低学出，rank 小的
 * 规则更"基础"，先合并 = 优先应用最可靠的规则，与训练生成顺序一致。 */
static int merge_rank(const ds_tokenizer_t *tok, const char *a, const char *b) {
    if (!tok->merge_map || tok->merge_map_cap <= 0) return INT_MAX;

    /* 每次查找都 malloc/拼串/释放，常数不小；本工程 prompt 只有几个词
     * （encode 结果被上游断言 ≤10 token），O(n²) 直白实现胜过复杂优化。 */
    size_t la = strlen(a), lb = strlen(b);
    char *pair = (char *)malloc(la + 1 + lb + 1);
    if (!pair) return INT_MAX;
    memcpy(pair, a, la);
    pair[la] = ' ';
    memcpy(pair + la + 1, b, lb);
    pair[la + 1 + lb] = '\0';

    int rank = map_get((const str_int_entry_t *)tok->merge_map, tok->merge_map_cap, pair);
    free(pair);
    return rank >= 0 ? rank : INT_MAX;
}

static int append_id(int **arr, int *n, int *cap, int id) {
    if (*n >= *cap) {
        int new_cap = (*cap == 0) ? 16 : (*cap * 2);
        int *tmp = (int *)realloc(*arr, (size_t)new_cap * sizeof(int));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = new_cap;
    }
    (*arr)[(*n)++] = id;
    return 0;
}

/* text_to_bpe_unicode - encode 第①步：字节 → 载体字符文本 (docs/07 §6)
 * "Multi page parsing." → "MultiĠpageĠparsing."（空格变 Ġ）。
 * 每个输入字节经映射表变成 1 个载体码位，最多占 3 字节 UTF-8，
 * 故输出缓冲 len*3+1。之后的合并全在这条"全可见"的串上进行。 */
static char *text_to_bpe_unicode(const char *text) {
    init_gpt2_mapping();
    size_t len = strlen(text);
    /* Every input byte becomes one codepoint up to 3 UTF-8 bytes. */
    char *out = (char *)malloc(len * 3 + 1);
    if (!out) return NULL;

    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = gpt2_byte_to_unicode[(unsigned char)text[i]];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        for (int j = 0; j < n; j++) out[w++] = tmp[j];
    }
    out[w] = '\0';
    return out;
}

/* encode_bpe_word - encode 第②步：把映射串当"一个词"做贪心合并 (docs/07 §6)
 *
 * 【为什么整段文本当一个词也能工作】官方 tokenizer.json 的 pre_tokenizer
 * 用四条正则先切段（数字 1-3 位一段、CJK 隔离、英文切词），保证合并只
 * 发生在段内。本实现省掉预分词：合并表学自"段内"，跨段相邻对在表里
 * 基本不存在 → 贪心循环自然停在正确边界。但"基本"≠"绝不"（数字长串、
 * 中英边界可能与官方有出入），上游 ds_ocr.c 因此设了三道保险：
 * 合理性检查(≤10 token) + fallback 常量 id + test_unit 钉扎断言。
 *
 * 【算法】循环 { 扫描全部相邻对，取 rank 最小者合并 } 直到无可合并。
 * 复杂度 O(n²) 次哈希查找（n=符号数）；工业实现用优先队列 O(n log n)，
 * 本工程追求直白。合并完成后逐符号查 vocab_map 得最终 id，
 * 任一符号查不到 → 整体失败返回 -1（调用方走 fallback，绝不带病输出）。 */
static int encode_bpe_word(const ds_tokenizer_t *tok, const char *mapped, int **out_ids, int *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    if (!mapped || !*mapped) return 0;

    char **syms = NULL;
    int n_syms = 0;
    if (split_utf8_symbols(mapped, &syms, &n_syms) != 0) return -1;
    if (n_syms <= 0) {
        free(syms);
        return 0;
    }

    /* 贪心主循环：每轮合并一个对，最多 n-1 轮 */
    while (n_syms > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < n_syms - 1; i++) {   /* 全扫 O(n) 个相邻对 */
            int r = merge_rank(tok, syms[i], syms[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0 || best_rank == INT_MAX) break;  /* 无可合并→收敛 */

        /* 二合一 + 数组左移补位。syms 的所有权在本函数内，
         * 失败路径要释放全部符号，不留半初始化状态。 */
        char *merged = str_concat2(syms[best_i], syms[best_i + 1]);
        if (!merged) {
            for (int i = 0; i < n_syms; i++) free(syms[i]);
            free(syms);
            return -1;
        }
        free(syms[best_i]);
        free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int j = best_i + 1; j < n_syms - 1; j++) syms[j] = syms[j + 1];
        n_syms--;
    }

    int *ids = NULL;
    int n_ids = 0, cap = 0;
    for (int i = 0; i < n_syms; i++) {
        int id = map_get((const str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, syms[i]);
        if (id < 0) {
            /* Should not happen with valid vocab + merges + byte-level mapping. */
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
        if (append_id(&ids, &n_ids, &cap, id) != 0) {
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
    }

    for (int i = 0; i < n_syms; i++) free(syms[i]);
    free(syms);

    *out_ids = ids;
    *out_n = n_ids;
    return 0;
}

/* derive_merges_path - 由 vocab.json 路径推出同级 merges.txt 路径
 * 两份文件必须同目录（模型仓库布局如此），用户只需传 vocab 一个路径。 */
static int derive_merges_path(const char *vocab_path, char *out_path, size_t out_cap) {
    const char *slash = strrchr(vocab_path, '/');
    if (!slash) {
        if (snprintf(out_path, out_cap, "merges.txt") >= (int)out_cap) return -1;
        return 0;
    }
    size_t dir_len = (size_t)(slash - vocab_path);
    if (dir_len + 12 >= out_cap) return -1;
    memcpy(out_path, vocab_path, dir_len);
    out_path[dir_len] = '\0';
    snprintf(out_path + dir_len, out_cap - dir_len, "/merges.txt");
    return 0;
}

static void trim_newline(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
}

static int parse_merge_pair(char *line, char **a, char **b) {
    *a = NULL;
    *b = NULL;
    trim_newline(line);
    if (line[0] == '\0') return -1;
    if (line[0] == '#') return -1;

    char *sp = strchr(line, ' ');
    if (!sp) return -1;
    *sp = '\0';
    char *p2 = sp + 1;
    while (*p2 == ' ') p2++;
    if (*line == '\0' || *p2 == '\0') return -1;
    *a = line;
    *b = p2;
    return 0;
}

/* load_merges_map - 加载 merges.txt → merge_map（key="a b" → value=rank）
 *
 * 【merges.txt 格式】每行 "A B"（GPT-2 载体字符），行号即 rank；
 * '#version' 注释行由 parse_merge_pair 以行首 '#' 过滤。
 *
 * 【两遍扫描】第一遍只数有效行数 → 精确定容量 next_pow2(n×2)
 * （负载因子 ≤0.5，127741 条规则 → 容量 262144=2¹⁸，约 4 MiB——
 * 对"4 个 token 的 prompt"奢侈，但换来加载后零重建、查询无锁）；
 * 第二遍真正插入。rank 递增赋值 = 文件行号 = 合并优先级。
 *
 * 【bug 1 的修复现场】(docs/07 §9) 初版解析器有两处失明：行首没有
 * '#' 过滤（'#version' 被当规则），以及不支持 'Ġ t' 这类载体对——
 * 导致 merges 一条都没读进来。现在 parse_merge_pair 统一处理。
 *
 * 【key 所有权】插入前 malloc 拼接（bug 2 修复：此前 strdup 缺失，
 * 所有槽指向同一个悬空行缓冲）；插入失败(表满/重复)时立即 free。 */
static int load_merges_map(ds_tokenizer_t *tok, const char *merges_path) {
    FILE *f = fopen(merges_path, "rb");
    if (!f) return -1;

    int n_pairs = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) == 0) n_pairs++;
    }
    if (n_pairs <= 0) {
        fclose(f);
        return -1;
    }

    tok->merge_map_cap = next_pow2(n_pairs * 2);
    tok->merge_map = calloc((size_t)tok->merge_map_cap, sizeof(str_int_entry_t));
    if (!tok->merge_map) {
        fclose(f);
        tok->merge_map_cap = 0;
        return -1;
    }

    rewind(f);
    int rank = 0;
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) != 0) continue;

        size_t la = strlen(a), lb = strlen(b);
        char *key = (char *)malloc(la + 1 + lb + 1);
        if (!key) continue;
        memcpy(key, a, la);
        key[la] = ' ';
        memcpy(key + la + 1, b, lb);
        key[la + 1 + lb] = '\0';

        if (map_insert((str_int_entry_t *)tok->merge_map, tok->merge_map_cap, key, rank) != 0) {
            free(key);
        }
        rank++;
    }
    fclose(f);
    return 0;
}

/* ========================================================================
 * Public API
 * docs/07 §6-§8: encode 三步走、decode 零拷贝查表、added_tokens 整词占位
 * ======================================================================== */

/* ds_tokenizer_load - 加载 vocab.json + merges.txt 构建分词器
 *
 * 【两阶段主流程】
 *   阶段1：通读 JSON，扫 max_id → 定 vocab_size → 分配 id_to_bpe/id_to_text
 *          → 回扫逐条填充（BPE 串 + 反变换后的 UTF-8 文本）
 *   阶段2：建 vocab_map（key 借用 id_to_bpe[i]）→ 读同级 merges.txt
 *          → 处理 added_tokens（必要时扩容数组，bug 3 修复）
 *
 * 【词表格式探测】DeepSeek 系词表用 ▁(U+2581) 表示空格（见
 * <｜begin▁of▁sentence｜>），GPT-2 系用 Ġ(U+0120)。扫前 50 个条目
 * 找 ▁ 的 UTF-8 字节 (0xE2 0x96 0x81) 判型，决定 decode 走哪条反变换。
 *
 * 【JSON 解析是手写的】只认 "key": int 这一种形态（vocab 对象的值全是
 * id），避免拉入 cJSON 依赖——qwen-asr 同款取舍。
 *
 * 【bug 3 的修复现场】(docs/07 §8-§9) added_tokens（<|det|>、
 * <image> 等 830 个整词占位符）住在顶层 "added_tokens" 数组、id 可超
 * vocab_size（如 <image>=128815）。初版没并入 → 只加载了 3 个特殊 token，
 * 生成 "<|det|>" 时输出乱码。修复 = 预扫 added_tokens 最大 id → realloc
 * 扩容 id_to_bpe/id_to_text → 逐条 strdup 原文（不做载体反变换，decode
 * 出来就是字面标签串，供后处理匹配）。 */
ds_tokenizer_t *ds_tokenizer_load(const char *vocab_json_path) {
    FILE *f = fopen(vocab_json_path, "rb");
    if (!f) {
        /* Not an error — caller will fallback to tokenizer.json */
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, f) != (size_t)file_size) {
        fclose(f);
        free(json);
        return NULL;
    }
    fclose(f);
    json[file_size] = '\0';

    /* Detect tokenizer type: DeepSeek/Llama uses ▁ (U+2581) for space,
     * GPT-2 uses Ġ (U+0120). Check first few tokens to decide. */
    int is_deepseek_tokenizer = 0;
    {
        const char *peek = json;
        skip_ws(&peek);
        if (*peek == '{') {
            peek++;
            /* Scan first ~20 entries looking for ▁ */
            int checked = 0;
            while (*peek && *peek != '}' && checked < 50) {
                skip_ws(&peek);
                if (*peek == ',' || *peek == '}') { if (*peek == ',') peek++; continue; }
                char test_key[4096];
                
                if (parse_json_string(&peek, test_key, sizeof(test_key)) == 0) {
                    /* Check if token contains ▁ (0xE2 0x96 0x81) */
                    if (strstr(test_key, "\xe2\x96\x81")) is_deepseek_tokenizer = 1;
                    checked++;
                }
                /* Skip value */
                skip_ws(&peek);
                if (*peek == ':') { peek++; skip_ws(&peek); parse_json_int(&peek); }
            }
        }
    }

    int max_id = 0;
    const char *p = json;
    skip_ws(&p);
    if (*p != '{') {
        free(json);
        return NULL;
    }
    p++;

    const char *saved = p;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);
        if (id > max_id) max_id = (int)id;
    }

    int vocab_size = max_id + 1;
    ds_tokenizer_t *tok = (ds_tokenizer_t *)calloc(1, sizeof(ds_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }
    tok->vocab_size = vocab_size;
    tok->id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
    tok->id_to_bpe = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->id_to_text || !tok->id_to_bpe) {
        ds_tokenizer_free(tok);
        free(json);
        return NULL;
    }

    p = saved;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);

        if (id >= 0 && id < vocab_size) {
            free(tok->id_to_bpe[id]);
            free(tok->id_to_text[id]);
            tok->id_to_bpe[id] = strdup(key);
            tok->id_to_text[id] = is_deepseek_tokenizer ?
                decode_deepseek_token(key) : decode_gpt2_token(key);
        }
    }
    free(json);

    int n_vocab_entries = 0;
    for (int i = 0; i < vocab_size; i++) if (tok->id_to_bpe[i]) n_vocab_entries++;
    tok->vocab_map_cap = next_pow2(n_vocab_entries * 2 + 1);
    tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
    if (!tok->vocab_map) {
        ds_tokenizer_free(tok);
        return NULL;
    }
    for (int i = 0; i < vocab_size; i++) {
        if (tok->id_to_bpe[i]) {
            map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
        }
    }

    /* Load added_tokens from tokenizer.json (special tokens like <image>, <td>, etc.)
     * These are in a top-level "added_tokens" array, separate from the vocab object.
     * Added tokens may have ids beyond vocab_size (e.g. 128815 for <image>), so
     * we need to expand id_to_bpe/id_to_text arrays if any added token id exceeds vocab_size. */
    {
        /* Pre-scan: find max id in added_tokens */
        int64_t max_added_id = -1;
        {
            const char *ps = json;
            while (*ps) {
                const char *pf = strstr(ps, "\"added_tokens\"");
                if (!pf) break;
                const char *pb = pf - 1;
                while (pb > json && (*pb == ' ' || *pb == '\t' || *pb == '\n' || *pb == '\r'))
                    pb--;
                if (*pb == '{' || *pb == ',') {
                    const char *pa = pf + strlen("\"added_tokens\"");
                    skip_ws(&pa);
                    if (*pa == ':') {
                        pa++; skip_ws(&pa);
                        if (*pa == '[') {
                            pa++;
                            while (*pa && *pa != ']') {
                                skip_ws(&pa); if (*pa==']') break; if (*pa==','){pa++;continue;}
                                if (*pa!='{'){pa++;continue;} pa++;
                                int64_t ps_id = -1;
                                while (*pa && *pa != '}') {
                                    skip_ws(&pa); if (*pa=='}') break; if (*pa==','){pa++;continue;}
                                    char psk[256];
                                    if (parse_json_string(&pa, psk, sizeof(psk)) != 0) { pa++; continue; }
                                    skip_ws(&pa); if (*pa != ':') break; pa++; skip_ws(&pa);
                                    if (strcmp(psk, "id") == 0) ps_id = parse_json_int(&pa);
                                    else { skip_json_value(&pa); }
                                }
                                if (*pa == '}') pa++;
                                if (ps_id > max_added_id) max_added_id = ps_id;
                            }
                        }
                        break; /* only first added_tokens array */
                    }
                }
                ps = pf + 1;
            }
        }

        /* Expand arrays if needed */
        if (max_added_id >= 0 && max_added_id >= vocab_size) {
            int new_size = (int)max_added_id + 1;
            char **new_bpe = (char **)realloc(tok->id_to_bpe, (size_t)new_size * sizeof(char *));
            char **new_text = (char **)realloc(tok->id_to_text, (size_t)new_size * sizeof(char *));
            if (new_bpe && new_text) {
                memset(new_bpe + vocab_size, 0, (size_t)(new_size - vocab_size) * sizeof(char *));
                memset(new_text + vocab_size, 0, (size_t)(new_size - vocab_size) * sizeof(char *));
                tok->id_to_bpe = new_bpe;
                tok->id_to_text = new_text;
                vocab_size = new_size;
                tok->vocab_size = new_size;
            }
        }

        int n_added = 0;
        const char *at = json;
        /* Find "added_tokens" key */
        while (*at) {
            const char *found = strstr(at, "\"added_tokens\"");
            if (!found) break;
            /* Verify it's a top-level key (preceded by { or ,) */
            const char *before = found - 1;
            while (before > json && (*before == ' ' || *before == '\t' || *before == '\n' || *before == '\r'))
                before--;
            if (*before == '{' || *before == ',') {
                at = found + strlen("\"added_tokens\"");
                skip_ws(&at);
                if (*at == ':') {
                    at++;
                    skip_ws(&at);
                    if (*at == '[') {
                        at++; /* skip [ */
                        while (*at && *at != ']') {
                            skip_ws(&at);
                            if (*at == ']') break;
                            if (*at == ',') { at++; continue; }
                            if (*at != '{') { at++; continue; }
                            at++; /* skip { */
                            char at_content[4096] = {0};
                            int64_t at_id = -1;
                            while (*at && *at != '}') {
                                skip_ws(&at);
                                if (*at == '}') break;
                                if (*at == ',') { at++; continue; }
                                char at_key[256];
                                if (parse_json_string(&at, at_key, sizeof(at_key)) != 0) { at++; continue; }
                                skip_ws(&at);
                                if (*at != ':') break;
                                at++;
                                skip_ws(&at);
                                if (strcmp(at_key, "id") == 0) {
                                    at_id = parse_json_int(&at);
                                } else if (strcmp(at_key, "content") == 0) {
                                    parse_json_string(&at, at_content, sizeof(at_content));
                                } else {
                                    /* Skip value: string, number, bool, null */
                                    if (*at == '"') { char tmp[4096]; parse_json_string(&at, tmp, sizeof(tmp)); }
                                    else if (*at == '-' || (*at >= '0' && *at <= '9')) parse_json_int(&at);
                                    else { while (*at && *at != ',' && *at != '}') at++; }
                                }
                            }
                            if (*at == '}') at++;
                            if (at_id >= 0 && at_id < vocab_size && at_content[0]) {
                                free(tok->id_to_text[(int)at_id]);
                                free(tok->id_to_bpe[(int)at_id]);
                                tok->id_to_bpe[(int)at_id] = strdup(at_content);
                                /* Special tokens are stored as-is (not byte-level encoded) */
                                tok->id_to_text[(int)at_id] = strdup(at_content);
                                n_added++;
                            }
                        }
                    }
                    break; /* processed added_tokens */
                }
            }
            at = found + 1;
        }

        if (n_added > 0) {
            /* Rebuild vocab_map with expanded vocab to include added_tokens */
            free(tok->vocab_map);
            int nve = 0;
            for (int i = 0; i < vocab_size; i++) if (tok->id_to_bpe[i]) nve++;
            tok->vocab_map_cap = next_pow2(nve * 2 + 1);
            tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
            if (tok->vocab_map) {
                for (int i = 0; i < vocab_size; i++) {
                    if (tok->id_to_bpe[i])
                        map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
                }
            }
            if (ds_verbose >= 1)
                fprintf(stderr, "Tokenizer: loaded %d added_tokens (vocab expanded to %d)\n", n_added, vocab_size);
        }
    }

    char merges_path[1024];
    if (derive_merges_path(vocab_json_path, merges_path, sizeof(merges_path)) == 0) {
        if (load_merges_map(tok, merges_path) != 0 && ds_verbose >= 2) {
            fprintf(stderr, "Tokenizer: merges not loaded from %s (encoding falls back to byte-level)\n",
                    merges_path);
        }
    }

    return tok;
}

/* ds_tokenizer_decode - id → UTF-8 文本（零拷贝 O(1) 查表）(docs/07 §7)
 *
 * id_to_text[id] 在加载时已做完全部反变换（Ġ/Ċ 载体还原 + ▁→空格），
 * 运行时只查表返指针：零分配、零拷贝。
 *
 * 【返回内部指针的纪律】不要跨调用保存——表的生命周期与 tokenizer
 * 相同，free(tok) 后指针即失效。推理主循环的用法是立刻 memcpy 进
 * 输出缓冲；多字节中文无需特殊处理，逐 piece 追加字节，UTF-8 自然愈合。
 * added_tokens 的 id_to_text 是字面标签串（"<|det|>"），供后处理匹配。 */
const char *ds_tokenizer_decode(const ds_tokenizer_t *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return "";
    return tok->id_to_text[token_id] ? tok->id_to_text[token_id] : "";
}

/* ds_tokenizer_encode - 文本 → token id 数组 (docs/07 §6)
 * 完整路径：text_to_bpe_unicode（字节→载体）→ encode_bpe_word（贪心合并
 * + 查词表）→ 返回 malloc 的 id 数组（调用者 free）。
 * 【注意】特殊 token（<|det|> 等）不经过本函数——喂给 encode 只会得到
 * "<"、"|"、"de" 之类的碎片；prompt 拼装时一律用常量 id 直接拼进序列
 * （ds_ocr.h: DS_TOKEN_BOS=0、DS_TOKEN_IMAGE_PLACEHOLDER=128815）。
 * 失败返回 NULL：上游 ds_ocr.c 用 fallback 常量 id 兜底，绝不带病输出。 */
int *ds_tokenizer_encode(const ds_tokenizer_t *tok, const char *text, int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!tok || !text || text[0] == '\0') return NULL;

    char *mapped = text_to_bpe_unicode(text);
    if (!mapped) return NULL;

    int *ids = NULL;
    int n_ids = 0;
    if (encode_bpe_word(tok, mapped, &ids, &n_ids) != 0) {
        free(mapped);
        free(ids);
        return NULL;
    }
    free(mapped);

    if (out_n_tokens) *out_n_tokens = n_ids;
    return ids;
}

/* ds_tokenizer_free - 释放分词器。
 * 所有权纪律的最终体现 (docs/07 §10.3)：
 *   - id_to_text/id_to_bpe：自持，逐条 free；
 *   - merge_map：key 自持（load 时 strdup/malloc 拼接的），逐槽 free(key)；
 *   - vocab_map：key 借用 id_to_bpe[i]，只 free 桶数组本身，
 *     绝不 double-free key——key 的真身已随 id_to_bpe 循环释放。
 * 一旦两者顺序/对象搞错，ASan (make debug) 会当场报 double-free。 */
void ds_tokenizer_free(ds_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->id_to_text) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_text[i]);
        free(tok->id_to_text);
    }
    if (tok->id_to_bpe) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_bpe[i]);
        free(tok->id_to_bpe);
    }

    if (tok->merge_map) {
        str_int_entry_t *m = (str_int_entry_t *)tok->merge_map;
        for (int i = 0; i < tok->merge_map_cap; i++) free(m[i].key);
        free(tok->merge_map);
    }
    free(tok->vocab_map);
    free(tok);
}

/* ──────────────────────────────────────────────────────────────
 * ds_tokenizer_load_from_tokenizer_json
 *
 * Load tokenizer from HuggingFace tokenizer.json format.
 * Structure: { "model": { "vocab": {token: id, ...}, "merges": ["a b", ...] },
 *              "added_tokens": [{id, content, ...}, ...] }
 *
 * 【与 ds_tokenizer_load 的关系】单文件替代方案：vocab 和 merges 都在
 * tokenizer.json 里（merges 是 "a b" 字符串数组，数组序即 rank），无需
 * 同目录 merges.txt。解析器同样手写、只认本项目用到的字段形态。
 * 【rank 语义】两条加载路径都必须保证 merges.txt 行号 == tokenizer.json
 * 数组序——rank 决定贪心合并优先级，语义错位会静默产出错误切分
 * （test_unit 的钉扎断言可捕获此类回归）。
 * ────────────────────────────────────────────────────────────── */
ds_tokenizer_t *ds_tokenizer_load_from_tokenizer_json(const char *tokenizer_json_path) {
    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, f) != (size_t)file_size) {
        fclose(f); free(json); return NULL;
    }
    fclose(f);
    json[file_size] = '\0';

    /* Navigate to model.vocab */
    const char *vocab_start = NULL;
    {
        const char *p = json;
        skip_ws(&p);
        if (*p != '{') { free(json); return NULL; }
        p++; /* skip outer { */
        while (*p && *p != '}') {
            skip_ws(&p);
            if (*p == ',') { p++; continue; }
            if (*p == '}') break;
            char key[256];
            if (parse_json_string(&p, key, sizeof(key)) != 0) { p++; continue; }
            skip_ws(&p);
            if (*p != ':') break;
            p++; skip_ws(&p);
            if (strcmp(key, "model") == 0 && *p == '{') {
                p++; /* skip model { */
                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == '}') break;
                    char mkey[256];
                    if (parse_json_string(&p, mkey, sizeof(mkey)) != 0) { p++; continue; }
                    skip_ws(&p);
                    if (*p != ':') break;
                    p++; skip_ws(&p);
                    if (strcmp(mkey, "vocab") == 0 && *p == '{') {
                        vocab_start = p;
                        goto found_vocab;
                    }
                    /* Skip non-vocab value */
                    skip_json_value(&p);
                }
            }
            /* Skip non-model value */
            skip_json_value(&p);
        }
    }
found_vocab:
    if (!vocab_start) { free(json); return NULL; }

    /* Find end of vocab object (matching }) */
    const char *vocab_end = vocab_start;
    {
        int depth = 0;
        while (*vocab_end) {
            if (*vocab_end == '"') { vocab_end++; while (*vocab_end && *vocab_end != '"') { if (*vocab_end == '\\') vocab_end++; vocab_end++; } if (*vocab_end) vocab_end++; continue; }
            if (*vocab_end == '{') depth++;
            else if (*vocab_end == '}') { depth--; if (depth == 0) { vocab_end++; break; } }
            vocab_end++;
        }
    }

    /* Extract vocab as standalone JSON string and parse with same logic as ds_tokenizer_load */
    size_t vlen = vocab_end - vocab_start;
    char *vjs = (char *)malloc(vlen + 1);
    if (!vjs) { free(json); return NULL; }
    memcpy(vjs, vocab_start, vlen);
    vjs[vlen] = '\0';

    /* Detect tokenizer type */
    int is_deepseek_tok = 0;
    {
        const char *pk = vjs; skip_ws(&pk);
        if (*pk == '{') { pk++; int ch=0;
            while(*pk&&*pk!='}'&&ch<50){ skip_ws(&pk); if(*pk==','||*pk=='}'){if(*pk==',')pk++;continue;}
                char tk[4096]; if(parse_json_string(&pk,tk,sizeof(tk))==0){if(strstr(tk,"\xe2\x96\x81"))is_deepseek_tok=1;ch++;}
                skip_ws(&pk); if(*pk==':'){pk++;skip_ws(&pk);parse_json_int(&pk);} } }
    }

    /* First pass: max id */
    int max_id = 0;
    const char *p = vjs; skip_ws(&p);
    if (*p != '{') { free(vjs); free(json); return NULL; }
    p++;
    const char *saved = p;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;
        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);
        if (id > max_id) max_id = (int)id;
    }

    int vocab_size = max_id + 1;
    ds_tokenizer_t *tok = (ds_tokenizer_t *)calloc(1, sizeof(ds_tokenizer_t));
    if (!tok) { free(vjs); free(json); return NULL; }
    tok->vocab_size = vocab_size;
    tok->id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
    tok->id_to_bpe = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->id_to_text || !tok->id_to_bpe) { ds_tokenizer_free(tok); free(vjs); free(json); return NULL; }

    /* Second pass: fill entries */
    p = saved;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;
        char key[4096];
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);
        if (id >= 0 && id < vocab_size) {
            free(tok->id_to_bpe[id]); free(tok->id_to_text[id]);
            tok->id_to_bpe[id] = strdup(key);
            tok->id_to_text[id] = is_deepseek_tok ? decode_deepseek_token(key) : decode_gpt2_token(key);
        }
    }

    /* Build vocab hash map */
    int nve = 0;
    for (int i = 0; i < vocab_size; i++) if (tok->id_to_bpe[i]) nve++;
    tok->vocab_map_cap = next_pow2(nve * 2 + 1);
    tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
    if (!tok->vocab_map) { ds_tokenizer_free(tok); free(vjs); free(json); return NULL; }
    for (int i = 0; i < vocab_size; i++) {
        if (tok->id_to_bpe[i]) map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
    }

    /* Load added_tokens from tokenizer.json
     * Added tokens may have ids beyond vocab_size (e.g. 128815 for <image>,
     * 128816-128819 for <|ref|>/<|det|> etc.), so we need to expand
     * id_to_bpe/id_to_text arrays if any added token id exceeds vocab_size. */
    {
        /* Pre-scan: find max id in added_tokens to determine if expansion needed */
        int64_t max_added_id = -1;
        {
            const char *ps = json;
            while (*ps) {
                const char *pf = strstr(ps, "\"added_tokens\"");
                if (!pf) break;
                const char *pb = pf - 1;
                while (pb > json && (*pb==' '||*pb=='\t'||*pb=='\n'||*pb=='\r')) pb--;
                if (*pb == '{' || *pb == ',') {
                    const char *pa = pf + strlen("\"added_tokens\"");
                    skip_ws(&pa);
                    if (*pa == ':') { pa++; skip_ws(&pa);
                        if (*pa == '[') { pa++;
                            while (*pa && *pa != ']') {
                                skip_ws(&pa); if (*pa==']') break; if (*pa==','){pa++;continue;}
                                if (*pa!='{'){pa++;continue;} pa++;
                                int64_t ps_id = -1;
                                while (*pa && *pa != '}') {
                                    skip_ws(&pa); if (*pa=='}') break; if (*pa==','){pa++;continue;}
                                    char psk[256];
                                    if (parse_json_string(&pa, psk, sizeof(psk)) != 0) { pa++; continue; }
                                    skip_ws(&pa); if (*pa != ':') break; pa++; skip_ws(&pa);
                                    if (strcmp(psk, "id") == 0) ps_id = parse_json_int(&pa);
                                    else { skip_json_value(&pa); }
                                }
                                if (*pa == '}') pa++;
                                if (ps_id > max_added_id) max_added_id = ps_id;
                            }
                        }
                    }
                }
                ps = pf + 1;
            }
        }

        /* Expand arrays if needed */
        if (max_added_id >= 0 && max_added_id >= vocab_size) {
            int new_size = (int)max_added_id + 1;
            char **new_bpe = (char **)realloc(tok->id_to_bpe, (size_t)new_size * sizeof(char *));
            char **new_text = (char **)realloc(tok->id_to_text, (size_t)new_size * sizeof(char *));
            if (new_bpe && new_text) {
                memset(new_bpe + vocab_size, 0, (size_t)(new_size - vocab_size) * sizeof(char *));
                memset(new_text + vocab_size, 0, (size_t)(new_size - vocab_size) * sizeof(char *));
                tok->id_to_bpe = new_bpe;
                tok->id_to_text = new_text;
                tok->vocab_size = new_size;
                vocab_size = new_size;  /* Update local variable for aid < vocab_size check below */
            }
        }

        int n_added = 0;
        const char *at = json;
        while (*at) {
            const char *found = strstr(at, "\"added_tokens\"");
            if (!found) break;
            const char *before = found - 1;
            while (before > json && (*before==' '||*before=='\t'||*before=='\n'||*before=='\r')) before--;
            if (*before == '{' || *before == ',') {
                at = found + strlen("\"added_tokens\"");
                skip_ws(&at);
                if (*at == ':') { at++; skip_ws(&at);
                    if (*at == '[') { at++;
                        while (*at && *at != ']') {
                            skip_ws(&at); if (*at==']') break; if (*at==','){at++;continue;}
                            if (*at!='{'){at++;continue;} at++;
                            char ac[4096]={0}; int64_t aid=-1;
                            while(*at&&*at!='}'){
                                skip_ws(&at); if(*at=='}')break; if(*at==','){at++;continue;}
                                char ak[256];
                                if(parse_json_string(&at,ak,sizeof(ak))!=0){at++;continue;}
                                skip_ws(&at); if(*at!=':')break; at++; skip_ws(&at);
                                if(strcmp(ak,"id")==0) aid=parse_json_int(&at);
                                else if(strcmp(ak,"content")==0) parse_json_string(&at,ac,sizeof(ac));
                                else { skip_json_value(&at); }
                            }
                            if(*at=='}')at++;
                            if(aid>=0&&aid<vocab_size&&ac[0]){
                                free(tok->id_to_bpe[aid]);free(tok->id_to_text[aid]);
                                tok->id_to_bpe[aid]=strdup(ac); tok->id_to_text[aid]=strdup(ac);
                                map_insert((str_int_entry_t*)tok->vocab_map,tok->vocab_map_cap,ac,(int)aid);
                                n_added++;
                            }
                        }
                    }
                }
            }
            at = found + 1;
        }
        if (n_added > 0) {
            /* Rebuild vocab_map with expanded vocab_size to include added_tokens */
            free(tok->vocab_map);
            int nve = 0;
            for (int i = 0; i < tok->vocab_size; i++) if (tok->id_to_bpe[i]) nve++;
            tok->vocab_map_cap = next_pow2(nve * 2 + 1);
            tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
            if (tok->vocab_map) {
                for (int i = 0; i < tok->vocab_size; i++) {
                    if (tok->id_to_bpe[i])
                        map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
                }
            }
            if (ds_verbose >= 1)
                fprintf(stderr, "Tokenizer: %d added_tokens from tokenizer.json (vocab expanded to %d)\n", n_added, tok->vocab_size);
        }
    }

    /* Load merges from model.merges
     * vocab_end points just past the vocab object's closing }.
     * Continue scanning within the model object for "merges" key. */
    {
        const char *mp2 = vocab_end;
        while(*mp2&&*mp2!='}'){
            skip_ws(&mp2); if(*mp2==','){mp2++;continue;} if(*mp2=='}')break;
            char mk[256]; if(parse_json_string(&mp2,mk,sizeof(mk))!=0){mp2++;continue;}
            skip_ws(&mp2); if(*mp2!=':')break; mp2++; skip_ws(&mp2);
            if(strcmp(mk,"merges")==0&&*mp2=='['){
                    mp2++; int mi=0;
                    while(*mp2&&*mp2!=']'){
                        skip_ws(&mp2); if(*mp2==']')break; if(*mp2==','){mp2++;continue;}
                        /* Support two merge formats:
                         *   Simple:  ["a b", "c d", ...]          (string per merge pair)
                         *   Nested:  [["a","b"], ["c","d"], ...]  (array per merge pair) */
                        if(*mp2=='"'){
                            /* Simple format: "a b" */
                            char ms[4096];
                            if(parse_json_string(&mp2,ms,sizeof(ms))==0){
                                char*space=strchr(ms,' ');
                                if(space){
                                    *space='\0';
                                    char mk2[8192]; snprintf(mk2,sizeof(mk2),"%s %s",ms,space+1);
                                    if(!tok->merge_map){tok->merge_map_cap=262144;tok->merge_map=calloc((size_t)tok->merge_map_cap,sizeof(str_int_entry_t));}
                                    if(tok->merge_map&&mi<tok->merge_map_cap){
                                        char *mk2_dup = strdup(mk2);
                                        if(mk2_dup) map_insert((str_int_entry_t*)tok->merge_map,tok->merge_map_cap,mk2_dup,mi);
                                    }
                                    mi++;
                                }
                            }
                        } else if(*mp2=='['){
                            /* Nested format: ["a","b"] */
                            mp2++; /* skip [ */
                            char a_str[4096]="", b_str[4096]="";
                            int got_a=0, got_b=0;
                            while(*mp2&&*mp2!=']'){
                                skip_ws(&mp2); if(*mp2==']')break; if(*mp2==','){mp2++;continue;}
                                if(*mp2=='"'){
                                    if(!got_a){
                                        if(parse_json_string(&mp2,a_str,sizeof(a_str))==0) got_a=1;
                                    } else if(!got_b){
                                        if(parse_json_string(&mp2,b_str,sizeof(b_str))==0) got_b=1;
                                    } else {
                                        char t[4096]; parse_json_string(&mp2,t,sizeof(t));
                                    }
                                } else mp2++;
                            }
                            if(*mp2==']') mp2++;
                            if(got_a&&got_b){
                                char mk2[8192]; snprintf(mk2,sizeof(mk2),"%s %s",a_str,b_str);
                                if(!tok->merge_map){tok->merge_map_cap=next_pow2(mi*4+262144);tok->merge_map=calloc((size_t)tok->merge_map_cap,sizeof(str_int_entry_t));}
                                if(tok->merge_map&&mi<tok->merge_map_cap){
                                    char *mk2_dup = strdup(mk2);
                                    if(mk2_dup) map_insert((str_int_entry_t*)tok->merge_map,tok->merge_map_cap,mk2_dup,mi);
                                }
                                mi++;
                            }
                        } else mp2++;
                    }
                    if(mi>0&&ds_verbose>=1) fprintf(stderr,"Tokenizer: %d merges from tokenizer.json\n",mi);
                    break;
                }
                /* Skip non-merges value */
                skip_json_value(&mp2);
            }
    }

    free(vjs);
    free(json);
    if (ds_verbose >= 1)
        fprintf(stderr, "Tokenizer loaded from tokenizer.json: %d vocab, %s format\n",
                tok->vocab_size, is_deepseek_tok ? "DeepSeek" : "GPT-2");
    return tok;
}
