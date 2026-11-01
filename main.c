/*
 * main.c - CLI entry point for DeepSeek-OCR
 * main.c — DeepSeek-OCR 命令行入口
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【程序流程】
 * ─────────────────────────────────────────────────────────────────────
 *   1. 解析命令行参数 (-d 模型目录, -i 图像路径, -t 线程数, 等)
 *   2. 初始化线程池 (默认使用所有CPU核心)
 *   3. 加载模型 (ds_load → safetensors mmap + 权重加载 + 缓冲区分配)
 *   4. 应用推理设置 (温度、重复惩罚、ngram阻断、INT4量化等)
 *   5. 执行OCR识别 (ds_recognize → 图像→编码→预填充→解码→文字)
 *   6. 输出结果 + 性能统计 + 逐层Profiler
 *
 * 【用法示例】
 *   基础:   ./ds_ocr -d model_dir -i image.png
 *   性能:   ./ds_ocr -d model_dir -i image.png --profile
 *   量化:   ./ds_ocr -d model_dir -i image.png --int4
 *   调参:   ./ds_ocr -d model_dir -i image.png --rp 1.1 --ngram 35
 *   调试:   ./ds_ocr -d model_dir -i image.png --debug
 *   安静:   ./ds_ocr -d model_dir -i image.png --silent
 *   交互:   ./ds_ocr -d model_dir --stream    (输入 @file.png 回车识别, q 退出)
 *
 * 【--stream 交互模式】
 *   启动后进入 REPL 循环, 模型只加载一次, 可连续识别多个文件:
 *     ds_ocr> @photo.png        ← 回车即开始识别图片
 *     ds_ocr> @document.pdf     ← PDF 逐页识别, 页间输出分隔标记
 *     ds_ocr> q                 ← 退出
 *   识别结果走 stdout, 提示符/状态/进度/页分隔走 stderr, 便于管道重定向。
 *   解码期间 stderr 每 32 token 打一个进度点（只计数不透出文本——V3/V4
 *   原始流含幻觉前缀, 正文必须等后处理）, 识别完输出 "✓ N tok, X.Xs"。
 *   --silent 在此模式仅关闭性能统计; 提示符/进度等交互反馈保留在 stderr,
 *   否则大图识别数十秒毫无动静, 看起来像卡死。
 *
 * 【与 qwen-asr 的对应】
 * 两者共享完全相同的 main.c 结构：参数解析 → 加载模型 → 推理 → 输出
 * 区别在于模型类型：ds-ocr 加载 DeepSeek-OCR 模型，qwen-asr 加载 Qwen3-ASR 模型
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Usage: ds_ocr -d <model_dir> -i <input.png> [options]
 */

#include "ds_ocr.h"
#include "ds_kernels.h"
#include "ds_pdf.h"             /* PDF 逐页渲染（--stream 模式） */
#include "ds_platform_ocr.h"    /* macOS Vision OCR 后端（不使用模型时的备选方案） */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>            /* strcasecmp */
#include <sys/types.h>          /* ssize_t (getline) */

/* stream_token — Token流式输出回调
 *
 * 每当解码器生成一个新token并解码为文本后，此函数被调用。
 * 效果: 将解码出的文本片段(piece)立即打印到stdout，实现"边生成边输出"。
 * 类似于 ChatGPT 的逐字显示效果。
 *
 * 为什么用 fflush(stdout)?
 *   stdout 默认是行缓冲的（遇到\n才刷新），但OCR文本可能很长没有换行，
 *   fflush确保每个token都立即显示，不等待缓冲区满。
 *
 * userdata: 未使用（(void)userdata 消除编译器警告）
 */
static void stream_token(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);  /* 输出文本片段到标准输出 */
    fflush(stdout);        /* 立即刷新，确保实时显示 */
}

/* ═══════════════════════════════════════════════════════════════════
 * --stream 解码进度反馈
 *
 * V3/V4 的原始 token 流含幻觉前缀/布局标签，正文必须等后处理，
 * 所以不能逐 token 打印内容——但"毫无动静"会让长时间识别看起来像卡死
 * （大图识别可长达数十秒）。进度回调只计数打点，不透出任何文本。
 * ═══════════════════════════════════════════════════════════════════ */

#define DS_STREAM_PROGRESS_EVERY 32
#define DS_STREAM_PROGRESS_EVERY_SLOW 128   /* 长输出降频: >512 token 后每 128 一个点 */
#define DS_STREAM_PROGRESS_SLOW_AFTER 512

typedef struct {
    int count;    /* 本次识别已生成的 token 数 */
    int started;  /* 1 = 已打印 "解码" 前缀（避免空转行） */
    int retried;  /* 1 = 本次识别含退化重试（✓ 行标注） */
} ds_stream_progress_t;

static ds_stream_progress_t g_stream_progress;

/* stream_progress_breakline — 进度点在行中时先结束该行（重试提示前调用） */
static void stream_progress_breakline(void) {
    if (g_stream_progress.started) {
        fputc('\n', stderr);
        g_stream_progress.started = 0;
    }
}

/* stream_progress_cb — 每 N 个 token 在 stderr 打一个进度点（内容忽略） */
static void stream_progress_cb(const char *piece, void *userdata) {
    (void)piece;
    ds_stream_progress_t *p = (ds_stream_progress_t *)userdata;
    p->count++;
    if (!p->started) {
        fputs("  解码", stderr);
        p->started = 1;
    }
    int every = p->count <= DS_STREAM_PROGRESS_SLOW_AFTER
                    ? DS_STREAM_PROGRESS_EVERY : DS_STREAM_PROGRESS_EVERY_SLOW;
    if (p->count % every == 0) {
        fprintf(stderr, " ·%d", p->count);
        fflush(stderr);
    }
}

/* stream_progress_begin — 识别前挂上进度回调（已有正文流式回调时不挂） */
static void stream_progress_begin(ds_ctx_t *ctx) {
    g_stream_progress.count = 0;
    g_stream_progress.started = 0;
    g_stream_progress.retried = 0;
    if (!ctx->token_cb)
        ds_set_token_callback(ctx, stream_progress_cb, &g_stream_progress);
}

/* stream_progress_end — 摘除进度回调并输出完成信号 */
static void stream_progress_end(ds_ctx_t *ctx) {
    if (ctx->token_cb == stream_progress_cb)
        ds_set_token_callback(ctx, NULL, NULL);
    stream_progress_breakline();
    if (ctx->last_output_loopy)
        fprintf(stderr, "  ⚠ 检测到内容循环，已在 %d tok 提前终止"
                        "（DS_NO_EARLY_STOP=1 关闭早停）\n",
                ctx->perf_text_tokens);
    fprintf(stderr, "  ✓ %d tok, %.1fs%s\n",
            ctx->perf_text_tokens, ctx->perf_total_ms / 1000.0,
            g_stream_progress.retried ? "（含 1 次退化重试）" : "");
}

/* ═══════════════════════════════════════════════════════════════════
 * 识别辅助函数 — 单次模式与 --stream 交互模式共用
 * ═══════════════════════════════════════════════════════════════════ */

/* model_version_name — 模型版本的可读名称（交互模式横幅用） */
static const char *model_version_name(const ds_ctx_t *ctx) {
    switch (ctx->config.model_version) {
        case 1: return "DeepSeek-OCR";
        case 2: return "DeepSeek-OCR-2";
        case 3: return "Unlimited-OCR";
        case 4: return "Unlimited-OCR-2";
        default: return "Unknown";
    }
}

/* recognize_file_with_retry — 从图像文件识别 + V4 退化重试
 *
 * V4 退化翻转自愈: 解码近并列 argmax 偶发翻转（运行间内存布局差异），
 * 症状为畸形标签行。检测到即整体重跑一次（含编码），命中概率 p → p²。
 * 例外: 首次尝试已打满 token 上限 = 内容循环而非翻转（模型对循环输出
 * "自信"，与内存布局无关），重试只会再烧一份上限 token——跳过。
 * 交互模式下重试提示前先结束进度行，并为重试复位进度计数。 */
static int stream_attempt_hit_cap(const ds_ctx_t *ctx) {
    return ctx->perf_text_tokens >= ctx->max_new_tokens;
}

static char *recognize_file_with_retry(ds_ctx_t *ctx, const char *path) {
    char *text = ds_recognize(ctx, path);
    if (text && ctx->config.model_version == 4 && ctx->last_output_degenerate) {
        if (stream_attempt_hit_cap(ctx)) {
            stream_progress_breakline();
            fprintf(stderr, "首次识别已打满 %d token 上限（疑似内容循环），跳过重试\n",
                    ctx->max_new_tokens);
            return text;
        }
        stream_progress_breakline();
        fprintf(stderr, "检测到退化输出片段，自动重试一次…\n");
        g_stream_progress.count = 0;
        g_stream_progress.started = 0;
        g_stream_progress.retried = 1;
        free(text);
        text = ds_recognize(ctx, path);
    }
    return text;
}

/* recognize_pixels_with_retry — 从RGB像素识别 + V4 退化重试（PDF 页用）
 *
 * 重试触发条件: last_output_degenerate（畸形标签流）或 last_output_loopy
 * （内容循环早停）。31页国标实测（V4）: 13页触发早停、平均仅 ~350 tok
 * 就被截断，页尾整段内容丢失（如 B.2.3.7.2 预期结果整段缺失）；而单独
 * 重跑同页 6/6 全部正常（8字滑窗命中率 96.8%）——循环是长会话下的解码
 * 边界效应，重试即自愈。早停页通常 5-9s，重试代价可控。
 * 打满 -n 上限的页不重试: 上限本身就是止损手段，重试大概率再次打满。 */
static char *recognize_pixels_with_retry(ds_ctx_t *ctx, const unsigned char *pixels,
                                          int width, int height, int channels) {
    char *text = ds_recognize_image(ctx, pixels, width, height, channels);
    if (text && ctx->config.model_version == 4 &&
        (ctx->last_output_degenerate || ctx->last_output_loopy)) {
        if (stream_attempt_hit_cap(ctx)) {
            stream_progress_breakline();
            fprintf(stderr, "首次识别已打满 %d token 上限（疑似内容循环），跳过重试\n",
                    ctx->max_new_tokens);
            return text;
        }
        stream_progress_breakline();
        fprintf(stderr, ctx->last_output_loopy
                ? "检测到内容循环早停（疑似截断），自动重试一次…\n"
                : "检测到退化输出片段，自动重试一次…\n");
        g_stream_progress.count = 0;
        g_stream_progress.started = 0;
        g_stream_progress.retried = 1;
        free(text);
        text = ds_recognize_image(ctx, pixels, width, height, channels);
    }
    return text;
}

/* text_looks_loopy — 疑似生成循环检测: 连续 ≥3 行完全相同且非空
 *
 * V4 在公式/表格密集页偶发内容循环（如反复输出同一公式），1995 token
 * 只产出几行重复文本。引擎的 last_output_degenerate 只覆盖畸形标签行，
 * 这里作为交互模式的补充提示——只告警不改动输出，建议 --ngram 抑制。 */
static int text_looks_loopy(const char *text) {
    if (!text) return 0;
    const char *prev = NULL;
    size_t prev_len = 0;
    int run = 1;
    const char *s = text;
    while (*s) {
        const char *eol = strchr(s, '\n');
        size_t len = eol ? (size_t)(eol - s) : strlen(s);
        if (len > 0) {
            if (prev && prev_len == len && memcmp(prev, s, len) == 0) {
                if (++run >= 3) return 1;
            } else {
                run = 1;
            }
            prev = s;
            prev_len = len;
        }
        s = eol ? eol + 1 : s + strlen(s);
    }
    return 0;
}

/* ── 循环残留清理（r4）────────────────────────────────────────────
 * 早停要求"整个检测窗周期化"，循环只持续 2-3 圈就自然结束/截断的页逃过
 * 检测，会在输出尾部留下 2-4 圈重复垃圾（如 `附录/（资料库）` 交替数组、
 * `$ x_{1}=2 $`×4）。对已判定循环的页，把周期尾环折叠到只剩一圈。 */

/* norm_line_eq — 两行归一化（去数字/空白/[]/,/.）后比较是否同形 */
static int norm_line_eq(const char *a, size_t alen, const char *b, size_t blen) {
    size_t i = 0, j = 0;
    for (;;) {
        while (i < alen && ((a[i] >= '0' && a[i] <= '9') || a[i] == ' ' ||
               a[i] == '\t' || a[i] == '[' || a[i] == ']' || a[i] == ',' || a[i] == '.'))
            i++;
        while (j < blen && ((b[j] >= '0' && b[j] <= '9') || b[j] == ' ' ||
               b[j] == '\t' || b[j] == '[' || b[j] == ']' || b[j] == ',' || b[j] == '.'))
            j++;
        if (i >= alen || j >= blen) break;
        if (a[i] != b[j]) return 0;
        i++; j++;
    }
    return i >= alen && j >= blen;
}

/* trim_loopy_tail_p — 迭代折叠输出尾部的周期环
 * 每轮: **最大周期后缀**——对每个 p，从尾部向下找连续配对 (j, j+p) 的
 * 最长周期区（末圈允许不完整，循环可能被截断），删除重复部分保留一个
 * 周期；取删行数最多的 p。所有索引有闭式边界，无越界。
 * min_periods: 折叠所需的最少周期数。未标记页用 3（黄金标签最大连重复
 * =1，两倍安全边际；V4 的"标题块+全文块"合法双写、文档重复链接行均
 * ≤2 连重复不受影响）；已判定循环页用 2（该页已确认是垃圾，激进清理）。 */
static size_t trim_loopy_tail_p(char *text, size_t len, int min_periods) {
    for (int pass = 0; pass < 16; pass++) {
        const char *starts[64];
        size_t lens[64];
        int n = 0;
        size_t pos = 0;
        while (pos < len && n < 64) {
            const char *s = text + pos;
            char *nl = memchr(s, '\n', len - pos);
            size_t l = nl ? (size_t)(nl - s) : len - pos;
            if (l > 0) { starts[n] = s; lens[n] = l; n++; }
            pos += l + (nl ? 1 : 0);
        }
        if (n < 2) break;

        int best_p = 0, best_drop = 0;
        for (int p = 1; p <= n / 2; p++) {
            int j = n - p - 1;
            int s = n - p;   /* 无任何配对时的空周期区 */
            while (j >= 0 && norm_line_eq(starts[j], lens[j], starts[j + p], lens[j + p])) {
                s = j;
                j--;
            }
            int wlen = n - s;
            if (wlen >= min_periods * p && wlen - p > best_drop) {
                best_drop = wlen - p;
                best_p = p;
            }
        }
        if (best_p == 0) break;

        int drop = best_drop;
        const char *cut_start = starts[n - drop];
        size_t new_len = (size_t)(cut_start - text);
        while (new_len > 0 && text[new_len - 1] == '\n') new_len--;
        text[new_len] = '\0';
        len = new_len;
    }
    return len;
}

/* ── 内容级循环清理（r5）────────────────────────────────────────
 * 早停/尾环折叠只覆盖"整窗周期化"的循环。附录B类**内容级循环**——同一
 * 主体带 a)/b)/c) 枚举前缀或"检查是否"变体交替输出，行间不相邻、整窗
 * 不周期——两类检测全部漏网（31页国标实测: 附录B区输出命中率仅30.2%，
 * 垃圾行如"b)具备…备份机制…"×4、"g)侵犯他人其他合法权益。"×6）。
 * 防线: 主体归一化 + 短滑窗重复计数。合法结构重复（"对服务提供者的
 * 要求如下。"×18、"附录"×13）跨节分布，任何12行窗内≤2次出现，不受
 * 影响；循环垃圾在数行内扎堆（GT计数=0，窗内4-6次），第3次起丢弃。
 * 仅作用于已判定循环的页（激进档），未标记页仍走保守 trim(3)。 */

/* is_cjk_punct3 — 3字节CJK标点: 全角标点块（，．：；？！FF00-FFEF）
 * 或CJK符号块（、。〈〉《》3000-303F）*/
static int is_cjk_punct3(const unsigned char *p) {
    if (p[0] == 0xEF && p[1] == 0xBC) return 1;
    if (p[0] == 0xE3 && p[1] == 0x80) return 1;
    return 0;
}

/* line_subject — 行主体归一化（仅用于重复匹配，不改动原行）
 * 剥离枚举前缀（"a)" "b）" "1." "1、" "(1)" "（1）"）、引导语"检查
 * 是否"、空白与全部标点（全角/半角），字母小写化。输出为匹配键。 */
static size_t line_subject(const char *s, size_t len, char *out, size_t outsz) {
    const unsigned char *p = (const unsigned char *)s;
    size_t i = 0, w = 0;
    while (i < len && (p[i] == ' ' || p[i] == '\t')) i++;
    /* 前缀1: 1-3位字母数字 + ) ） . ， 、 */
    {
        size_t j = i;
        while (j < len && j - i < 3 &&
               ((p[j] >= 'a' && p[j] <= 'z') || (p[j] >= 'A' && p[j] <= 'Z') ||
                (p[j] >= '0' && p[j] <= '9'))) j++;
        if (j > i && j < len) {
            unsigned char c = p[j];
            int sep = (c == ')' || c == '.' || c == ',');
            if (!sep && c == 0xEF && j + 2 < len && is_cjk_punct3(p + j)) sep = 1;
            if (sep) i = j + 1;
        }
    }
    /* 前缀2: (X) / （X），X为≤4位字母数字 */
    {
        size_t j = i;
        int open = (p[j] == '(');
        if (!open && p[j] == 0xEF && j + 2 < len && (unsigned char)p[j+1] == 0xBC &&
            (unsigned char)p[j+2] == 0x88) open = 2;   /* （ */
        if (open) {
            size_t k = j + (open == 2 ? 3 : 1), d = 0;
            while (k < len && d < 4 &&
                   ((p[k] >= 'a' && p[k] <= 'z') || (p[k] >= 'A' && p[k] <= 'Z') ||
                    (p[k] >= '0' && p[k] <= '9'))) { k++; d++; }
            if (d > 0) {
                int close = (k < len && p[k] == ')');
                if (!close && k + 2 < len && p[k] == 0xEF &&
                    (unsigned char)p[k+1] == 0xBC && (unsigned char)p[k+2] == 0x89) close = 2; /* ）*/
                if (close) i = k + (close == 2 ? 3 : 1);
            }
        }
    }
    /* 引导语"检查是否"（B.2.3.x 测评方法的循环变体开头） */
    static const unsigned char lead[] = {0xE6,0xA3,0x80, 0xE6,0x9F,0xA5,
                                         0xE6,0x98,0xAF, 0xE5,0x90,0xA6};
    if (len - i >= sizeof(lead) && memcmp(p + i, lead, sizeof(lead)) == 0) i += sizeof(lead);
    /* 主体: 保留 ASCII字母数字（小写化）+ 非标点CJK字符（整字符） */
    for (; i < len && w + 4 < outsz; i++) {
        unsigned char c = p[i];
        if (c >= 'A' && c <= 'Z') { out[w++] = (char)(c - 'A' + 'a'); continue; }
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) { out[w++] = (char)c; continue; }
        if (c < 0x80) continue;                                /* 空白/ASCII标点 → 删 */
        if (is_cjk_punct3(p + i) && i + 2 < len) { i += 2; continue; }
        out[w++] = (char)c;
        while (i + 1 < len && (p[i+1] & 0xC0) == 0x80) out[w++] = (char)p[++i];
    }
    out[w] = '\0';
    return w;
}

#define SUBJ_MIN_BYTES 12   /* 主体≥4个汉字才参与判定（"附录"等短标题永不折叠） */
#define SUBJ_WINDOW 12      /* 滑窗行数: 合法重复跨节分布(窗内≤2)，循环扎堆(窗内4-6) */

/* collapse_dup_subject — 内容级循环折叠: 12行滑窗内同主体第3次出现丢弃
 * 与 trim_loopy_tail_p(整窗周期尾环)互补: 处理 a)/b)/c) 前缀变化、
 * "检查是否"变体等非连续同主体循环。仅用于已判定循环的页。
 * 正确性锚点: 黄金真值中窗内同主体出现≥3次的合法行 = 0。 */
static void collapse_dup_subject(char *text) {
    if (!text) return;
    size_t len = strlen(text);
    char *tmp = (char *)malloc(len + 2);
    if (!tmp) return;
    char subj[512];
    static char win[SUBJ_WINDOW][512];
    static size_t win_len[SUBJ_WINDOW];
    int wn = 0;
    size_t w = 0, pos = 0;
    while (pos < len) {
        char *nl = memchr(text + pos, '\n', len - pos);
        size_t e = nl ? (size_t)(nl - (text + pos)) : len - pos;
        size_t sl = line_subject(text + pos, e, subj, sizeof(subj));
        int dup = 0;
        if (sl >= SUBJ_MIN_BYTES) {
            int hits = 0;
            for (int k = 0; k < wn; k++)
                if (win_len[k] == sl && memcmp(win[k], subj, sl) == 0) hits++;
            if (hits >= 2) dup = 1;                 /* 窗内第3次出现 → 丢弃 */
        }
        if (!dup) {
            memcpy(tmp + w, text + pos, e);
            w += e;
            if (nl) tmp[w++] = '\n';
            if (sl > 0) {                           /* 空行/短行不入窗 */
                size_t keep = sl < sizeof(win[0]) ? sl : sizeof(win[0]) - 1;
                if (wn == SUBJ_WINDOW) {
                    memmove(win, win + 1, sizeof(win[0]) * (SUBJ_WINDOW - 1));
                    memmove(win_len, win_len + 1, sizeof(win_len[0]) * (SUBJ_WINDOW - 1));
                    wn--;
                }
                memcpy(win[wn], subj, keep);
                win_len[wn] = keep;
                wn++;
            }
        }
        if (!nl) break;
        pos += e + 1;
    }
    while (w > 0 && tmp[w - 1] == '\n') w--;
    memcpy(text, tmp, w);
    text[w] = '\0';
    free(tmp);
}

/* defrase_text — 行内句级去重: ≥3个连续相同段折叠为1段
 * 处理 "检查X，检查X，检查X…" 行内循环（附录B实测"确保系统安全"×3）。
 * 实现: 按分隔符（, ; ，；、。）整行分段 → 连续同形段（原始字节相等）
 * run≥3 时保留1段 → 重组。段长 6-256 字节才参与（≥2汉字，<6字节的
 * 短段可能是修辞排比，保守跳过）。仅用于已判定循环的页。 */
static void defrase_text(char *text) {
    if (!text) return;
    size_t len = strlen(text);
    char *tmp = (char *)malloc(len + 2);
    if (!tmp) return;
    size_t w = 0, pos = 0;
    while (pos < len) {
        char *nl = memchr(text + pos, '\n', len - pos);
        size_t lend = nl ? (size_t)(nl - (text + pos)) : len - pos;
        const char *line = text + pos;
        /* 按分隔符分段，记录每段 (偏移, 长度, 分隔符宽) */
        size_t soff[128], sseg[128], ssep[128];
        int ns = 0;
        size_t i = 0;
        while (i < lend && ns < 128) {
            size_t start = i, sep_l = 0;
            while (i < lend) {
                unsigned char c = (unsigned char)line[i];
                if (c == ',' || c == ';') { sep_l = 1; break; }
                if (c == 0xEF && i + 2 < lend && (unsigned char)line[i+1] == 0xBC &&
                    ((unsigned char)line[i+2] == 0x8C ||   /* ， */
                     (unsigned char)line[i+2] == 0x9B)) { sep_l = 3; break; }  /* ； */
                if (c == 0xE3 && i + 2 < lend && (unsigned char)line[i+1] == 0x80 &&
                    ((unsigned char)line[i+2] == 0x81 ||   /* 、 */
                     (unsigned char)line[i+2] == 0x82)) { sep_l = 3; break; }  /* 。 */
                if (c < 0x80) i += 1;
                else if ((c & 0xE0) == 0xC0) i += 2;
                else if ((c & 0xF0) == 0xE0) i += 3;
                else i += 4;
            }
            soff[ns] = start; sseg[ns] = i - start; ssep[ns] = sep_l; ns++;
            if (sep_l == 0) break;   /* 行尾无分隔符 */
            i += sep_l;
        }
        /* 连续同形段 run≥3 → 折叠为1段（段长6-256字节才参与判定） */
        int a = 0;
        while (a < ns) {
            int b = a + 1;
            if (sseg[a] >= 6 && sseg[a] <= 256) {
                while (b < ns && sseg[b] == sseg[a] &&
                       memcmp(line + soff[b], line + soff[a], sseg[a]) == 0)
                    b++;
            }
            memcpy(tmp + w, line + soff[a], sseg[a]); w += sseg[a];
            memcpy(tmp + w, line + soff[a] + sseg[a], ssep[a]); w += ssep[a];
            a = b;
        }
        /* 段数组满（>128段的病态行）: 剩余部分原样保留，不丢内容 */
        if (ns >= 128 && i < lend) {
            memcpy(tmp + w, line + i, lend - i);
            w += lend - i;
        }
        if (!nl) break;
        tmp[w++] = '\n';
        pos += lend + 1;
    }
    while (w > 0 && tmp[w - 1] == '\n') w--;
    memcpy(text, tmp, w);
    text[w] = '\0';
    free(tmp);
}

/* collapse_dup_lines — 折叠连续同形行到一行（已判定循环页的激进清理） */
static void collapse_dup_lines(char *text) {
    if (!text) return;
    size_t len = strlen(text);
    char *tmp = (char *)malloc(len + 2);
    if (!tmp) return;
    size_t w = 0, pos = 0;
    const char *prev = NULL;
    size_t prev_len = 0;
    while (pos < len) {
        char *nl = memchr(text + pos, '\n', len - pos);
        size_t e = nl ? (size_t)(nl - (text + pos)) : len - pos;
        int dup = (prev && norm_line_eq(prev, prev_len, text + pos, e));
        if (!dup) {
            memcpy(tmp + w, text + pos, e);
            w += e;
            if (nl) tmp[w++] = '\n';
            prev = text + pos;
            prev_len = e;
        }
        if (!nl) break;
        pos += e + 1;
    }
    while (w > 0 && tmp[w - 1] == '\n') w--;
    memcpy(text, tmp, w);
    text[w] = '\0';
    free(tmp);
}

/* cleanup_loopy_output — 分层清理: 未标记页保守（≥3 周期），已判定循环页
 * 激进（连重行折叠 + ≥2 周期尾环折叠）。正常页的合法重复（Hello×2、
 * Back to TOC×2）只出现在未标记页，不受影响。 */
static void cleanup_loopy_output(const ds_ctx_t *ctx, char *text) {
    if (!text) return;
    int flagged = ctx->last_output_loopy || ctx->last_output_degenerate;
    if (flagged) {
        defrase_text(text);            /* 行内: "检查X，检查X，检查X" → "检查X" */
        collapse_dup_lines(text);      /* 行间: 连续同形行折叠 */
        collapse_dup_subject(text);    /* 内容级: 滑窗内同主体第3次出现丢弃（非连续） */
        trim_loopy_tail_p(text, strlen(text), 2);
    } else {
        trim_loopy_tail_p(text, strlen(text), 3);
    }
}

/* warn_loopy_output — 疑似循环时的可操作 stderr 提示
 * 实测（31 页标准文档 A/B）: --ngram 35 / --rp 1.1 均无法抑制这类循环
 * （V4 循环时原始流带逐圈抖动的坐标标签，非精确重复；且模型对重复内容
 * 输出"自信"，logit 差距大于温和惩罚）。循环会持续到 token 上限，唯一
 * 有效的止损是 -n 上限——正常页多在 ~1000 token，循环页会一路打满。 */
static void warn_loopy_output(const ds_ctx_t *ctx, const char *text, const char *label) {
    if (text_looks_loopy(text)) {
        if (ctx->max_new_tokens <= 1536) {
            fprintf(stderr,
                    "  ⚠ %s输出疑似重复循环，已按 -n %d 上限截断。"
                    "如需该页完整内容，可用更高 -n 单独重跑本页。\n",
                    label, ctx->max_new_tokens);
        } else {
            fprintf(stderr,
                    "  ⚠ %s输出疑似重复循环（模型固有退化，已按原样输出，"
                    "将持续到 token 上限）。\n"
                    "  批处理建议: 加 -n 1536 止损（正常页多在 ~1000 tok 内，"
                    "被截页可单独用更高 -n 重跑）。\n",
                    label);
        }
        /* 实测（31 页国标文档四模型 A/B，8 字滑窗内容召回）:
         *   V2 DeepSeek-OCR-2   68.2%（附录B 表格区 76.5%）零重试 405s ← 文档类最优
         *   V3 Unlimited-OCR    59.4%（表格区 49.6%）零重试 544s
         *   V4 Unlimited-OCR-2  38.8%（表格区 20.7%）重试 10 次 327s
         * V4/DASH 对文档类页面退化倾向是模型属性（官方 vLLM 亦内置
         * page_count_guard 护栏）。文档类批量场景建议用 V2。 */
        if (ctx->config.model_version == 4)
            fprintf(stderr,
                    "  建议: V4 对文档类页面退化率高（实测召回 38.8%% vs V2 68.2%%），"
                    "建议改用 -d models/DeepSeek-OCR-2（V2）重跑本文档。\n");
    }
}

/* print_ocr_result — 输出一次识别的最终结果到 stdout
 *
 * V3/V4 或静默模式: 一次性打印完整结果（V3/V4 可能带幻觉前缀需后处理，
 * 无法逐token输出，回调已被禁用，结果在这里统一打印）。
 * V1/V2 非静默: 结果已通过stream_token逐token打印，此处只补换行。
 */
static void print_ocr_result(const ds_ctx_t *ctx, const char *text, int verbosity) {
    if (verbosity == 0 || ctx->config.model_version >= 3) {
        printf("%s\n", text);
    } else {
        printf("\n");
    }
}

/* print_perf_summary — 输出性能统计（--profile 时附加逐层Profiler） */
static void print_perf_summary(const ds_ctx_t *ctx, int profile, int verbosity) {
    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_decode_ms > 0) {
            tokens_per_sec = (1000.0 * ctx->perf_text_tokens) / ctx->perf_decode_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (%.2f tok/s decode)\n"
                "  Encoding: %.0f ms (SAM %.0f + Encoder %.0f) | Prefill: %.0f ms | Decode: %.0f ms\n",
                ctx->perf_total_ms, ctx->perf_text_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_sam_ms, ctx->perf_encoder_ms,
                ctx->perf_prefill_ms, ctx->perf_decode_ms);
    }

    if (profile && ctx->profile_enabled) {
        double decode_tok = ctx->perf_decode_ms;
        int n_layers = ctx->config.dec_layers;
        int n_steps = ctx->perf_decode_steps;
        double avg_step = n_steps > 0 ? decode_tok / n_steps : 0;

        fprintf(stderr, "\n=== PROFILER (decode: %.0fms, %d steps, avg %.2fms/step) ===\n",
                decode_tok, n_steps, avg_step);

        double total_qkv = 0, total_attn = 0, total_proj = 0, total_mlp = 0;
        for (int l = 0; l < n_layers; l++) {
            total_qkv += ctx->perf_layer_qkv_ms[l];
            total_attn += ctx->perf_layer_attn_ms[l];
            total_proj += ctx->perf_layer_proj_ms[l];
            total_mlp += ctx->perf_layer_mlp_ms[l];
        }

        fprintf(stderr, "  QKV proj:   %7.1f ms (%5.1f%%)\n", total_qkv, decode_tok > 0 ? 100.0 * total_qkv / decode_tok : 0);
        fprintf(stderr, "  Attention:  %7.1f ms (%5.1f%%)\n", total_attn, decode_tok > 0 ? 100.0 * total_attn / decode_tok : 0);
        fprintf(stderr, "  Out proj:   %7.1f ms (%5.1f%%)\n", total_proj, decode_tok > 0 ? 100.0 * total_proj / decode_tok : 0);
        fprintf(stderr, "  MLP/MoE:    %7.1f ms (%5.1f%%)\n", total_mlp, decode_tok > 0 ? 100.0 * total_mlp / decode_tok : 0);
        fprintf(stderr, "  LM head:    %7.1f ms (%5.1f%%)\n", ctx->perf_lm_head_ms, decode_tok > 0 ? 100.0 * ctx->perf_lm_head_ms / decode_tok : 0);
        fprintf(stderr, "  Sampling:   %7.1f ms (%5.1f%%)\n", ctx->perf_sampling_ms, decode_tok > 0 ? 100.0 * ctx->perf_sampling_ms / decode_tok : 0);

        fprintf(stderr, "\n  Per-layer breakdown (ms/step avg):\n");
        fprintf(stderr, "  %-6s %7s %7s %7s %7s %7s\n", "Layer", "QKV", "Attn", "Proj", "MLP", "Total");
        for (int l = 0; l < n_layers; l++) {
            double d = n_steps > 0 ? (double)n_steps : 1.0;
            fprintf(stderr, "  %-6d %7.3f %7.3f %7.3f %7.3f %7.3f\n",
                    l,
                    ctx->perf_layer_qkv_ms[l] / d,
                    ctx->perf_layer_attn_ms[l] / d,
                    ctx->perf_layer_proj_ms[l] / d,
                    ctx->perf_layer_mlp_ms[l] / d,
                    ctx->perf_layer_total_ms[l] / d);
        }
    }
}

/* is_pdf_path — 按扩展名判断是否 PDF（.pdf，大小写不敏感） */
static int is_pdf_path(const char *path) {
    size_t len = strlen(path);
    return len >= 4 && strcasecmp(path + len - 4, ".pdf") == 0;
}

/* strip_running_header_footer — 多页文档的运行页眉/页脚去重
 * 某行作为页首（或页尾）非空行在 ≥3 页逐字节重复 ⇒ 运行页眉（页脚），
 * 从所有页移除该行——页眉是每页版面装饰，不是文档内容。
 * 页码类页脚因逐页数字不同不聚簇，天然不受影响。
 * DS_KEEP_PAGE_HEADERS=1 保留页眉页脚。 */
static void strip_running_header_footer(char **texts, int n) {
    if (n < 4 || getenv("DS_KEEP_PAGE_HEADERS")) return;

    /* 每页的首/尾非空行位置（offset 与长度；无则 len=-1） */
    size_t first_off[256], first_len[256], last_off[256], last_len[256];
    if (n > 256) n = 256;
    for (int i = 0; i < n; i++) {
        first_off[i] = 0; first_len[i] = (size_t)-1;
        last_off[i] = 0;  last_len[i] = (size_t)-1;
        if (!texts[i]) continue;
        size_t len = strlen(texts[i]);
        size_t pos = 0;
        while (pos <= len) {
            char *nl = pos < len ? memchr(texts[i] + pos, '\n', len - pos) : NULL;
            size_t e = nl ? (size_t)(nl - (texts[i] + pos)) : len - pos;
            if (e > 0) {
                if (first_len[i] == (size_t)-1) { first_off[i] = pos; first_len[i] = e; }
                last_off[i] = pos; last_len[i] = e;
            }
            if (!nl) break;
            pos += e + 1;
        }
    }

    /* 两阶段: 先按原始计数标记（删除会移位偏移，不能边删边数），再统一删除 */
    int rm_first[256] = {0}, rm_last[256] = {0};
    char hdr_val[256];
    size_t hdr_len = 0;
    int have_hdr = 0;

    for (int i = 0; i < n; i++) {
        if (first_len[i] == (size_t)-1 || first_len[i] == 0) continue;
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (i == j || first_len[j] != first_len[i]) continue;
            if (memcmp(texts[j] + first_off[j], texts[i] + first_off[i], first_len[i]) == 0) c++;
        }
        if (c + 1 >= 3) {
            rm_first[i] = 1;
            if (!have_hdr && first_len[i] < sizeof(hdr_val)) {
                memcpy(hdr_val, texts[i] + first_off[i], first_len[i]);
                hdr_val[first_len[i]] = '\0';
                hdr_len = first_len[i];
                have_hdr = 1;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (last_len[i] == (size_t)-1 || last_len[i] == 0) continue;
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (i == j || last_len[j] != last_len[i]) continue;
            if (memcmp(texts[j] + last_off[j], texts[i] + last_off[i], last_len[i]) == 0) c++;
        }
        if (c + 1 >= 3) rm_last[i] = 1;
    }

    /* 删除阶段: 先删页尾（偏移较大）再删页首，互不影响 */
    for (int i = 0; i < n; i++) {
        if (!rm_last[i]) continue;
        size_t cut = last_off[i];
        if (cut > 0 && texts[i][cut - 1] == '\n') cut--;   /* 连同前置换行移除 */
        texts[i][cut] = '\0';
    }
    for (int i = 0; i < n; i++) {
        if (!rm_first[i] || !texts[i] || !texts[i][0]) continue;
        size_t off = first_off[i], l = first_len[i];
        size_t rest = strlen(texts[i] + off + l);
        if (rest > 0 && texts[i][off + l] == '\n') l++;    /* 连同换行移除 */
        memmove(texts[i] + off, texts[i] + off + l, rest + 1);
    }

    /* 运行页眉的页中残留: 模型不一定把页眉块输出在页首，凡独立成行且
     * 与页眉值完全相等的行一律移除（整行恰等于页眉只可能是页眉块）。
     * 用重建法（逐行过滤再拼接）避免指针移位算术出错。 */
    if (have_hdr) {
        for (int i = 0; i < n; i++) {
            if (!texts[i]) continue;
            size_t len = strlen(texts[i]);
            char *tmp = (char *)malloc(len + 2);
            if (!tmp) continue;
            size_t w = 0, pos = 0;
            while (pos < len) {
                char *nl = memchr(texts[i] + pos, '\n', len - pos);
                size_t e = nl ? (size_t)(nl - (texts[i] + pos)) : len - pos;
                int is_hdr = (e == hdr_len &&
                              memcmp(texts[i] + pos, hdr_val, hdr_len) == 0);
                if (!is_hdr) {
                    memcpy(tmp + w, texts[i] + pos, e);
                    w += e;
                    if (nl) tmp[w++] = '\n';
                }
                if (!nl) break;
                pos += e + 1;
            }
            while (w > 0 && tmp[w - 1] == '\n') w--;
            memcpy(texts[i], tmp, w);
            texts[i][w] = '\0';
            free(tmp);
        }
    }
}

/* page_is_blank — 渲染页墨水覆盖率检测（非白像素 <0.3% 视为空白页）
 * 空白页/分隔页送进模型只会得到幻觉输出（并触发循环防护、浪费时长），
 * 跳过是唯一正确处理。0.3% 阈值: 文本页墨迹 2-10%，仅页码页 ~0.06%
 * （页码属版面装饰，跳过与页眉/页脚去重哲学一致）。 */
static int page_is_blank(const unsigned char *px, int w, int h) {
    size_t checked = 0, ink = 0;
    for (int y = 0; y < h; y += 2) {
        const unsigned char *row = px + (size_t)y * w * 3;
        for (int x = 0; x < w; x += 2) {
            const unsigned char *p = row + (size_t)x * 3;
            int minc = p[0] < p[1] ? p[0] : p[1];
            if (p[2] < minc) minc = p[2];
            if (minc < 200) ink++;
            checked++;
        }
    }
    return checked > 0 && ink * 1000 < checked * 3;
}

/* recognize_pdf_stream — PDF 逐页渲染 + 逐页识别（--stream 模式）
 *
 * 每页独立走完整流水线（渲染→SAM→编码→解码），页与页之间无上下文。
 * 页分隔/进度/状态走 stderr，识别结果走 stdout——重定向 stdout 即得纯文本。
 * 这些 stderr 反馈不受 --silent 压制：长时间识别若毫无动静看起来像卡死。
 */
static void recognize_pdf_stream(ds_ctx_t *ctx, const char *path,
                                  int profile, int verbosity) {
    ds_pdf_doc_t *doc = ds_pdf_open(path);
    if (!doc) {
        fprintf(stderr, "无法打开 PDF: %s\n", path);
        return;
    }
    int pages = ds_pdf_page_count(doc);
    fprintf(stderr, "识别中: %s (%d 页)\n", path, pages);

    /* 缓冲各页结果: 页间需要做循环尾环折叠与运行页眉/页脚去重 */
    char **page_texts = (char **)calloc((size_t)pages, sizeof(char *));
    if (!page_texts) {
        fprintf(stderr, "内存分配失败\n");
        ds_pdf_close(doc);
        return;
    }

    /* 两遍流水: 先渲染全部页 → 关闭 PDF → 再统一识别。
     *
     * 【为什么渲染与识别必须分离】31 页国标实测（V4）: 逐页"渲染一张识别
     * 一张"时，第 23 页起 6/6 页全部内容循环早停（111-1263 tok 截断，页尾
     * 内容丢失），且重试复现同样截断、截断点逐字节一致；改为两遍流水后
     * 全部页正常（~950 tok）。根因是 CoreGraphics 的 CGPDFDocument 内部
     * 状态与跨页 draw 时序有关：同一 PDF 连续 draw 多页时，后续页的渲染
     * 位图受前序 draw 影响（单页 PDF/独立 PNG 完全正常，31/31 页 PDF 中
     * 的像素与独立渲染逐位一致——污染不可见于像素，作用于识别路径）。
     * 全部渲染完成后（doc 关闭）再识别，绕开该状态。代价: 31 页位图常驻
     * 内存（~400MB，2048px RGB），桌面可接受；空白页不驻留。 */
    unsigned char **page_px = (unsigned char **)calloc((size_t)pages, sizeof(unsigned char *));
    int *px_w = (int *)calloc((size_t)pages, sizeof(int));
    int *px_h = (int *)calloc((size_t)pages, sizeof(int));
    if (!page_px || !px_w || !px_h) {
        fprintf(stderr, "内存分配失败\n");
        free(page_px); free(px_w); free(px_h);
        ds_pdf_close(doc);
        return;
    }
    /* 第一遍: 渲染全部页（空白页立即丢弃不驻留） */
    for (int i = 0; i < pages; i++) {
        unsigned char *pixels = NULL;
        int w = 0, h = 0;
        if (ds_pdf_render_page(doc, i, &pixels, &w, &h) != 0 || !pixels) {
            fprintf(stderr, "第 %d/%d 页渲染失败，跳过\n", i + 1, pages);
            continue;
        }
        if (page_is_blank(pixels, w, h)) {
            fprintf(stderr, "第 %d/%d 页为空白页，跳过\n", i + 1, pages);
            free(pixels);
            continue;
        }
        page_px[i] = pixels; px_w[i] = w; px_h[i] = h;
    }
    ds_pdf_close(doc);   /* CGPDF 状态随 doc 释放——识别阶段不再触碰 */

    /* 第二遍: 纯识别（无 PDF 状态介入） */
    for (int i = 0; i < pages; i++) {
        if (!page_px[i]) continue;
        if (pages > 1) {
            fprintf(stderr, "── 第 %d/%d 页 ──\n", i + 1, pages);
        }
        stream_progress_begin(ctx);
        char *text = recognize_pixels_with_retry(ctx, page_px[i], px_w[i], px_h[i], 3);
        free(page_px[i]);
        page_px[i] = NULL;
        stream_progress_end(ctx);
        if (text) {
            char label[32];
            snprintf(label, sizeof(label), "第 %d 页", i + 1);
            if (!ctx->last_output_loopy)
                warn_loopy_output(ctx, text, label);
            cleanup_loopy_output(ctx, text);
            page_texts[i] = text;
        } else {
            fprintf(stderr, "第 %d/%d 页识别失败\n", i + 1, pages);
        }
        print_perf_summary(ctx, profile, verbosity);
    }
    free(page_px); free(px_w); free(px_h);

    strip_running_header_footer(page_texts, pages);
    for (int i = 0; i < pages; i++) {
        if (page_texts[i]) {
            print_ocr_result(ctx, page_texts[i], verbosity);
            free(page_texts[i]);
        }
    }
    free(page_texts);
}

/* ═══════════════════════════════════════════════════════════════════
 * --stream 交互模式（REPL）
 * ═══════════════════════════════════════════════════════════════════ */

/* stream_help — 交互模式命令帮助 */
static void stream_help(int model_mode) {
    fprintf(stderr,
        "命令:\n"
        "  @<文件路径>   识别文件并输出结果（图片: PNG/JPEG/WebP/BMP/TIFF%s）\n"
        "  h / help      显示本帮助\n"
        "  q / quit      退出\n",
        model_mode ? "；PDF 逐页识别" : "");
}

/* run_stream_mode — 交互式识别循环
 *
 * 每行一个命令:
 *   @<路径>  识别指定文件（路径可含空格，或用引号包裹）；PDF 逐页识别
 *   q/quit/exit  退出;  h/help  帮助;  空行  忽略
 * 其余输入给出提示。EOF (Ctrl-D) 视为退出。
 *
 * use_platform_ocr=1 时为 Apple Vision 模式（无 ctx，仅图片）。
 */
static int run_stream_mode(ds_ctx_t *ctx, int use_platform_ocr, int platform_accurate,
                            int profile, int verbosity) {
    if (use_platform_ocr) {
        fprintf(stderr, "交互模式: Apple Vision OCR（仅图片，PDF 需模型模式）\n");
    } else {
        fprintf(stderr, "交互模式: %s (build %s %s)\n输入 @<文件路径> 回车识别，q 退出\n",
                model_version_name(ctx), __DATE__, __TIME__);
    }
    stream_help(!use_platform_ocr);

    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        /* 提示符走 stderr，保持 stdout 纯净；始终显示——
         * 即使 --silent（识别期间若无任何动静会像卡死） */
        fputs("ds_ocr> ", stderr);
        fflush(stderr);

        ssize_t nread = getline(&line, &cap, stdin);
        if (nread < 0) {
            /* EOF (Ctrl-D 或管道结束) */
            fputc('\n', stderr);
            break;
        }

        /* 去除首尾空白 */
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        size_t len = strlen(s);
        while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r' ||
                           s[len-1] == ' '  || s[len-1] == '\t'))
            s[--len] = '\0';
        if (len == 0) continue;

        if (strcmp(s, "q") == 0 || strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0)
            break;
        if (strcmp(s, "h") == 0 || strcmp(s, "help") == 0) {
            stream_help(!use_platform_ocr);
            continue;
        }
        if (s[0] != '@') {
            fprintf(stderr, "无法识别的输入 — 以 @ 开头接文件路径（如 @doc.pdf），q 退出\n");
            continue;
        }

        /* 取路径: @后的剩余内容（支持空格），或剥去成对引号 */
        char *path = s + 1;
        while (*path == ' ' || *path == '\t') path++;
        size_t plen = strlen(path);
        if (plen >= 2 && ((path[0] == '"' && path[plen-1] == '"') ||
                          (path[0] == '\'' && path[plen-1] == '\''))) {
            path[plen-1] = '\0';
            path++;
        }
        if (!*path) {
            fprintf(stderr, "用法: @<文件路径>\n");
            continue;
        }

        if (is_pdf_path(path)) {
            if (use_platform_ocr) {
                fprintf(stderr, "Vision 模式不支持 PDF，请使用模型模式（去掉 --vision）\n");
                continue;
            }
            recognize_pdf_stream(ctx, path, profile, verbosity);
        } else if (use_platform_ocr) {
            fprintf(stderr, "识别中: %s\n", path);
            char *text = ds_platform_ocr_file(path, platform_accurate, verbosity);
            if (text) {
                printf("%s\n", text);
                free(text);
            } else {
                fprintf(stderr, "识别失败: %s\n", path);
            }
        } else {
            fprintf(stderr, "识别中: %s\n", path);
            stream_progress_begin(ctx);
            char *text = recognize_file_with_retry(ctx, path);
            stream_progress_end(ctx);
            if (text) {
                cleanup_loopy_output(ctx, text);
                if (!ctx->last_output_loopy)
                    warn_loopy_output(ctx, text, "");
                print_ocr_result(ctx, text, verbosity);
                free(text);
            } else {
                fprintf(stderr, "识别失败: %s\n", path);
            }
            print_perf_summary(ctx, profile, verbosity);
        }
    }
    free(line);
    return 0;
}


/* usage — 打印帮助信息
 *
 * 各选项的详细说明:
 *   -d <dir>       模型目录，需包含 *.safetensors 和 config.json
 *   -i <file>      输入图像路径（--stream 模式下可省略）
 *   --stream       交互模式: 输入 @<文件> 回车识别图片/PDF，q 退出
 *   -t <n>         线程数（默认: 所有CPU核心，Apple M2 Pro = 8核心）
 *   -n <n>         最大生成token数（默认: 4096，OCR通常200-500足够）
 *   --temp <f>     采样温度（0=贪心解码，>0增加随机性，OCR推荐0）
 *   --rp <f>       重复惩罚（1.0=无惩罚，1.1-1.5常用，>1减少重复）
 *   --ngram <n>    N-gram阻断（0=禁用，20-35可防止退化解）
 *   --min-tokens   最少生成token数（防止过早输出EOS）
 *   --vision       使用macOS原生Vision OCR（不需要模型文件）
 *   --profile      启用逐层性能剖析（输出每层QKV/Attn/Proj/MLP耗时）
 *   --int4         INT8量化MoE专家权重（内存带宽减半，精度略降）
 *   --debug        调试输出（显示每步生成的token）
 *   --silent       静默模式（仅输出识别结果）
 */
static void usage(const char *prog) {
    fprintf(stderr, "ds_ocr — DeepSeek-OCR document recognition (pure C)\n");
    fprintf(stderr, "build: %s %s\n\n", __DATE__, __TIME__);
    fprintf(stderr, "Usage: %s -d <model_dir> -i <input.png> [options]\n", prog);
    fprintf(stderr, "       %s -d <model_dir> --stream\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with *.safetensors, config.json)\n");
    fprintf(stderr, "  -i <file>     Input image (PNG, JPEG, WebP, BMP, TIFF) or PDF (rendered per page)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --stream      Interactive mode: type @<file> per line to OCR images/PDFs, q to quit\n");
    fprintf(stderr, "  -t <n>        Number of threads (default: all CPUs)\n");
    fprintf(stderr, "  -n <n>        Max new tokens (default: 4096)\n");
    fprintf(stderr, "  --temp <f>    Sampling temperature (default: 0 = greedy)\n");
    fprintf(stderr, "  --rp <f>      Repetition penalty (default: 1.0, try 1.1-1.5)\n");
    fprintf(stderr, "  --ngram <n>   No-repeat ngram size (default: 0, try 20-35 if output is degenerate)\n");
    fprintf(stderr, "  --min-tokens <n> Min tokens before allowing EOS (default: 0; Unlimited-OCR: 32; 0 disables)\n");
    fprintf(stderr, "  --vision      Use macOS Vision OCR backend (no model required)\n");
    fprintf(stderr, "  --vision-fast Use macOS Vision OCR backend in fast mode\n");
    fprintf(stderr, "  --profile     Profile per-layer timing breakdown\n");
    fprintf(stderr, "  --int4        INT8 quantize MoE expert weights (2x less memory bandwidth)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer details)\n");
    fprintf(stderr, "  --silent      No stats output (recognition only on stdout; stream mode keeps prompts/progress on stderr)\n");
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    /* ── 命令行参数变量（带默认值）── */
    const char *model_dir = NULL;       /* 模型目录路径 */
    const char *input_image = NULL;     /* 输入图像路径 */
    int verbosity = 1;                  /* 输出详细度: 0=静默, 1=正常, 2=调试 */
    int n_threads = 0;                  /* 线程数: 0=自动(=CPU核心数) */
    int max_new_tokens = 4096;          /* 最大生成token数 */
    int raw_layout = 0;                 /* V4: 保留原生结构化输出 */
    float temperature = 0.0f;           /* 采样温度: 0=贪心(确定性) */
    float repeat_penalty = 1.0f;        /* 重复惩罚: 1.0=无惩罚 */
    int no_repeat_ngram_size = 0;       /* N-gram阻断: 0=禁用 */
    int min_new_tokens = -1;            /* 最少token数: -1=使用ctx默认值 */
    int use_platform_ocr = 0;           /* 使用macOS Vision OCR后端 */
    int platform_accurate = 1;          /* Vision OCR精度模式: 1=高精度, 0=快速 */
    int profile = 0;                    /* 启用逐层Profiler */
    int int4 = 0;                       /* 启用INT4/INT8量化 */
    int stream_mode = 0;                /* --stream 交互模式 */

    /* ── 解析命令行参数 ── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_image = argv[++i];
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream_mode = 1;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rp") == 0 && i + 1 < argc) {
            repeat_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ngram") == 0 && i + 1 < argc) {
            no_repeat_ngram_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-tokens") == 0 && i + 1 < argc) {
            min_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vision") == 0) {
            use_platform_ocr = 1;
            platform_accurate = 1;
        } else if (strcmp(argv[i], "--vision-fast") == 0) {
            use_platform_ocr = 1;
            platform_accurate = 0;
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = 1;
        } else if (strcmp(argv[i], "--int4") == 0) {
            int4 = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--raw-layout") == 0) {
            raw_layout = 1;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* 参数校验: 至少需要模型目录或Vision模式, 以及输入图像（--stream 下可省略 -i） */
    if ((!model_dir && !use_platform_ocr) || (!input_image && !stream_mode)) {
        usage(argv[0]);
        return 1;
    }
    if (stream_mode && input_image) {
        fprintf(stderr, "注意: --stream 模式忽略 -i %s，文件请用 @<路径> 逐个输入\n", input_image);
    }

    /* 设置全局verbose级别（影响所有模块的日志输出） */
    ds_verbose = verbosity;
    /* BF16中间截断: 默认开启，模拟Python的BF16计算路径
     * 不开启时F32精度与Python的BF16训练路径不一致，12层MoE后EOS logit偏低，
     * 导致OCR文本结束后产生幻觉内容。
     * 关闭方式: DS_BF16_SIMULATE_PYTHON=0（也接受 false/off）。
     * 旧实现是"env 存在即开启"——默认反而关闭，且设为 0 仍会开启，
     * 与 ds_ocr.c 里文档化的契约（默认开启）完全相反。 */
    {
        const char *env = getenv("DS_BF16_SIMULATE_PYTHON");
        if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0 ||
                    strcmp(env, "FALSE") == 0 || strcmp(env, "off") == 0))
            ds_bf16_simulate_python = 0;
        else
            ds_bf16_simulate_python = 1;
    }

    /* ── macOS Vision OCR 快捷路径（不使用模型） ── */
    if (use_platform_ocr) {
        if (is_pdf_path(input_image)) {
            fprintf(stderr, "Vision 模式不支持 PDF，请使用模型模式（去掉 --vision）\n");
            return 1;
        }
        if (stream_mode) {
            return run_stream_mode(NULL, 1, platform_accurate, 0, verbosity);
        }
        char *text = ds_platform_ocr_file(input_image, platform_accurate, verbosity);
        if (!text) {
            fprintf(stderr, "Platform OCR failed or is unavailable on this build\n");
            return 1;
        }
        printf("%s\n", text);
        free(text);
        return 0;
    }

    /* ── 初始化线程池 ── */
    if (n_threads <= 0) n_threads = ds_get_num_cpus();  /* 默认: 使用所有CPU核心 */
    ds_set_threads(n_threads);

    /* ── 加载模型 ──
     * ds_load() 执行以下步骤:
     *   1. detect_model_version() — 检测V1/V2/V3版本
     *   2. init_config() — 初始化模型配置参数
     *   3. multi_safetensors_open() — mmap加载权重文件
     *   4. load_all_weights() — 解析并加载所有权重
     *   5. alloc_decoder_buffers() — 分配KV缓存和解码缓冲区
     *   6. ds_metal_init() — (可选)初始化Metal GPU加速
     */
    ds_ctx_t *ctx = ds_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* ── 应用推理设置 ── */
    ctx->max_new_tokens = max_new_tokens;
    ctx->keep_raw_layout = raw_layout;
    ctx->temperature = temperature;
    ctx->repeat_penalty = repeat_penalty;
    ctx->no_repeat_ngram_size = no_repeat_ngram_size;
    if (min_new_tokens >= 0) ctx->min_new_tokens = min_new_tokens;
    if (profile) ctx->profile_enabled = 1;
    if (int4) {
        /* INT4量化: 将BF16专家权重量化为INT8格式
         * 减少解码时内存带宽约4倍，精度损失极小
         * 量化在加载后一次性完成，不影响后续推理速度 */
        ctx->int4_enabled = 1;
        ds_quantize_moe_int4(ctx);
    }

    /* Unlimited-OCR (V3) 默认开启 ngram=35 重复抑制
     * Python使用 SlidingWindowNoRepeatNgramProcessor(ngram_size=35, window=128)
     * 这防止V3在长文档解析时产生重复的n-gram。
     * V4 (Unlimited-OCR-2) 官方 infer() 默认 no_repeat_ngram_size=0（纯贪心，
     * ngram 是可选项），保持一致避免干扰 DASH 路由下的正常生成。*/
    if (ctx->config.model_version == 3 && no_repeat_ngram_size == 0) {
        ctx->no_repeat_ngram_size = 35;
    }

    /* ── 设置流式输出回调 ──
     * V3/V4 禁用流式输出: 因为可能产生幻觉前缀（如"[No text detected]"），
     * 需要后处理删除，这无法逐token完成——必须等全部生成完毕后再处理。
     * V1/V2: 启用流式输出，边生成边打印 */
    if (verbosity > 0 && ctx->config.model_version != 3 && ctx->config.model_version != 4) {
        ds_set_token_callback(ctx, stream_token, NULL);
    }

    /* ── 交互模式: 模型加载完成后进入 REPL 循环 ── */
    if (stream_mode) {
        int rc = run_stream_mode(ctx, 0, 0, profile, verbosity);
        ds_free(ctx);
        return rc;
    }

    /* ── 单次模式: PDF 输入 → 逐页渲染 + 逐页识别（与 --stream 同一路径） ── */
    if (is_pdf_path(input_image)) {
        recognize_pdf_stream(ctx, input_image, profile, verbosity);
        ds_free(ctx);
        return 0;
    }

    /* ── 执行OCR识别 ──
     * ds_recognize() 完成完整的推理流水线:
     *   图像加载 → SAM编码 → 编码器(CLIP/DeepEncoder) → 投影 → 构造Prompt
     *   → Prefill(批量填充KV缓存) → Decode循环(逐token生成) → 文本输出
     */
    char *text = recognize_file_with_retry(ctx, input_image);

    if (text) {
        /* V3 或静默模式: 一次性打印完整结果
         * V1/V2 非静默: 结果已通过stream_token逐token打印，此处只打换行 */
        cleanup_loopy_output(ctx, text);
        print_ocr_result(ctx, text, verbosity);
        free(text);
    } else {
        fprintf(stderr, "Recognition failed\n");
        ds_free(ctx);
        return 1;
    }

    /* ── 性能统计输出（--profile 时附加逐层Profiler） ── */
    print_perf_summary(ctx, profile, verbosity);

    /* ── 释放模型资源 ── */
    ds_free(ctx);
    return 0;
}