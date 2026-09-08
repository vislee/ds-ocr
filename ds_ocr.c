/*
 * ds_ocr.c - DeepSeek-OCR Pure C Inference Engine
 * ds_ocr.c — DeepSeek-OCR 纯C推理引擎主协调器
 * ds_ocr.c — DeepSeek-OCR 纯C推理引擎主协调器
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】整个推理流水线的"总指挥"
 * ─────────────────────────────────────────────────────────────────────
 * 协调各个子模块（图像加载、SAM编码器、CLIP/DeepEncoder、MoE解码器、分词器），
 * 完成从图像到OCR文本的端到端推理。
 *
 * 【完整推理流程】(详见 04-完整推理流程篇.md)
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ 1. 图像加载: ds_image_load() → RGB像素 [H, W, 3]           │
 *   │ 2. 图像预处理: pad/resize/crop → 模型所需尺寸               │
 *   │ 3. SAM编码: ds_sam_forward() → 视觉特征 [256, 1024/896]     │
 *   │ 4. 编码器:                                                    │
 *   │    V1/V3: CLIP(sam_features) → Concat(SAM,CLIP) → [256,2048]│
 *   │    V2:   DeepEncoder(sam_features) → [256, 896]              │
 *   │ 5. 投影: Projector(2048/896→1280) → 与解码器维度对齐         │
 *   │ 6. 构造Prompt: [BOS] + image_tokens + prompt_text            │
 *   │ 7. Prefill: 批量处理Prompt → 填充KV缓存                     │
 *   │ 8. Decode循环: 逐token生成 → argmax → 下一个token → 文本     │
 *   │ 9. 后处理: V3去除det/ref标签, 截断空格                       │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * 【三版本差异】(详见 ds_ocr.h 中的 ds_config_t)
 *   V1: 单图1024×1024 + CLIP双编码器 + 标准因果注意力
 *   V2: 多裁剪768×768 + DeepEncoder V2 + 标准因果注意力
 *   V3: 单图1024×1024 + CLIP双编码器 + R-SWA滑动窗口注意力
 *
 * 【性能优化历程】(详见 05-性能优化实战篇.md)
 *   v0.5(97s) → v0.7(28s) → v0.8(17s) → v0.9(15s) = 6.5×加速
 *   关键优化: 并行N+1编码 / 批量sgemm Prefill / Argmax LM Head / 连续expert块
 *
 * 【与 qwen-asr 的对应】
 *   qwen-asr: audio → Mel → Conv+Encoder → Proj → Dense Decoder → 文字
 *   ds-ocr:   image → SAM → CLIP/DeepEncoder → Proj → MoE Decoder → 文字
 *   结构相同，输入不同（音频 vs 图像），解码器不同（Dense vs MoE）
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Main coordinator: image → visual tokenizer → encoder → MoE decoder → text
 */

#include "ds_ocr.h"                /* 公共头文件：常量、结构体、API声明 */
#include "ds_kernels.h"            /* 数学内核：矩阵运算、线程池、RoPE */
#include "ds_safetensors.h"       /* Safetensors权重读取器（mmap零拷贝） */
#include "ds_image.h"             /* 图像加载与预处理（stb_image封装） */
#include "ds_visual_tokenizer.h"  /* SAM视觉分词器：patch嵌入+Transformer编码 */
#include "ds_deep_encoder.h"      /* DeepEncoder V2 (Qwen2-0.5B) / CLIP编码器 */
#include "ds_moe_decoder.h"       /* MoE解码器：64路由专家+2共享专家 */
#include "ds_tokenizer.h"         /* BPE分词器：token ID ←→ 文本 */
#include "ds_quantize.h"          /* INT4/INT8量化工具 */

#include "ds_dump.h"              /* 调试用tensor dump工具 */
#include <stdio.h>   /* printf, fprintf, snprintf */
#include <stdlib.h>  /* malloc, free, calloc, realloc, posix_memalign */
#include <string.h>  /* memset, memcpy, memcmp, strstr */
#include <math.h>    /* powf, fabsf, sqrtf */
#include <pthread.h> /* 多线程并行裁剪编码 */
#include <sys/time.h>/* gettimeofday — 毫秒级计时 */
#include <sys/stat.h>/* stat — 文件状态检查 */

/* ds_verbose — 全局日志详细度
 * 0 = 静默（仅输出识别结果）
 * 1 = 正常（显示加载进度、性能统计）
 * 2 = 调试（显示每步token、权重加载详情）
 * 3 = 详细（显示前几步的top-k logits）
 */
int ds_verbose = 1;

/* ds_bf16_simulate_python — BF16中间截断开关
 *
 * 默认开启(1): 模拟Python PyTorch的BF16计算路径
 * 在每层MoE的gate+up+down计算后，将F32结果截断为BF16精度再转回F32
 * 这样C引擎的中间计算精度与Python一致
 *
 * 为什么必须开启?
 *   Python PyTorch训练/推理使用BF16自动混合精度(AMP)
 *   如果C引擎用F32全精度，12层MoE后累积误差使EOS logit偏低
 *   导致OCR文本完成后不输出EOS，继续产生幻觉内容
 *   开启后截断使C与Python精度一致，EOS在正确位置触发
 *
 * 关闭方式: 设置环境变量 DS_BF16_SIMULATE_PYTHON=0
 */
int ds_bf16_simulate_python = 1;

/* g_dump_crop_id — 当前裁剪ID（用于调试dump，-1表示未设置）
 * V2多裁剪模式下，每个裁剪编码时设置此值，便于ds_dump按crop ID保存中间结果
 */
int g_dump_crop_id = -1;

/* ========================================================================
 * Timing Helper — 毫秒级计时器
 * ======================================================================== */

/* now_ms: 获取当前时间的毫秒数（基于gettimeofday）
 * 用于测量推理各阶段耗时：编码、prefill、decode
 */
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ds_time_sec now provided by ds_kernels.h (inline) */

/* ========================================================================
 * V3 Streaming Det Tag Filter
 * V3 流式 Det 标签过滤器
 *
 * 【背景】Unlimited-OCR (V3) 输出格式中包含检测标签:
 *   <|det|>label [x1, y1, x2, y2]<|/det|>  ← 检测框坐标
 *   <|ref|>...<|/ref|><|det|>...<|/det|>    ← 引用+检测对
 * 这些标签对用户无用（只需要文本），需要实时过滤掉。
 *
 * 【挑战】标签可能跨多个token:
 *   token1: "<|de"  token2: "t|>label"  token3: "[0,"  token4: "0]<|/det|>"
 * 不能逐token判断——必须缓冲可能构成标签的token，确认后再决定输出或丢弃
 *
 * 【算法】状态机式缓冲:
 *   1. 将每个token追加到缓冲区
 *   2. 尝试匹配完整的 det/ref 标签模式
 *   3. 匹配成功 → 丢弃整个标签，继续处理
 *   4. 无法匹配 → 找到最后一个 '<'（可能是标签开始）
 *      '<' 之前的文本安全输出，'<..' 之后继续缓冲
 * ======================================================================== */

/* ds_try_strip_det_tag: 尝试从缓冲区中移除完整的det/ref标签
 *
 * 检测两种标签模式:
 *   1. 独立 <|det|>...<|/det|> — 检测框标签（含标签名和坐标）
 *   2. <|ref|>...<|/ref|><|det|>...<|/det|> — 引用+检测对
 *
 * 返回值: 消耗的字符数（0=未找到完整标签）
 * 找到完整标签时，buf原地修改，*out_new_len设为新长度
 */
static int ds_try_strip_det_tag(char *buf, int buf_len, int *out_new_len) {
    /* 尝试匹配独立 <|det|>...<|/det|> 模式 */
    /* 从位置7开始搜索<|/det|>结束标签 */
    if (buf_len >= 7 && memcmp(buf, "<|det|>", 7) == 0) {
        for (int i = 7; i + 8 <= buf_len; i++) {
            if (memcmp(buf + i, "<|/det|>", 8) == 0) {
                /* 找到完整的det标签：跳过整个 buf[0..i+8) */
                int consumed = i + 8;
                *out_new_len = 0;
                return consumed;
            }
        }
    }
    /* 尝试匹配 <|ref|>...<|/ref|><|det|>...<|/det|> 对 */
    /* <|ref|>是7字符，<|/ref|>是8字符，<|det|>是7字符，<|/det|>是8字符 */
    if (buf_len >= 7 && memcmp(buf, "<|ref|>", 7) == 0) {
        int ref_end = -1;
        /* 先找<|/ref|>结束位置 */
        for (int i = 7; i + 8 <= buf_len; i++) {
            if (memcmp(buf + i, "<|/ref|>", 8) == 0) {
                ref_end = i + 8;
                break;
            }
        }
        /* ref结束后紧接着<|det|>，再找<|/det|> */
        if (ref_end > 0 && ref_end + 7 <= buf_len && memcmp(buf + ref_end, "<|det|>", 7) == 0) {
            for (int i = ref_end + 7; i + 8 <= buf_len; i++) {
                if (memcmp(buf + i, "<|/det|>", 8) == 0) {
                    /* 找到完整的ref+det对：跳过整个缓冲区内容 */
                    int consumed = i + 8;
                    *out_new_len = 0;
                    return consumed;
                }
            }
        }
    }
    return 0;  /* 缓冲区中尚未出现完整标签模式 */
}

/* ds_stream_filter_det: 流式token经过det标签过滤器
 *
 * 逐个将解码后的token送入缓冲区累积，当判断缓冲区内容为完整标签时丢弃；
 * 否则将非标签的安全前缀（'<'之前的内容）通过回调输出。
 *
 * 为什么不能逐token判断?
 *   BPE分词可能将一个标签切分成多个token，如 "<|de" + "t|>"
 *   缓冲后才能正确识别完整的标签边界
 */
static void ds_stream_filter_det(ds_ctx_t *ctx, const char *piece) {
    if (!ctx || !piece || !ctx->token_cb) return;
    if (ctx->config.model_version != 3) {
        /* 非V3模型：直接透传，无需缓冲过滤det标签 */
        ctx->token_cb(piece, ctx->token_cb_userdata);
        return;
    }

    int piece_len = (int)strlen(piece);
    ds_token_cb cb = ctx->token_cb;
    void *ud = ctx->token_cb_userdata;
    char *buf = ctx->_det_buf;
    int *pbuf_len = &ctx->_det_buf_len;
    int buf_len = *pbuf_len;

    /* Append piece to buffer */
    if (buf_len + piece_len >= (int)sizeof(ctx->_det_buf) - 1) {
        /* Buffer would overflow — flush what we have.
         * Overflow during the prefix phase is implausible (256 bytes of
         * closing tags); give up on prefix suppression if it happens. */
        ctx->_v3_prefix_done = 1;
        if (buf_len > 0) {
            buf[buf_len] = '\0';
            cb(buf, ud);
            *pbuf_len = 0;
            buf_len = 0;
        }
        /* Check if piece itself starts a tag */
        if (piece_len >= 2 && piece[0] == '<' && piece[1] == '|') {
            /* Could be start of a tag — buffer it */
            int copy = piece_len < (int)sizeof(ctx->_det_buf) - 1 ? piece_len : (int)sizeof(ctx->_det_buf) - 1;
            memcpy(buf, piece, copy);
            buf[copy] = '\0';
            *pbuf_len = copy;
            buf_len = copy;
        } else {
            /* Not a tag start — emit directly */
            cb(piece, ud);
            return;
        }
    } else {
        memcpy(buf + buf_len, piece, piece_len);
        buf_len += piece_len;
        buf[buf_len] = '\0';
        *pbuf_len = buf_len;
    }

    /* ── Start-of-output orphaned closing-tag suppression ──
     * The model occasionally opens with a run of HTML closing tags such as
     * "</td></tr></table>" when it tries to parse diagram/table-like content.
     * Recognized text never begins mid-table, so drop the run (and any
     * leading whitespace). Tokens are held back until the decision is
     * possible: either real content appears, or generation ends. */
    if (!ctx->_v3_prefix_done) {
        int p = 0;
        int partial_tag = 0;
        int saw_content = 0;
        while (p < buf_len) {
            char c = buf[p];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { p++; continue; }
            if (c == '<') {
                if (p + 1 >= buf_len) { partial_tag = 1; break; }        /* lone '<' */
                if (buf[p + 1] == '/') {
                    int gt = p + 2;
                    while (gt < buf_len && buf[gt] != '>') gt++;
                    if (gt >= buf_len) { partial_tag = 1; break; }       /* partial "</x" */
                    p = gt + 1;
                    continue;
                }
            }
            saw_content = 1;
            break;
        }
        /* Decision not yet possible — keep buffering */
        if (partial_tag || (!saw_content && p == buf_len)) {
            if (buf_len < (int)sizeof(ctx->_det_buf) - 8) {
                *pbuf_len = buf_len;
                return;
            }
            /* Buffer nearly full of tags — give up suppressing */
        }
        ctx->_v3_prefix_done = 1;
        if (p > 0) {
            buf_len -= p;
            memmove(buf, buf + p, buf_len);
            buf[buf_len] = '\0';
            *pbuf_len = buf_len;
            if (buf_len == 0) return;  /* nothing but the dropped prefix */
        }
        /* Fall through to det-tag logic with the remaining text */
    }

    /* Try to strip complete det/ref tags from buffer */
    while (buf_len > 0) {
        int new_len = 0;
        int consumed = ds_try_strip_det_tag(buf, buf_len, &new_len);
        if (consumed > 0) {
            /* Found and stripped a complete tag */
            buf_len -= consumed;
            memmove(buf, buf + consumed, buf_len);
            buf[buf_len] = '\0';
            *pbuf_len = buf_len;
            continue;
        }

        /* No complete tag found. Check if buffer could be a partial tag start.
         * A tag start begins with '<' followed by '|'. If we have '<' at the
         * end without the matching '|', we need to wait for more tokens.
         * Otherwise, flush safe prefix. */
        int safe_prefix = buf_len;
        /* Find the last '<' in buffer — anything before it is safe to emit */
        for (int i = buf_len - 1; i >= 0; i--) {
            if (buf[i] == '<') {
                safe_prefix = i;
                break;
            }
        }
        /* But also check if '<' could be part of a partial "<|" tag start */
        int has_partial_tag = 0;
        for (int i = 0; i < buf_len; i++) {
            if (buf[i] == '<') {
                /* Check if from position i, we could have the start of a tag */
                int remaining = buf_len - i;
                /* "<|det|>" is 7 chars, "<|ref|>" is 7, "<|/det|>" is 8, "<|/ref|>" is 8 */
                if (remaining >= 2 && buf[i + 1] == '|') {
                    /* This is already inside our buffer — check if complete tag */
                    /* Already handled by ds_try_strip_det_tag above */
                    has_partial_tag = 1;
                    safe_prefix = i;
                    break;
                } else if (remaining == 1) {
                    /* Just '<' — could be start of "<|det|>" etc. */
                    has_partial_tag = 1;
                    safe_prefix = i;
                    break;
                }
                /* '<' not followed by '|' — not a tag start, safe */
            }
        }

        if (!has_partial_tag) {
            /* No potential tag in buffer — flush all */
            if (buf_len > 0) {
                buf[buf_len] = '\0';
                cb(buf, ud);
                *pbuf_len = 0;
                return;
            }
            return;
        }

        /* Flush safe prefix (everything before potential tag start) */
        if (safe_prefix > 0) {
            char saved = buf[safe_prefix];
            buf[safe_prefix] = '\0';
            cb(buf, ud);
            buf[safe_prefix] = saved;
            buf_len -= safe_prefix;
            memmove(buf, buf + safe_prefix, buf_len);
            buf[buf_len] = '\0';
            *pbuf_len = buf_len;
        } else {
            /* Entire buffer is potential tag — wait for more tokens */
            return;
        }
    }
}

/* ========================================================================
 * Parallel crop encoding (for multi-crop SAM + encoder)
 * V2 并行多裁剪编码
 *
 * 【设计原理】
 * V2 模型处理大图像时，将图像分割为多个768×768的局部裁剪 + 1个1024×1024全局图。
 * 每个裁剪的编码完全独立（SAM + DeepEncoder 无数据依赖），天然可并行。
 *
 * 【性能对比】
 *   串行 (v0.5): global(12s) + local1(9.5s) + ... + local4(9.5s) = ~50s
 *   并行 (v0.9): max(global(12s), local(9.5s)) = ~9s → 实测5.5×加速
 *
 * 【线程模型】
 *   N+1 个 worker 线程同时启动（N个local + 1个global）
 *   M2 Pro: 6个性能核 + 4个能效核，5个worker自然分到不同核心
 *   性能核处理全局图（1024²，计算量大），能效核处理局部裁剪（768²，计算量小）
 * ======================================================================== */

typedef struct {
    ds_ctx_t *ctx;
    ds_image_t *crop;
    int crop_id;
    int tokens_per_crop;
    float *sam_tokens;
    int n_sam_tokens;
    float *enc_tokens;
    int n_enc_tokens;
    double sam_time;
    double enc_time;
    int failed;
} crop_task_t;

static void *crop_worker(void *arg) {
    crop_task_t *t = (crop_task_t *)arg;
    double t0 = ds_time_sec();
    t->sam_tokens = ds_sam_forward_image(t->ctx, t->crop, &t->n_sam_tokens, NULL);
    t->sam_time = ds_time_sec() - t0;
    if (!t->sam_tokens) { t->failed = 1; return NULL; }

    double e0 = ds_time_sec();
    t->enc_tokens = ds_encoder_forward_v2(t->ctx, t->sam_tokens, t->n_sam_tokens,
                                           &t->n_enc_tokens, t->tokens_per_crop,
                                           t->ctx->vis_tokenizer.causal_query_768_embeddings);
    t->enc_time = ds_time_sec() - e0;
    free(t->sam_tokens);
    t->sam_tokens = NULL;
    if (!t->enc_tokens) { t->failed = 1; return NULL; }
    return NULL;
}

/* Worker for V3 (Unlimited-OCR) local crop: SAM + CLIP + projector.
 * Stores the raw CLIP output (WITH newlines) — grid assembly happens on the
 * main thread after joining, so each crop's [100, 1280] block lands in place. */
static void *v3_crop_worker(void *arg) {
    crop_task_t *t = (crop_task_t *)arg;
    double t0 = ds_time_sec();
    t->sam_tokens = ds_sam_forward_image(t->ctx, t->crop, &t->n_sam_tokens, NULL);
    t->sam_time = ds_time_sec() - t0;
    if (!t->sam_tokens) { t->failed = 1; return NULL; }

    double e0 = ds_time_sec();
    t->enc_tokens = ds_clip_encoder_forward(t->ctx, NULL, 0, 0, 0,
                                             t->sam_tokens, t->n_sam_tokens,
                                             &t->n_enc_tokens);
    t->enc_time = ds_time_sec() - e0;
    free(t->sam_tokens);
    t->sam_tokens = NULL;
    if (!t->enc_tokens) { t->failed = 1; return NULL; }
    return NULL;
}

/* Worker for global crop (1024x1024) — uses causal_query_embeddings (256 queries) */
static void *crop_worker_global(void *arg) {
    crop_task_t *t = (crop_task_t *)arg;
    double t0 = ds_time_sec();
    t->sam_tokens = ds_sam_forward_image(t->ctx, t->crop, &t->n_sam_tokens, NULL);
    t->sam_time = ds_time_sec() - t0;
    if (!t->sam_tokens) { t->failed = 1; return NULL; }

    double e0 = ds_time_sec();
    t->enc_tokens = ds_encoder_forward_v2(t->ctx, t->sam_tokens, t->n_sam_tokens,
                                           &t->n_enc_tokens, t->tokens_per_crop,
                                           t->ctx->vis_tokenizer.causal_query_embeddings);
    t->enc_time = ds_time_sec() - e0;
    free(t->sam_tokens);
    t->sam_tokens = NULL;
    if (!t->enc_tokens) { t->failed = 1; return NULL; }
    return NULL;
}

/* ========================================================================
 * Configuration Detection — 模型版本自动检测
 *
 * 读取 config.json，通过关键字检测模型版本:
 *   V3 (Unlimited-OCR): 包含 "sliding_window_size" 或 "unlimited-ocr"
 *   V2 (DeepEncoder V2): 包含 "DeepEncoderV2" 或 "causal_flow" 或 enc_type=2
 *   V1 (DeepSeek-OCR): 以上都不匹配 → 默认V1
 *
 * 【为什么需要自动检测?】
 * 三个版本共享大部分代码，差异仅在编码器类型和注意力机制。
 * 自动检测让用户无需指定版本，一个二进制兼容所有模型。
 * ======================================================================== */

static int detect_model_version(const char *model_dir) {
    /* Check config.json for model type indicators */
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "detect_model_version: cannot open %s\n", path);
        return -1;
    }

    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    /* Check for Unlimited-OCR first (has sliding_window_size in config) */
    if (strstr(buf, "sliding_window_size") || strstr(buf, "unlimited-ocr") ||
        strstr(buf, "Unlimited-OCR") || strstr(buf, "UnlimitedOCR")) {
        return 3;
    }

    if (strstr(buf, "DeepEncoderV2") || strstr(buf, "deepencoderv2") ||
        strstr(buf, "causal_flow") ||
        (strstr(buf, "enc_type") && strstr(buf, "\"2\""))) {
        return 2;
    }
    return 1;
}

static void init_config(ds_config_t *cfg, int version) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->model_version = version;

    /* Vision tokenizer (SAM ViT-B) */
    cfg->image_size = DS_IMAGE_SIZE;
    cfg->sam_patch_size = DS_SAM_PATCH_SIZE;
    cfg->sam_embed_dim = DS_SAM_EMBED_DIM;
    cfg->sam_heads = DS_SAM_HEADS;
    cfg->sam_head_dim = DS_SAM_HEAD_DIM;
    cfg->sam_mlp_dim = DS_SAM_MLP_DIM;
    cfg->sam_window_size = DS_SAM_WINDOW_SIZE;
    cfg->sam_neck_dim = DS_SAM_NECK_DIM;
    cfg->sam_ds1_dim = DS_SAM_DS1_DIM;
    cfg->visual_tokens_base = DS_VISUAL_TOKENS_BASE;
    cfg->max_local_crops = DS_MAX_LOCAL_CROPS;
    cfg->sam_global_attn_indexes[0] = 2;
    cfg->sam_global_attn_indexes[1] = 5;
    cfg->sam_global_attn_indexes[2] = 8;
    cfg->sam_global_attn_indexes[3] = 11;

    if (version == 2) {
        /* DeepEncoder V2 (Qwen2-0.5B based) */
        cfg->enc_type = 2;
        cfg->enc_layers = DS_ENC_V2_LAYERS;
        cfg->enc_hidden = DS_ENC_V2_HIDDEN;
        cfg->enc_heads = DS_ENC_V2_HEADS;
        cfg->enc_kv_heads = DS_ENC_V2_KV_HEADS;
        cfg->enc_head_dim = DS_ENC_V2_HEAD_DIM;
        cfg->enc_intermediate = DS_ENC_V2_INTERMEDIATE;
        cfg->enc_causal_flow_queries = DS_VISUAL_TOKENS_BASE; /* 256 queries */
        cfg->proj_input_dim = DS_PROJECTOR_V2_INPUT; /* 896 */
        cfg->enc_rope_theta = 1000000.0f;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM_V2; /* 896 for V2 */
        cfg->sliding_window_size = 0; /* No R-SWA for V2 */
    } else if (version == 3) {
        /* Unlimited-OCR: same as V1 (CLIP encoder) but with R-SWA decoder */
        cfg->enc_type = 1;
        cfg->enc_layers = DS_CLIP_LAYERS;
        cfg->enc_hidden = DS_CLIP_HIDDEN;
        cfg->enc_heads = DS_CLIP_HEADS;
        cfg->enc_kv_heads = DS_CLIP_HEADS;
        cfg->enc_head_dim = DS_CLIP_HEAD_DIM;
        cfg->enc_intermediate = DS_CLIP_MLP_DIM;
        cfg->enc_causal_flow_queries = 0;
        cfg->proj_input_dim = DS_PROJECTOR_V1_INPUT; /* 2048: CLIP(1024) + SAM(1024) */
        cfg->enc_rope_theta = 0;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM; /* 1024 for V1/Unlimited-OCR */
        cfg->sliding_window_size = 128; /* R-SWA window for Unlimited-OCR */
    } else {
        /* V1: CLIP ViT-L/14 */
        cfg->enc_type = 1;
        cfg->enc_layers = DS_CLIP_LAYERS;
        cfg->enc_hidden = DS_CLIP_HIDDEN;
        cfg->enc_heads = DS_CLIP_HEADS;
        cfg->enc_kv_heads = DS_CLIP_HEADS;
        cfg->enc_head_dim = DS_CLIP_HEAD_DIM;
        cfg->enc_intermediate = DS_CLIP_MLP_DIM;
        cfg->enc_causal_flow_queries = 0;
        cfg->proj_input_dim = DS_PROJECTOR_V1_INPUT; /* 2048 */
        cfg->enc_rope_theta = 0;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM; /* 1024 for V1 */
    }

    /* MoE Decoder (same for V1 and V2) */
    cfg->enc_output_dim = DS_DEC_HIDDEN;
    cfg->dec_hidden = DS_DEC_HIDDEN;
    cfg->dec_layers = DS_DEC_LAYERS;
    cfg->dec_heads = DS_DEC_HEADS;
    cfg->dec_kv_heads = DS_DEC_KV_HEADS;
    cfg->dec_head_dim = DS_DEC_HEAD_DIM;
    cfg->dec_intermediate = DS_DEC_INTERMEDIATE;
    cfg->dec_moe_inter = DS_DEC_MOE_INTER;
    cfg->dec_n_routed_experts = DS_DEC_NUM_EXPERTS;
    cfg->dec_n_shared_experts = DS_DEC_SHARED_EXPERTS;
    cfg->dec_top_k = DS_DEC_TOP_K;
    cfg->dec_first_k_dense = DS_DEC_FIRST_K_DENSE;
    cfg->vocab_size = DS_DEC_VOCAB_SIZE;
    cfg->dec_rms_norm_eps = 1e-6f;
    cfg->dec_rope_theta = 10000.0f;

    if (version == 3) {
        /* Unlimited-OCR: SAM + CLIP-L/14 + 2D grid + sliding window */
        cfg->enc_type = 1;  /* CLIP encoder */
        cfg->enc_layers = DS_CLIP_LAYERS;
        cfg->enc_hidden = DS_CLIP_HIDDEN;
        cfg->enc_heads = DS_CLIP_HEADS;
        cfg->enc_kv_heads = DS_CLIP_HEADS;  /* CLIP uses same heads for KV */
        cfg->enc_head_dim = DS_CLIP_HEAD_DIM;
        cfg->enc_intermediate = DS_CLIP_MLP_DIM;
        cfg->enc_causal_flow_queries = 0;
        cfg->proj_input_dim = DS_PROJECTOR_V1_INPUT; /* 2048 */
        cfg->enc_rope_theta = 0;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM; /* 1024 */

        /* Unlimited-OCR specific */
        cfg->sliding_window_size = 128;
        cfg->has_qk_norm = 0;  /* No per-head Q/K RMSNorm */
        cfg->image_token_id = 128815;
        cfg->image_size_crop = 640;  /* crop image size for dynamic_preprocess */
    } else if (version == 2) {
        /* DeepEncoder V2 (Qwen2-0.5B based) */
        cfg->enc_type = 2;
        cfg->enc_layers = DS_ENC_V2_LAYERS;
        cfg->enc_hidden = DS_ENC_V2_HIDDEN;
        cfg->enc_heads = DS_ENC_V2_HEADS;
        cfg->enc_kv_heads = DS_ENC_V2_KV_HEADS;
        cfg->enc_head_dim = DS_ENC_V2_HEAD_DIM;
        cfg->enc_intermediate = DS_ENC_V2_INTERMEDIATE;
        cfg->enc_causal_flow_queries = DS_VISUAL_TOKENS_BASE; /* 256 queries */
        cfg->proj_input_dim = DS_PROJECTOR_V2_INPUT; /* 896 */
        cfg->enc_rope_theta = 1000000.0f;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM_V2; /* 896 for V2 */

        cfg->sliding_window_size = 0;
        cfg->has_qk_norm = 0;  /* V2 doesn't have per-head Q/K RMSNorm either */
        cfg->image_token_id = DS_TOKEN_IMAGE_START; /* V2 uses image_start token */
        cfg->image_size_crop = 640;
    } else {
        /* V1: CLIP ViT-L/14 */
        cfg->enc_type = 1;
        cfg->enc_layers = DS_CLIP_LAYERS;
        cfg->enc_hidden = DS_CLIP_HIDDEN;
        cfg->enc_heads = DS_CLIP_HEADS;
        cfg->enc_kv_heads = DS_CLIP_HEADS;
        cfg->enc_head_dim = DS_CLIP_HEAD_DIM;
        cfg->enc_intermediate = DS_CLIP_MLP_DIM;
        cfg->enc_causal_flow_queries = 0;
        cfg->proj_input_dim = DS_PROJECTOR_V1_INPUT; /* 2048 */
        cfg->enc_rope_theta = 0;
        cfg->sam_ds2_dim = DS_SAM_DS2_DIM; /* 1024 for V1 */

        cfg->sliding_window_size = 0;
        cfg->has_qk_norm = 0;  /* V1 doesn't have per-head Q/K RMSNorm */
        cfg->image_token_id = DS_TOKEN_IMAGE_START;
        cfg->image_size_crop = 640;
    }
}

/* ========================================================================
 * Weight Loading — 权重加载
 *
 * 【加载策略】
 * 编码器权重 (SAM/CLIP/DeepEncoder): 以 F32 格式加载（预转换，适合批量计算）
 * 解码器权重 (MoE Decoder): 以 BF16 格式零拷贝加载（mmap直接指针，适合逐token解码）
 * 路由器权重 (Gate): 同时保留 BF16 和 F32 两个版本（BF16保精度，F32备用）
 * 归一化权重 (RMSNorm/LayerNorm): 以 F32 格式加载（维度小，精度敏感）
 *
 * 【融合权重构建】
 * 加载完成后，为每个MoE层构建:
 *   1. gate_up_fused: gate和up权重拼接为 [2*moe_inter, hidden]，
 *      推理时只需1次matvec代替2次，更好的缓存复用
 *   2. 连续expert_block: 所有64个expert + shared的gate_up_fused
 *      在一块连续内存中，减少page fault（v0.9优化，2.4× MoE加速）
 *
 * 【加载宏】
 *   LOAD_F32(name, target) — 查找tensor → 转F32 → 存入target
 *   LOAD_BF16(name, target) — 查找tensor → 取BF16直接指针 → 存入target（零拷贝）
 * ======================================================================== */

static int load_all_weights(ds_ctx_t *ctx) {
    multi_safetensors_t *ms = (multi_safetensors_t *)ctx->safetensors;
    if (!ms) return -1;

    ds_config_t *cfg = &ctx->config;
    safetensors_file_t *sf = NULL;
    const safetensor_t *t;

    /* ---- Visual Tokenizer (SAM) Weights ---- */
    ds_visual_tokenizer_t *vt = &ctx->vis_tokenizer;

    #define LOAD_F32(name, target) do { \
        t = multi_safetensors_find(ms, name, &sf); \
        if (t) { \
            target = safetensors_get_f32(sf, t); \
            if (!target) { fprintf(stderr, "Failed to load %s\n", name); return -1; } \
            if (ds_verbose >= 2) fprintf(stderr, "Loaded %s\n", name); \
        } else { \
            if (ds_verbose >= 1) fprintf(stderr, "Warning: tensor %s not found\n", name); \
        } \
    } while(0)

    #define LOAD_F32_OPT(name, target) do { \
        t = multi_safetensors_find(ms, name, &sf); \
        if (t) { \
            target = safetensors_get_f32(sf, t); \
            if (ds_verbose >= 2) fprintf(stderr, "Loaded %s\n", name); \
        } \
    } while(0)

    LOAD_F32("model.sam_model.patch_embed.proj.weight", vt->sam_patch_embed_weight);
    LOAD_F32("model.sam_model.patch_embed.proj.bias", vt->sam_patch_embed_bias);
    LOAD_F32("model.sam_model.pos_embed", vt->sam_pos_embed);

    /* SAM transformer layers (12 layers) */
    for (int l = 0; l < 12; l++) {
        char name[256];
        #define SAM_WEIGHT(field, weight_name) do { \
            snprintf(name, sizeof(name), "model.sam_model.blocks.%d." weight_name, l); \
            LOAD_F32(name, vt->sam_layers[l].field); \
        } while(0)

        SAM_WEIGHT(norm1_weight, "norm1.weight");
        SAM_WEIGHT(norm1_bias, "norm1.bias");
        SAM_WEIGHT(attn_qkv_weight, "attn.qkv.weight");
        SAM_WEIGHT(attn_qkv_bias, "attn.qkv.bias");
        SAM_WEIGHT(attn_proj_weight, "attn.proj.weight");
        SAM_WEIGHT(attn_proj_bias, "attn.proj.bias");
        SAM_WEIGHT(rel_pos_h, "attn.rel_pos_h");
        SAM_WEIGHT(rel_pos_w, "attn.rel_pos_w");
        SAM_WEIGHT(norm2_weight, "norm2.weight");
        SAM_WEIGHT(norm2_bias, "norm2.bias");
        SAM_WEIGHT(mlp_lin1_weight, "mlp.lin1.weight");
        SAM_WEIGHT(mlp_lin1_bias, "mlp.lin1.bias");
        SAM_WEIGHT(mlp_lin2_weight, "mlp.lin2.weight");
        SAM_WEIGHT(mlp_lin2_bias, "mlp.lin2.bias");
    }

    /* SAM neck — V1/V2/V3 all use neck.{0,1,2,3}.{weight,bias} naming.
     * Conv biases (neck.0.bias, neck.2.bias) may not exist in any version. */
    LOAD_F32("model.sam_model.neck.0.weight", vt->sam_neck_conv1_weight);
    LOAD_F32("model.sam_model.neck.1.weight", vt->sam_neck_ln1_weight);
    LOAD_F32("model.sam_model.neck.1.bias", vt->sam_neck_ln1_bias);
    LOAD_F32("model.sam_model.neck.2.weight", vt->sam_neck_conv2_weight);
    LOAD_F32("model.sam_model.neck.3.weight", vt->sam_neck_ln2_weight);
    LOAD_F32("model.sam_model.neck.3.bias", vt->sam_neck_ln2_bias);
    /* Conv biases — try to load, may not exist in V1/V2/V3 */
    LOAD_F32_OPT("model.sam_model.neck.0.bias", vt->sam_neck_conv1_bias);
    LOAD_F32_OPT("model.sam_model.neck.2.bias", vt->sam_neck_conv2_bias);

    /* SAM downsample — all versions use net_2/net_3 (no .0 suffix, no bias) */
    LOAD_F32("model.sam_model.net_2.weight", vt->sam_net2_weight);
    LOAD_F32("model.sam_model.net_3.weight", vt->sam_net3_weight);

    /* Learnable tokens — image_newline is V1 only, view_seperator is shared */
    LOAD_F32_OPT("model.image_newline", vt->image_newline);
    LOAD_F32("model.view_seperator", vt->view_seperator);

    /* ---- CLIP Encoder Weights (V1 only) ---- */
    if (cfg->enc_type == 1) {
        ds_clip_encoder_t *clip = &ctx->clip_encoder;

        LOAD_F32("model.vision_model.embeddings.class_embedding", clip->class_embedding);
        LOAD_F32("model.vision_model.embeddings.patch_embedding.weight", clip->patch_embedding_weight);
        LOAD_F32("model.vision_model.embeddings.position_embedding.weight", clip->position_embedding);
        LOAD_F32("model.vision_model.pre_layrnorm.weight", clip->pre_layernorm_weight);
        LOAD_F32("model.vision_model.pre_layrnorm.bias", clip->pre_layernorm_bias);

        for (int l = 0; l < DS_CLIP_LAYERS; l++) {
            char name[256];
            #define CLIP_WEIGHT(field, weight_name) do { \
                snprintf(name, sizeof(name), "model.vision_model.transformer.layers.%d." weight_name, l); \
                LOAD_F32(name, clip->layers[l].field); \
            } while(0)

            CLIP_WEIGHT(layer_norm1_weight, "layer_norm1.weight");
            CLIP_WEIGHT(layer_norm1_bias, "layer_norm1.bias");
            CLIP_WEIGHT(qkv_proj_weight, "self_attn.qkv_proj.weight");
            CLIP_WEIGHT(qkv_proj_bias, "self_attn.qkv_proj.bias");
            CLIP_WEIGHT(out_proj_weight, "self_attn.out_proj.weight");
            CLIP_WEIGHT(out_proj_bias, "self_attn.out_proj.bias");
            CLIP_WEIGHT(layer_norm2_weight, "layer_norm2.weight");
            CLIP_WEIGHT(layer_norm2_bias, "layer_norm2.bias");
            CLIP_WEIGHT(mlp_fc1_weight, "mlp.fc1.weight");
            CLIP_WEIGHT(mlp_fc1_bias, "mlp.fc1.bias");
            CLIP_WEIGHT(mlp_fc2_weight, "mlp.fc2.weight");
            CLIP_WEIGHT(mlp_fc2_bias, "mlp.fc2.bias");
        }

        /* Post-LayerNorm — V1 and V3 don't have this, V2 (DeepEncoder) skips CLIP entirely */
        LOAD_F32_OPT("model.vision_model.post_layernorm.weight", clip->final_norm_weight);
        LOAD_F32_OPT("model.vision_model.post_layernorm.bias", clip->final_norm_bias);
    }

    /* ---- DeepEncoder V2 Weights (V2 only) ---- */
    if (cfg->enc_type == 2) {
        ds_deep_encoder_t *enc = &ctx->encoder;

        /* V2 encoder prefix: model.qwen2_model.model.model.layers.* */
        const char *enc_prefix = "model.qwen2_model.model.model";

        for (int l = 0; l < cfg->enc_layers; l++) {
            char name[256];
            #define ENC_WEIGHT(field, weight_name) do { \
                snprintf(name, sizeof(name), "%s.layers.%d." weight_name, enc_prefix, l); \
                LOAD_F32(name, enc->layers[l].field); \
            } while(0)

            ENC_WEIGHT(layer_norm1_weight, "input_layernorm.weight");
            ENC_WEIGHT(wq_weight, "self_attn.q_proj.weight");
            ENC_WEIGHT(wk_weight, "self_attn.k_proj.weight");
            ENC_WEIGHT(wv_weight, "self_attn.v_proj.weight");
            ENC_WEIGHT(wo_weight, "self_attn.o_proj.weight");
            ENC_WEIGHT(wq_bias, "self_attn.q_proj.bias");
            ENC_WEIGHT(wk_bias, "self_attn.k_proj.bias");
            ENC_WEIGHT(wv_bias, "self_attn.v_proj.bias");
            ENC_WEIGHT(layer_norm2_weight, "post_attention_layernorm.weight");
            ENC_WEIGHT(gate_weight, "mlp.gate_proj.weight");
            ENC_WEIGHT(up_weight, "mlp.up_proj.weight");
            ENC_WEIGHT(down_weight, "mlp.down_proj.weight");
        }

        char enc_norm_name[256];
        snprintf(enc_norm_name, sizeof(enc_norm_name), "%s.norm.weight", enc_prefix);
        LOAD_F32(enc_norm_name, enc->final_norm_weight);

        /* V2 causal flow query embeddings */
        LOAD_F32("model.qwen2_model.query_1024.weight", vt->causal_query_embeddings);
        /* query_768 is for 768x768 resolution (144 tokens) */
        LOAD_F32("model.qwen2_model.query_768.weight", vt->causal_query_768_embeddings);
    }

    /* ---- Projector Weights ---- */
    LOAD_F32("model.projector.layers.weight", ctx->projector.weight);
    LOAD_F32("model.projector.layers.bias", ctx->projector.bias);

    /* ---- Decoder Weights (BF16) ---- */
    ds_moe_decoder_t *dec = &ctx->decoder;

    #define LOAD_BF16(name, target) do { \
        t = multi_safetensors_find(ms, name, &sf); \
        if (t && safetensor_is_bf16(t)) { \
            target = safetensors_get_bf16_direct(sf, t); \
            if (!target) { fprintf(stderr, "Failed to load %s\n", name); return -1; } \
            if (ds_verbose >= 2) fprintf(stderr, "Loaded %s (bf16)\n", name); \
        } else if (t) { \
            fprintf(stderr, "Warning: %s is not BF16, loading as F32\n", name); \
        } else { \
            if (ds_verbose >= 1) fprintf(stderr, "Warning: tensor %s not found\n", name); \
        } \
    } while(0)

    LOAD_BF16("model.embed_tokens.weight", dec->tok_embeddings_bf16);

    for (int l = 0; l < cfg->dec_layers; l++) {
        char name[256];
        ds_dec_layer_t *layer = &dec->layers[l];

        #define DEC_BF16(field, weight_name) do { \
            snprintf(name, sizeof(name), "model.layers.%d." weight_name, l); \
            LOAD_BF16(name, layer->field); \
        } while(0)

        #define DEC_F32(field, weight_name) do { \
            snprintf(name, sizeof(name), "model.layers.%d." weight_name, l); \
            LOAD_F32(name, layer->field); \
        } while(0)

        DEC_BF16(wq_weight_bf16, "self_attn.q_proj.weight");
        DEC_BF16(wk_weight_bf16, "self_attn.k_proj.weight");
        DEC_BF16(wv_weight_bf16, "self_attn.v_proj.weight");
        DEC_BF16(wo_weight_bf16, "self_attn.o_proj.weight");

        /* Q/K norm weights — only present in V1/V2, not Unlimited-OCR (has_qk_norm=0) */
        if (cfg->has_qk_norm) {
            DEC_F32(q_norm_weight, "self_attn.q_norm.weight");
            DEC_F32(k_norm_weight, "self_attn.k_norm.weight");
        }
        DEC_F32(input_norm, "input_layernorm.weight");
        DEC_F32(post_attn_norm, "post_attention_layernorm.weight");

        if (l < cfg->dec_first_k_dense) {
            /* Dense FFN for first K layers */
            DEC_BF16(dense_gate_weight_bf16, "mlp.gate_proj.weight");
            DEC_BF16(dense_up_weight_bf16, "mlp.up_proj.weight");
            DEC_BF16(dense_down_weight_bf16, "mlp.down_proj.weight");
        } else {
            /* MoE for remaining layers */
            DEC_F32(gate_weight, "mlp.gate.weight");
            DEC_BF16(gate_weight_bf16, "mlp.gate.weight");

            /* Routed experts */
            for (int e = 0; e < cfg->dec_n_routed_experts; e++) {
                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
                LOAD_BF16(name, layer->experts[e].gate_weight_bf16);

                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.up_proj.weight", l, e);
                LOAD_BF16(name, layer->experts[e].up_weight_bf16);

                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.down_proj.weight", l, e);
                LOAD_BF16(name, layer->experts[e].down_weight_bf16);
            }

            /* Shared experts */
            DEC_BF16(shared_gate_weight_bf16, "mlp.shared_experts.gate_proj.weight");
            DEC_BF16(shared_up_weight_bf16, "mlp.shared_experts.up_proj.weight");
            DEC_BF16(shared_down_weight_bf16, "mlp.shared_experts.down_proj.weight");
        }
    }

    LOAD_F32("model.norm.weight", dec->norm);
    LOAD_BF16("lm_head.weight", dec->lm_head_bf16);

    #undef LOAD_F32
    #undef LOAD_BF16
    #undef SAM_WEIGHT
    #undef CLIP_WEIGHT
    #undef ENC_WEIGHT
    #undef DEC_BF16
    #undef DEC_F32

    /* ---- Build fused gate+up weights for decode-path expert forward ----
     * Each expert's gate and up projections share the same input x,
     * so concatenating [gate_W; up_W] into a single [2*inter, hidden] matrix
     * allows one matvec instead of two — better L2 cache reuse of x, fewer
     * thread dispatches. This is only used for single-token decode;
     * prefill uses the separate gate/up weights for batched sgemm.
     *
     * Contiguous block layout: all experts' gate_up_fused + shared gate_up_fused
     * are packed into ONE allocation per layer for better page-in locality.
     * When the OS pages in expert N's weights, expert N+1 is already adjacent
     * in virtual memory, reducing page faults from random expert access. */
    {
        int moe_inter = cfg->dec_moe_inter;
        int n_experts = cfg->dec_n_routed_experts;
        int n_shared = cfg->dec_n_shared_experts;
        int shared_inter = n_shared * moe_inter;
        int dec_hidden = cfg->dec_hidden;
        size_t fused_per_expert = (size_t)2 * moe_inter * dec_hidden;  /* BF16 elements */
        size_t shared_fused_size = (size_t)2 * shared_inter * dec_hidden;  /* BF16 elements */

        for (int l = cfg->dec_first_k_dense; l < cfg->dec_layers; l++) {
            ds_dec_layer_t *layer = &dec->layers[l];

            /* Single contiguous block:
             *   [expert 0 gate_up_fused | expert 1 gate_up_fused | ... | expert N-1 gate_up_fused | shared gate_up_fused]
             * Total elements = n_experts * fused_per_expert + shared_fused_size */
            size_t block_elements = (size_t)n_experts * fused_per_expert + shared_fused_size;
            size_t block_bytes = block_elements * sizeof(uint16_t);
            uint16_t *block = (uint16_t *)malloc(block_bytes);
            if (!block) {
                fprintf(stderr, "Failed to allocate expert block for layer %d (%.1f MB)\n",
                        l, block_bytes / 1048576.0);
                return -1;
            }
            layer->expert_block_bf16 = block;
            layer->expert_block_size = block_bytes;

            /* Fill routed experts: gate_up_fused for each */
            uint16_t *ptr = block;
            for (int e = 0; e < n_experts; e++) {
                if (layer->experts[e].gate_weight_bf16 && layer->experts[e].up_weight_bf16) {
                    /* Layout: [gate_W(moe_inter, hidden); up_W(moe_inter, hidden)]
                     * Row-major: gate rows 0..moe_inter-1, then up rows 0..moe_inter-1 */
                    memcpy(ptr, layer->experts[e].gate_weight_bf16,
                           (size_t)moe_inter * dec_hidden * sizeof(uint16_t));
                    memcpy(ptr + (size_t)moe_inter * dec_hidden,
                           layer->experts[e].up_weight_bf16,
                           (size_t)moe_inter * dec_hidden * sizeof(uint16_t));
                    layer->experts[e].gate_up_fused_bf16 = ptr;
                }
                ptr += fused_per_expert;
            }

            /* Fill shared experts: gate_up_fused at the end of the block */
            if (layer->shared_gate_weight_bf16 && layer->shared_up_weight_bf16) {
                memcpy(ptr, layer->shared_gate_weight_bf16,
                       (size_t)shared_inter * dec_hidden * sizeof(uint16_t));
                memcpy(ptr + (size_t)shared_inter * dec_hidden,
                       layer->shared_up_weight_bf16,
                       (size_t)shared_inter * dec_hidden * sizeof(uint16_t));
                layer->shared_gate_up_fused_bf16 = ptr;
            }
        }

        if (ds_verbose >= 1) {
            size_t total_fused = 0;
            for (int l = cfg->dec_first_k_dense; l < cfg->dec_layers; l++) {
                total_fused += dec->layers[l].expert_block_size;
            }
            fprintf(stderr, "Contiguous expert blocks: %.1f MB allocated (%d experts × %d layers + shared, 1 block/layer)\n",
                    total_fused / 1048576.0, n_experts, cfg->dec_layers - cfg->dec_first_k_dense);
        }
    }

    /* ── INT8 quantization (optional, enabled by --int4) ── */
    if (ds_verbose >= 1)
        fprintf(stderr, "All weights loaded\n");

    return 0;
}

/* ========================================================================
 * INT4 Quantization (called after ds_load, when --int4 flag is set)
 * ======================================================================== */

int ds_quantize_moe_int4(ds_ctx_t *ctx) {
    if (!ctx || !ctx->int4_enabled) return 0;
    ds_config_t *cfg = &ctx->config;
    int moe_inter = cfg->dec_moe_inter;
    int hidden = cfg->dec_hidden;
    int n_shared = cfg->dec_n_shared_experts;
    int n_exp = cfg->dec_n_routed_experts;
    double t0 = ds_time_sec();

    for (int l = 0; l < cfg->dec_layers; l++) {
        ds_dec_layer_t *ly = &ctx->decoder.layers[l];
        if (l < cfg->dec_first_k_dense) {
            /* Dense FFN (layer 0): skip */
        } else {
            for (int e = 0; e < n_exp; e++) {
                if (ly->experts[e].gate_up_fused_bf16) {
                    ds_int4_quantize_bf16(&ly->experts_int4[e].gate_up_fused,
                                          ly->experts[e].gate_up_fused_bf16,
                                          2 * moe_inter, hidden);
                }
                if (ly->experts[e].down_weight_bf16) {
                    ds_int4_quantize_bf16(&ly->experts_int4[e].down_weight,
                                          ly->experts[e].down_weight_bf16,
                                          hidden, moe_inter);
                }
            }
            if (ly->shared_gate_up_fused_bf16) {
                ds_int4_quantize_bf16(&ly->shared_gate_up_int4,
                                      ly->shared_gate_up_fused_bf16,
                                      2 * n_shared * moe_inter, hidden);
            }
            if (ly->shared_down_weight_bf16) {
                ds_int4_quantize_bf16(&ly->shared_down_int4,
                                      ly->shared_down_weight_bf16,
                                      hidden, n_shared * moe_inter);
            }
        }
        ly->int4_enabled = 1;
    }

    double dt = ds_time_sec() - t0;
    if (ds_verbose >= 1) {
        size_t int4_total = 0;
        for (int l = cfg->dec_first_k_dense; l < cfg->dec_layers; l++) {
            ds_dec_layer_t *ly2 = &ctx->decoder.layers[l];
            for (int e = 0; e < n_exp; e++) {
                int4_total += ly2->experts_int4[e].gate_up_fused.bytes;
                int4_total += ly2->experts_int4[e].down_weight.bytes;
            }
            int4_total += ly2->shared_gate_up_int4.bytes;
            int4_total += ly2->shared_down_int4.bytes;
        }
        size_t bf16_total = 0;
        for (int l = cfg->dec_first_k_dense; l < cfg->dec_layers; l++)
            bf16_total += ctx->decoder.layers[l].expert_block_size;
        fprintf(stderr, "INT8 quantization: %.2fs, %zu MB INT8 vs %.1f MB BF16 (%.1fx smaller)\n",
                dt, int4_total / 1048576, bf16_total / 1048576.0,
                bf16_total > 0 ? (double)bf16_total / int4_total : 0.0);
    }
    return 0;
}

/* ========================================================================
 * Context Allocation — 上下文缓冲区分配
 *
 * 【KV缓存】
 * 布局: [layers, max_seq, kv_dim]，F32格式，cache-line 64字节对齐
 * 为什么用F32而非BF16存储?
 *   之前用BF16存储，但每步decode都要将整个KV缓存BF16→F32转换
 *   对于长序列(1000+ tokens)，每层每步转换1000+元素，12层×2(K和V) = 24000+次转换
 *   这完全占据了注意力计算时间！
 *   改为F32存储后: 内存翻倍(240MB→480MB)，但消除所有转换开销
 *
 * 【行对齐】
 * kv_row_stride = (kv_dim + 15) & ~15 — 将每行长度向上取整到16个float(64字节)
 * 这样每行起始地址对齐到cache line，注意力计算时内存访问更高效
 *
 * 【预分配复用缓冲区】
 * 所有decode步骤共享同一组缓冲区（gate_buf, up_buf等），避免反复malloc/free
 * 这是安全的，因为MoE专家串行处理，不同专家可复用同一块内存
 *
 * 【RoPE预计算】
 * 一次性计算所有位置的cos/sin值（4096个位置），decode时直接查表
 * 比每步重新计算快得多，且结果完全精确
 * ======================================================================== */

static int alloc_decoder_buffers(ds_ctx_t *ctx) {
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int intermediate = cfg->dec_intermediate;

    /* KV cache: [layers, max_seq, kv_dim] stored as F32, cache-line aligned.
     * Using F32 directly avoids the per-step BF16→F32 batch conversion that
     * previously dominated attention time (reconverting entire cache each step).
     * Memory is 2x larger but the bandwidth savings from eliminating conversion
     * and the sequential read performance gain more than compensate. */
    int max_seq = 4096;
    ctx->kv_cache_max = max_seq;
    ctx->kv_cache_len = 0;

    /* Each row: kv_dim floats. Align each layer's cache start to 64 bytes.
     * Row stride = kv_dim rounded up to multiple of 16 floats (64 bytes). */
    int kv_row_stride = (kv_dim + 15) & ~15;  /* aligned row stride */
    size_t kv_layer_size = (size_t)max_seq * kv_row_stride * sizeof(float);
    size_t kv_total = (size_t)cfg->dec_layers * kv_layer_size;

    /* posix_memalign for 64-byte alignment (cache line) */
    if (posix_memalign((void **)&ctx->kv_cache_k, 64, kv_total) != 0) return -1;
    if (posix_memalign((void **)&ctx->kv_cache_v, 64, kv_total) != 0) return -1;
    memset(ctx->kv_cache_k, 0, kv_total);
    memset(ctx->kv_cache_v, 0, kv_total);
    ctx->_kv_row_stride = kv_row_stride;  /* store for attention access */


    /* Single-token decoder buffers */
    ctx->dec_x = (float *)malloc(hidden * sizeof(float));
    ctx->dec_x_norm = (float *)malloc(hidden * sizeof(float));
    ctx->dec_q = (float *)malloc(n_heads * head_dim * sizeof(float));
    ctx->dec_k = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(n_heads * head_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(hidden * sizeof(float));
    ctx->dec_expert_out = (float *)malloc(hidden * sizeof(float));
    ctx->dec_shared_out = (float *)malloc(hidden * sizeof(float));
    ctx->dec_layer_out = (float *)malloc(hidden * sizeof(float));
    ctx->dec_gate_scores = (float *)malloc(cfg->dec_n_routed_experts * sizeof(float));

    /* Repetition penalty: logits buffer and token history */
    ctx->dec_logits = (float *)malloc(cfg->vocab_size * sizeof(float));
    ctx->token_history_cap = max_seq + 512;
    ctx->token_history = (int *)malloc(ctx->token_history_cap * sizeof(int));
    ctx->token_history_len = 0;

    /* Dense FFN buffers (for layer 0 and any layer < first_k_dense) */
    if (cfg->dec_first_k_dense > 0) {
        ctx->dec_dense_gate = (float *)malloc(intermediate * sizeof(float));
        ctx->dec_dense_up = (float *)malloc(intermediate * sizeof(float));
        ctx->dec_dense_swiglu = (float *)malloc(intermediate * sizeof(float));
        ctx->dec_dense_out = (float *)malloc(hidden * sizeof(float));
    }

    /* MoE scratch buffers (reused across all decode steps) */
    {
        int moe_inter = cfg->dec_moe_inter;
        int shared_inter = cfg->dec_n_shared_experts * moe_inter;
        int top_k = cfg->dec_top_k;

        ctx->moe_expert_gate_buf = (float *)malloc(moe_inter * sizeof(float));
        ctx->moe_expert_up_buf = (float *)malloc(moe_inter * sizeof(float));
        ctx->moe_expert_gate_up_buf = (float *)malloc(2 * moe_inter * sizeof(float));
        ctx->moe_expert_hidden_buf = (float *)malloc(moe_inter * sizeof(float));
        ctx->moe_expert_outputs = (float *)malloc(top_k * hidden * sizeof(float));

        ctx->moe_shared_gate_buf = (float *)malloc(shared_inter * sizeof(float));
        ctx->moe_shared_up_buf = (float *)malloc(shared_inter * sizeof(float));
        ctx->moe_shared_gate_up_buf = (float *)malloc(2 * shared_inter * sizeof(float));
        ctx->moe_shared_swiglu_buf = (float *)malloc(shared_inter * sizeof(float));
        ctx->moe_shared_out_buf = (float *)malloc(hidden * sizeof(float));
    }

    /* RoPE cache */
    ctx->rope_inv_freq_half = head_dim / 2;
    ctx->rope_inv_freq = (float *)malloc(ctx->rope_inv_freq_half * sizeof(float));
    for (int i = 0; i < ctx->rope_inv_freq_half; i++) {
        ctx->rope_inv_freq[i] = 1.0f / powf(cfg->dec_rope_theta,
                                              (float)(2 * i) / (float)head_dim);
    }

    /* Precompute RoPE cache for all positions */
    ctx->rope_cache_cap = max_seq;
    ctx->rope_cache_cos = (float *)malloc(max_seq * head_dim * sizeof(float));
    ctx->rope_cache_sin = (float *)malloc(max_seq * head_dim * sizeof(float));
    int *positions = (int *)malloc(max_seq * sizeof(int));
    for (int i = 0; i < max_seq; i++) positions[i] = i;
    ds_compute_rope_neox(ctx->rope_cache_cos, ctx->rope_cache_sin,
                          positions, max_seq, head_dim, cfg->dec_rope_theta);
    free(positions);

    /* Encoder output buffer */
    int max_tokens = DS_VISUAL_TOKENS_BASE + DS_MAX_LOCAL_CROPS * DS_LOCAL_CROP_TOKENS;
    ctx->enc_output = (float *)malloc(max_tokens * hidden * sizeof(float));

    /* Default settings */
    ctx->max_new_tokens = 4096;
    ctx->temperature = 0.0f; /* Greedy by default */
    ctx->repeat_penalty = 1.0f; /* No penalty by default */
    ctx->no_repeat_ngram_size = 0; /* Disabled by default; ngram blocking causes premature EOS in OCR.
                                      Use --ngram N to enable (Python: 20 non-eval, 35 eval) */
    ctx->min_new_tokens = (cfg->model_version == 3) ? 32 : 0;
    /* V3 needs min_new_tokens because the model's F32 logits make EOS
     * dominate early decode steps. The 32-step warmup with EOS+Ġ ban
     * lets the model settle into stable content generation.
     * After warmup, the model's context window has enough content to
     * generate naturally without reverting to EOS. */

    return 0;
}

/* ========================================================================
 * Public API — 公共API函数
 *
 * ds_load()     — 加载模型（检测版本→打开权重→加载→分配缓冲区→初始化Metal）
 * ds_free()     — 释放所有资源（Metal→safetensors→KV缓存→解码缓冲区→expert块→ctx）
 * ds_recognize()  — 从图像文件路径识别文字
 * ds_recognize_image() — 从RGB像素识别文字（核心实现）
 * ======================================================================== */

ds_ctx_t *ds_load(const char *model_dir) {
    if (ds_verbose >= 1)
        fprintf(stderr, "Loading DeepSeek-OCR model from %s\n", model_dir);

    /* Detect model version */
    int version = detect_model_version(model_dir);
    if (version < 0) {
        fprintf(stderr, "Failed to detect model version\n");
        return NULL;
    }
    if (ds_verbose >= 1)
        fprintf(stderr, "Detected DeepSeek-OCR version %d\n", version);

    /* Allocate context */
    ds_ctx_t *ctx = (ds_ctx_t *)calloc(1, sizeof(ds_ctx_t));
    if (!ctx) return NULL;

    /* Initialize config */
    init_config(&ctx->config, version);
    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);
    snprintf(ctx->vis_tokenizer.model_dir, sizeof(ctx->vis_tokenizer.model_dir), "%s", model_dir);

    /* Open safetensors files */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "Failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Load weights */
    if (load_all_weights(ctx) != 0) {
        fprintf(stderr, "Failed to load model weights\n");
        ds_free(ctx);
        return NULL;
    }

    /* Allocate decoder buffers */
    if (alloc_decoder_buffers(ctx) != 0) {
        fprintf(stderr, "Failed to allocate decoder buffers\n");
        ds_free(ctx);
        return NULL;
    }

    /* Initialize Metal GPU acceleration (optional, no error if unavailable) */
    ctx->metal_ctx = ds_metal_init();
    ctx->metal_enabled = (ctx->metal_ctx && ds_metal_is_available(ctx->metal_ctx)) ? 1 : 0;
    if (ctx->metal_enabled && getenv("DS_NO_METAL")) {
        ctx->metal_enabled = 0;
        if (ds_verbose >= 1) fprintf(stderr, "Metal GPU acceleration: disabled by DS_NO_METAL\n");
    } else if (ctx->metal_enabled) {
        /* Register expert weight blocks with Metal for zero-copy offset access */
        for (int l = 0; l < ctx->config.dec_layers; l++) {
            if (ctx->decoder.layers[l].expert_block_bf16) {
                ds_metal_register_expert_block(ctx->metal_ctx,
                    ctx->decoder.layers[l].expert_block_bf16,
                    ctx->decoder.layers[l].expert_block_size);
            }
        }
        if (ds_verbose >= 1)
            fprintf(stderr, "Metal GPU acceleration: ENABLED\n");
    } else if (ds_verbose >= 1)
        fprintf(stderr, "Metal GPU acceleration: disabled (CPU only)\n");

    if (ds_verbose >= 1)
        fprintf(stderr, "Model loaded successfully (version %d, %d layers)\n",
                version, ctx->config.dec_layers);

    return ctx;
}

void ds_free(ds_ctx_t *ctx) {
    if (!ctx) return;

    /* Free Metal GPU context */
    if (ctx->metal_ctx) {
        ds_metal_free(ctx->metal_ctx);
        ctx->metal_ctx = NULL;
    }

    /* Free safetensors */
    if (ctx->safetensors) {
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    }

    /* Free decoder buffers (posix_memalign requires free(), not custom allocator) */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    free(ctx->dec_x); free(ctx->dec_x_norm);
    free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_expert_out); free(ctx->dec_shared_out);
    free(ctx->dec_gate_scores);
    free(ctx->dec_dense_gate); free(ctx->dec_dense_up);
    free(ctx->dec_dense_swiglu); free(ctx->dec_dense_out);
    free(ctx->rope_inv_freq);
    free(ctx->rope_cache_cos); free(ctx->rope_cache_sin);
    free(ctx->enc_output);
    free(ctx->lm_head_f32);
    free(ctx->tok_emb_f32);

    /* Free contiguous expert blocks (malloc'd, not mmap'd).
     * All experts' gate_up_fused + shared gate_up_fused are in one block per layer. */
    {
        ds_config_t *cfg = &ctx->config;
        ds_moe_decoder_t *dec = &ctx->decoder;
        for (int l = cfg->dec_first_k_dense; l < cfg->dec_layers; l++) {
            ds_dec_layer_t *layer = &dec->layers[l];
            free(layer->expert_block_bf16);
            layer->expert_block_bf16 = NULL;
            layer->expert_block_size = 0;
            /* Nullify pointers into the freed block */
            for (int e = 0; e < cfg->dec_n_routed_experts; e++)
                layer->experts[e].gate_up_fused_bf16 = NULL;
            layer->shared_gate_up_fused_bf16 = NULL;
        }
    }

    /* Note: visual tokenizer, encoder, and decoder weight pointers
     * point into mmap'd safetensors data — they are freed when
     * safetensors is closed. F32 copies need explicit free. */

    free(ctx);
}

void ds_set_token_callback(ds_ctx_t *ctx, ds_token_cb cb, void *userdata) {
    if (!ctx) return;
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

char *ds_recognize(ds_ctx_t *ctx, const char *image_path) {
    if (!ctx || !image_path) return NULL;

    /* Load image */
    ds_image_t *img = ds_image_load(image_path);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        return NULL;
    }

    char *result = ds_recognize_image(ctx, img->pixels, img->width, img->height, img->channels);
    ds_image_free(img);
    return result;
}

char *ds_recognize_image(ds_ctx_t *ctx, const unsigned char *pixels,
                          int width, int height, int channels) {
    if (!ctx || !pixels) return NULL;

    double t0 = now_ms();
    double encode_start = t0;
    ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden;

    /* Reset per-run V3 streaming filter state */
    ctx->_det_buf_len = 0;
    ctx->_det_buf[0] = '\0';
    ctx->_v3_prefix_done = 0;

    /* Initialize dump directory if DS_DUMP_TENSORS is set */
    ds_dump_init();

    /* ---- V2: Multi-crop encoding ----
     * Python flow:
     *   dynamic_preprocess → P patches (768x768) + 1 thumbnail (1024x1024)
     *   for each crop:   SAM(768x768) → Qwen2(144 queries) → Projector → 144 tokens
     *   for global image: SAM(1024x1024) → Qwen2(256 queries) → Projector → 256 tokens
     *   concat: [local(P*144), global(256), view_separator(1)]
     */

    int n_encoder_tokens = 0;
    float *encoder_output = NULL;  /* [n_encoder_tokens, dec_hidden] */
    char tokenizer_path[4096];
    const char *model_dir_str = ctx->model_dir ? ctx->model_dir : ".";
    ds_tokenizer_t *tokenizer = NULL;

    if (cfg->model_version == 2 && cfg->enc_type != 1) {
        /* V2 multi-crop path (DeepEncoder V2, crop_size=768) */
        ds_image_t img = { .pixels = (unsigned char *)pixels, .width = width, .height = height, .channels = channels };

        /* Fast-path: skip encoding entirely and load from Python reference dump */
        if (getenv("DS_SKIP_ENCODER")) {
            const char *npy_path = "dump/multicrop/full_proj_output.npy";
            FILE *f = fopen(npy_path, "rb");
            if (f) {
                unsigned char buf[10];
                fread(buf, 1, 10, f);
                uint16_t header_len = *(uint16_t *)(buf + 8);
                fseek(f, 10 + header_len, SEEK_SET);
                n_encoder_tokens = 1121;
                encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));
                fread(encoder_output, sizeof(float), n_encoder_tokens * hidden, f);
                fclose(f);
                if (ds_verbose >= 1)
                    fprintf(stderr, "SKIP_ENCODER: loaded %d tokens from %s (no SAM/encoder compute)\n",
                            n_encoder_tokens, npy_path);
                goto prompt_construction;
            } else {
                fprintf(stderr, "DS_SKIP_ENCODER: %s not found, falling back to encoding\n", npy_path);
            }
        }

        /* Step 1: Dynamic preprocess — generate crops + thumbnail
         * For small images (both dims <= 768), dynamic_preprocess returns 1 image padded to 768.
         * We still need a 1024x1024 global image for the encoder, so we handle this as
         * "no crop" case: just encode the global image at 1024x1024.
         * For larger images, dynamic_preprocess returns N crops (768x768) + 1 thumbnail (1024x1024).
         */
        int n_crops = 0;
        int use_crop = (width > 768 || height > 768);
        ds_image_t **crops = NULL;

        /* Optional: load PIL-preprocessed pixels from bin files, bypassing C resize.
         * DS_LOAD_PIL_PIXELS=1 loads:
         *   dump/py_pil_local_crops.bin  (n_crops x 3 x 768 x 768 float32 CHW, [0,1])
         *   dump/py_pil_global_view.bin  (3 x 1024 x 1024 float32 CHW, [0,1]) */
        const char *pil_pixels = getenv("DS_LOAD_PIL_PIXELS");
        if (pil_pixels && use_crop) {
            int pil_n_crops = 6; /* fixed for test截屏.png */
            int pil_local_size = 768;
            int pil_global_size = 1024;

            /* Load local crops from bin: 6 x 3 x 768 x 768 */
            int local_total = pil_n_crops * 3 * pil_local_size * pil_local_size;
            float *local_chw = (float *)malloc(local_total * sizeof(float));
            FILE *f = fopen("dump/py_pil_local_crops.bin", "rb");
            if (!f) { fprintf(stderr, "Cannot open py_pil_local_crops.bin\n"); free(local_chw); pil_pixels = NULL; }
            else {
                fread(local_chw, sizeof(float), local_total, f);
                fclose(f);

                crops = (ds_image_t **)malloc(pil_n_crops * sizeof(ds_image_t *));
                for (int ci = 0; ci < pil_n_crops; ci++) {
                    crops[ci] = (ds_image_t *)malloc(sizeof(ds_image_t));
                    crops[ci]->width = pil_local_size;
                    crops[ci]->height = pil_local_size;
                    crops[ci]->channels = 3;
                    crops[ci]->owns_stb = 0;
                    int npix = pil_local_size * pil_local_size;
                    crops[ci]->pixels = (unsigned char *)malloc(npix * 3);
                    /* Convert CHW [0,1] float -> HWC uint8 */
                    float *crop_chw = local_chw + ci * 3 * npix;
                    for (int p = 0; p < npix; p++) {
                        for (int c = 0; c < 3; c++) {
                            float v = crop_chw[c * npix + p];
                            int iv = (int)(v * 255.0f + 0.5f);
                            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
                            crops[ci]->pixels[p * 3 + c] = (unsigned char)iv;
                        }
                    }
                }
                n_crops = pil_n_crops;
                free(local_chw);
                if (ds_verbose >= 1)
                    fprintf(stderr, "PIL_PIXELS: loaded %d local crops (768x768) from dump/\n", n_crops);
            }
        }

        if (use_crop && !pil_pixels) {
            crops = ds_dynamic_preprocess(&img, 768, 2, 6, 0, &n_crops);
            if (!crops || n_crops < 1) {
                fprintf(stderr, "dynamic_preprocess failed\n");
                return NULL;
            }
        }

        if (!use_crop) {
            /* Small image (both dims <= 768): Python only processes the global view
             * at 1024x1024. No local crops. tokenized_image = 256+1 = 257 tokens.
             * global_local_features = [global_features(256), view_seperator(1)] */
            if (ds_verbose >= 1)
                fprintf(stderr, "Small image %dx%d: global only (1024x1024)\n",
                        width, height);

            /* Global: pad to 1024x1024 */
            ds_image_t *global_img = ds_image_pad(&img, 1024, 127);
            if (!global_img) return NULL;

            int n_sam_tokens;
            g_dump_crop_id = 6;  /* Global crop uses ID 6 */
            double t_sam_v2 = ds_time_sec();
            float *global_sam = ds_sam_forward_image(ctx, global_img, &n_sam_tokens, NULL);
            ctx->perf_sam_ms += (ds_time_sec() - t_sam_v2) * 1000.0;
            ds_image_free(global_img);
            if (!global_sam) return NULL;

            /* Override global SAM tokens with Python reference if DS_LOAD_SAM_ALL set */
            {
                const char *load_sam_all = getenv("DS_LOAD_SAM_ALL");
                if (load_sam_all) {
                    char auto_path[512];
                    snprintf(auto_path, sizeof(auto_path), "%s6.bin", load_sam_all);
                    FILE *sf = fopen(auto_path, "rb");
                    if (sf) {
                        int n_read = (int)fread(global_sam, sizeof(float), n_sam_tokens * 896, sf);
                        fclose(sf);
                        fprintf(stderr, "DS_LOAD_SAM: global crop loaded %d floats from %s (expected %d)\n",
                                n_read, auto_path, n_sam_tokens * 896);
                    } else {
                        /* Try python_sam_global.bin as fallback */
                        snprintf(auto_path, sizeof(auto_path), "%sglobal.bin", load_sam_all);
                        FILE *sf2 = fopen(auto_path, "rb");
                        if (sf2) {
                            int n_read = (int)fread(global_sam, sizeof(float), n_sam_tokens * 896, sf2);
                            fclose(sf2);
                            fprintf(stderr, "DS_LOAD_SAM: global crop loaded %d floats from %s (expected %d)\n",
                                    n_read, auto_path, n_sam_tokens * 896);
                        } else {
                            fprintf(stderr, "Warning: DS_LOAD_SAM_ALL global not found (tried crop6 and global)\n");
                        }
                    }
                }
            }

            /* Dump SAM tokens and encoder output for debugging (DS_DUMP_DIR env var) */
            {
                const char *dump_dir = getenv("DS_DUMP_DIR");
                if (dump_dir) {
                    char path[512];
                    int sam_dim = ctx->config.sam_ds2_dim;  /* 896 for V2 */
                    snprintf(path, sizeof(path), "%s/sam_tokens_global.bin", dump_dir);
                    FILE *f = fopen(path, "wb");
                    if (f) { fwrite(global_sam, sizeof(float), n_sam_tokens * sam_dim, f); fclose(f);
                        if (ds_verbose >= 1) fprintf(stderr, "Dumped SAM tokens (%d x %d) to %s\n", n_sam_tokens, sam_dim, path); }
                }
            }

            int n_global_enc;
            double t_enc_v2 = ds_time_sec();
            float *global_enc = ds_encoder_forward_v2(ctx, global_sam, n_sam_tokens,
                                                       &n_global_enc, 256,
                                                       ctx->vis_tokenizer.causal_query_embeddings);
            ctx->perf_encoder_ms += (ds_time_sec() - t_enc_v2) * 1000.0;
            free(global_sam);
            if (!global_enc) {
                fprintf(stderr, "Encoder forward failed for global image\n");
                return NULL;
            }

            /* Concat: [global(256), view_sep(1)] */
            n_encoder_tokens = n_global_enc + 1;
            encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));
            memcpy(encoder_output, global_enc, n_global_enc * hidden * sizeof(float));
            if (ctx->vis_tokenizer.view_seperator) {
                memcpy(encoder_output + n_global_enc * hidden,
                       ctx->vis_tokenizer.view_seperator, hidden * sizeof(float));
            } else {
                memset(encoder_output + n_global_enc * hidden, 0, hidden * sizeof(float));
            }
            free(global_enc);

            /* Dump encoder output for debugging (DS_DUMP_DIR env var) */
            {
                const char *dump_dir = getenv("DS_DUMP_DIR");
                if (dump_dir) {
                    char path[512];
                    snprintf(path, sizeof(path), "%s/encoder_output_global.bin", dump_dir);
                    FILE *f = fopen(path, "wb");
                    if (f) { fwrite(encoder_output, sizeof(float), n_encoder_tokens * hidden, f); fclose(f);
                        if (ds_verbose >= 1) fprintf(stderr, "Dumped encoder output (%d) to %s\n", n_encoder_tokens, path); }
                }
            }
        } else {
            /* Large image: multi-crop encoding */
            if (ds_verbose >= 1)
                fprintf(stderr, "Multi-crop: %d local crops (768x768)\n", n_crops);

            /* Step 2: Encode local crops + global image in parallel.
             * Parallelize ALL crops (local + global) together to overlap
             * the global SAM+Encoder with local SAM+Encoder computations.
             * This saves ~4s (previously global was serial after local). */
            int tokens_per_crop = 144;  /* 48/4 * 48/4 = 12*12 = 144 */
            float *local_features = NULL;
            int local_token_count = 0;

            /* Prepare global image (1024x1024) for parallel encoding */
            if (ds_verbose >= 1)
                fprintf(stderr, "Encoding global image (1024x1024)\n");
            g_dump_crop_id = 6;  /* Global crop uses ID 6 */

            ds_image_t *global_img = NULL;
            if (pil_pixels) {
                /* Load PIL-preprocessed global view */
                int gsize = 1024;
                int gtotal = 3 * gsize * gsize;
                float *g_chw = (float *)malloc(gtotal * sizeof(float));
                FILE *gf = fopen("dump/py_pil_global_view.bin", "rb");
                if (gf) {
                    fread(g_chw, sizeof(float), gtotal, gf);
                    fclose(gf);
                    global_img = (ds_image_t *)malloc(sizeof(ds_image_t));
                    global_img->width = gsize;
                    global_img->height = gsize;
                    global_img->channels = 3;
                    global_img->owns_stb = 0;
                    int gnpix = gsize * gsize;
                    global_img->pixels = (unsigned char *)malloc(gnpix * 3);
                    for (int p = 0; p < gnpix; p++) {
                        for (int c = 0; c < 3; c++) {
                            float v = g_chw[c * gnpix + p];
                            int iv = (int)(v * 255.0f + 0.5f);
                            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
                            global_img->pixels[p * 3 + c] = (unsigned char)iv;
                        }
                    }
                    free(g_chw);
                    if (ds_verbose >= 1)
                        fprintf(stderr, "PIL_PIXELS: loaded global view (1024x1024) from dump/\n");
                } else {
                    fprintf(stderr, "Cannot open py_pil_global_view.bin, using C pad\n");
                    free(g_chw);
                    global_img = ds_image_pad(&img, 1024, 127);
                }
            } else {
                global_img = ds_image_pad(&img, 1024, 127);
            }
            if (!global_img) {
                goto cleanup_crops;
            }

            /* n_parallel = n_crops local + 1 global */
            int n_parallel = n_crops + 1;
            /* Crops already run concurrently — disable SAM's internal window
             * threading to avoid oversubscription (restored after joins). */
            ds_sam_window_parallel = 0;
            crop_task_t *tasks = (crop_task_t *)calloc(n_parallel, sizeof(crop_task_t));
            pthread_t *threads = (pthread_t *)calloc(n_parallel, sizeof(pthread_t));

            /* Launch local crop encoding */
            for (int i = 0; i < n_crops; i++) {
                tasks[i].ctx = ctx;
                tasks[i].crop = crops[i];
                tasks[i].tokens_per_crop = tokens_per_crop;
                if (ds_verbose >= 1)
                    fprintf(stderr, "Encoding local crop %d/%d\n", i + 1, n_crops);
                pthread_create(&threads[i], NULL, crop_worker, &tasks[i]);
            }

            /* Launch global crop encoding (task index = n_crops) */
            {
                int gi = n_crops;
                tasks[gi].ctx = ctx;
                tasks[gi].crop = global_img;
                tasks[gi].tokens_per_crop = 256;  /* 64/4 * 64/4 = 16*16 = 256 */
                /* Global uses causal_query_embeddings (256 queries), not 768 version */
                pthread_create(&threads[gi], NULL, crop_worker_global, &tasks[gi]);
            }

            /* Wait for all tasks and collect results */
            int global_failed = 0;
            float *global_enc_tokens = NULL;
            int n_global_enc_tokens = 0;

            for (int i = 0; i < n_parallel; i++) {
                pthread_join(threads[i], NULL);
                if (tasks[i].failed) {
                    fprintf(stderr, "Encoding failed for crop %d\n", i);
                    if (i < n_crops) {
                        /* Local crop failed — cleanup */
                        for (int j = 0; j < n_parallel; j++) free(tasks[j].enc_tokens);
                        free(tasks); free(threads); free(local_features);
                        ds_image_free(global_img);
                        goto cleanup_crops;
                    } else {
                        /* Global crop failed */
                        global_failed = 1;
                    }
                    continue;
                }
                if (ds_verbose >= 1) {
                    const char *crop_type = (i < n_crops) ? "local" : "global";
                    fprintf(stderr, "  %s crop %d: SAM %.2fs + Encoder %.2fs = %.2fs\n",
                            crop_type, (i < n_crops) ? i + 1 : 0,
                            tasks[i].sam_time, tasks[i].enc_time,
                            tasks[i].sam_time + tasks[i].enc_time);
                }

                if (i < n_crops) {
                    /* Local crop result */
                    int old_count = local_token_count;
                    local_token_count += tasks[i].n_enc_tokens;
                    local_features = (float *)realloc(local_features, local_token_count * hidden * sizeof(float));
                    memcpy(local_features + old_count * hidden, tasks[i].enc_tokens,
                           tasks[i].n_enc_tokens * hidden * sizeof(float));
                    free(tasks[i].enc_tokens);
                } else {
                    /* Global crop result */
                    global_enc_tokens = tasks[i].enc_tokens;
                    n_global_enc_tokens = tasks[i].n_enc_tokens;
                }
            }
            /* Accumulate crop SAM/encoder times for the perf summary */
            for (int i = 0; i < n_parallel; i++) {
                ctx->perf_sam_ms += tasks[i].sam_time * 1000.0;
                ctx->perf_encoder_ms += tasks[i].enc_time * 1000.0;
            }
            free(tasks); free(threads);
            ds_image_free(global_img);
            ds_sam_window_parallel = 1;

            if (global_failed || !global_enc_tokens) {
                fprintf(stderr, "Global image encoding failed\n");
                free(local_features);
                goto cleanup_crops;
            }

            /* Concatenate [local, global, view_separator] */
            int view_sep = 1;
            n_encoder_tokens = local_token_count + n_global_enc_tokens + view_sep;
            encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));

            int offset = 0;
            /* Local tokens first */
            if (local_token_count > 0) {
                memcpy(encoder_output, local_features, local_token_count * hidden * sizeof(float));
                offset = local_token_count;
            }
            /* Global tokens next */
            memcpy(encoder_output + offset * hidden, global_enc_tokens, n_global_enc_tokens * hidden * sizeof(float));
            offset += n_global_enc_tokens;
            /* View separator last */
            if (ctx->vis_tokenizer.view_seperator) {
                memcpy(encoder_output + offset * hidden, ctx->vis_tokenizer.view_seperator, hidden * sizeof(float));
            } else {
                memset(encoder_output + offset * hidden, 0, hidden * sizeof(float));
            }

            free(local_features);
            free(global_enc_tokens);

            if (ds_verbose >= 1)
                fprintf(stderr, "Multi-crop encoding: %d local + %d global + %d sep = %d tokens\n",
                        local_token_count, n_global_enc_tokens, view_sep, n_encoder_tokens);

        cleanup_crops:
            for (int i = 0; i < n_crops; i++) ds_image_free(crops[i]);
            free(crops);
            if (!encoder_output) return NULL;
        } /* end else (large image) */
    } else if (cfg->model_version == 3 && cfg->enc_type == 1) {
        /* ── V3 (Unlimited-OCR) multi-crop path ──
         *
         * Python logic (infer() default: base_size=1024, image_size=640):
         *   Small image (≤640): pad(image_size=640) → SAM(100) → CLIP(100) → Proj(111)
         *     tokenized_image = ([128815]*10+[128815])*10+[128815] = 111 tokens
         *   Large image (>640):
         *     Global: pad(base_size=1024) → SAM(256) → CLIP(256) → Proj(273)
         *     Local: dynamic_preprocess(640) → N crops, each → SAM(100)→CLIP(100)→Proj(111)
         *       strip newlines, reassemble grid, re-insert newlines
         *     Concat: [local, global(272), view_seperator]
         *
         * CLIP encoder returns tokens WITH newlines (273 for 1024², 111 for 640²).
         * We strip newlines, then re-insert in correct grid layout.
         */
        ds_image_t img = { .pixels = (unsigned char *)pixels, .width = width, .height = height, .channels = channels };
        int use_crop = (width > 640 || height > 640);
        int lgrid = 10;  /* 640/16/4 */
        float *nl = ctx->vis_tokenizer.image_newline;
        float *vsep = ctx->vis_tokenizer.view_seperator;

        /* ── Global view ──
         * Python: small image → pad(image_size=640), large image → pad(base_size=1024)
         * Small image: 640×640 → SAM 100 tokens → CLIP 111 tokens (10 rows × 11 + 1 view_sep)
         * Large image: 1024×1024 → SAM 256 tokens → CLIP 273 tokens (16 rows × 17 + 1 view_sep)
         */
        int global_size = use_crop ? 1024 : 640;
        int ggrid = global_size / 16 / 4;  /* 1024→16, 640→10 */

        ds_image_t *gimg = ds_image_pad(&img, global_size, 127);
        if (!gimg) return NULL;
        double t_v3sam = ds_time_sec();
        int n_sg; float *sg = ds_sam_forward_image(ctx, gimg, &n_sg, NULL);
        ctx->perf_sam_ms += (ds_time_sec() - t_v3sam) * 1000.0;
        ds_image_free(gimg);
        if (!sg) return NULL;
        double t_v3clip = ds_time_sec();
        int n_cg; float *cg = ds_clip_encoder_forward(ctx, NULL, 0, 0, 0, sg, n_sg, &n_cg);
        ctx->perf_encoder_ms += (ds_time_sec() - t_v3clip) * 1000.0;
        free(sg);
        if (!cg) return NULL;

        /* Strip newlines: CLIP 273 = 16 rows × (16+1) + 1 view_sep → 256 raw features */
        int n_global_raw = ggrid * ggrid;
        float *global_raw = (float *)malloc(n_global_raw * hidden * sizeof(float));
        for (int r = 0; r < ggrid; r++)
            memcpy(global_raw + r * ggrid * hidden,
                   cg + r * (ggrid + 1) * hidden, ggrid * hidden * sizeof(float));
        free(cg);

        /* Reassemble global with newlines: [16, 17, dim] = 272 tokens */
        int n_gnl = ggrid * (ggrid + 1);
        float *gnl = (float *)malloc(n_gnl * hidden * sizeof(float));
        for (int r = 0; r < ggrid; r++) {
            memcpy(gnl + r * (ggrid + 1) * hidden, global_raw + r * ggrid * hidden, ggrid * hidden * sizeof(float));
            if (nl) memcpy(gnl + (r * (ggrid + 1) + ggrid) * hidden, nl, hidden * sizeof(float));
        }
        free(global_raw);

        if (!use_crop) {
            /* Small image: [global(272), view_sep(1)] = 273 tokens */
            n_encoder_tokens = n_gnl + 1;
            encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));
            memcpy(encoder_output, gnl, n_gnl * hidden * sizeof(float));
            if (vsep) memcpy(encoder_output + n_gnl * hidden, vsep, hidden * sizeof(float));
            free(gnl);
            if (ds_verbose >= 1) fprintf(stderr, "V3: Small: %d+1=%d tokens\n", n_gnl, n_encoder_tokens);
        } else {
            /* ── Local crops ── */
            int nc = 0;
            ds_image_t **crops = ds_dynamic_preprocess(&img, 640, 2, 32, 0, &nc);
            if (!crops || nc < 1) { free(gnl); return NULL; }

            float iar = (float)width / height;
            int cw = 1, ch = 1; float bd = 1e10f;
            for (int tw = 1; tw <= 32; tw++)
                for (int th = 1; th <= 32; th++) {
                    if (tw * th != nc) continue;
                    float d = fabsf(iar - (float)tw / th);
                    if (d < bd) { bd = d; cw = tw; ch = th; }
                }

            int lh = ch * lgrid, lw = cw * lgrid, lrw = lw + 1;
            int n_lnl = lh * lrw;

            if (ds_verbose >= 1)
                fprintf(stderr, "V3: %d crops (%dx%d), local %dx%d→%d tokens\n", nc, cw, ch, lh, lw, n_lnl);

            /* Parallel: encode local crops concurrently (same pattern as V2
             * multi-crop). SAM and CLIP forwards only read shared weights and
             * use thread-local buffers, so one pthread per crop is safe.
             * Crops run in waves of at most num_cpus to bound BLAS
             * oversubscription for large crop counts. */
            crop_task_t *vtasks = (crop_task_t *)calloc(nc, sizeof(crop_task_t));
            pthread_t *vthreads = (pthread_t *)calloc(nc, sizeof(pthread_t));
            /* Crops already run concurrently — disable SAM's internal window
             * threading to avoid oversubscription (restored after joins). */
            ds_sam_window_parallel = 0;
            int max_in_flight = ds_get_num_cpus();
            if (max_in_flight < 1) max_in_flight = 1;
            for (int base = 0; base < nc; base += max_in_flight) {
                int wave = nc - base < max_in_flight ? nc - base : max_in_flight;
                for (int i = 0; i < wave; i++) {
                    vtasks[base + i].ctx = ctx;
                    vtasks[base + i].crop = crops[base + i];
                    pthread_create(&vthreads[base + i], NULL, v3_crop_worker, &vtasks[base + i]);
                }
                for (int i = 0; i < wave; i++)
                    pthread_join(vthreads[base + i], NULL);
            }
            free(vthreads);
            ds_sam_window_parallel = 1;

            /* Accumulate crop SAM/CLIP times for the perf summary */
            for (int ci = 0; ci < nc; ci++) {
                ctx->perf_sam_ms += vtasks[ci].sam_time * 1000.0;
                ctx->perf_encoder_ms += vtasks[ci].enc_time * 1000.0;
            }

            /* Assemble local grid from each crop's CLIP output */
            float *lf = (float *)calloc(n_lnl, hidden * sizeof(float));
            for (int ci = 0; ci < nc; ci++) {
                if (vtasks[ci].failed || !vtasks[ci].enc_tokens) continue;
                int cr = ci / cw, cc = ci % cw;
                /* Strip newlines: 111 → 100 */
                for (int lr = 0; lr < lgrid; lr++) {
                    int so = lr * (lgrid + 1) * hidden;
                    int dr = cr * lgrid + lr, dc = cc * lgrid;
                    int do_ = (dr * lrw + dc) * hidden;
                    memcpy(lf + do_, vtasks[ci].enc_tokens + so, lgrid * hidden * sizeof(float));
                }
                free(vtasks[ci].enc_tokens);
            }
            free(vtasks);
            /* Insert newlines in local grid */
            if (nl) {
                float *lnl = (float *)malloc(n_lnl * hidden * sizeof(float));
                for (int r = 0; r < lh; r++) {
                    memcpy(lnl + r * lrw * hidden, lf + r * lw * hidden, lw * hidden * sizeof(float));
                    memcpy(lnl + (r * lrw + lw) * hidden, nl, hidden * sizeof(float));
                }
                free(lf); lf = lnl;
            }

            /* Concat [local, global, view_seperator] */
            n_encoder_tokens = n_lnl + n_gnl + 1;
            encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));
            int off = 0;
            memcpy(encoder_output + off * hidden, lf, n_lnl * hidden * sizeof(float)); off += n_lnl;
            memcpy(encoder_output + off * hidden, gnl, n_gnl * hidden * sizeof(float)); off += n_gnl;
            if (vsep) memcpy(encoder_output + off * hidden, vsep, hidden * sizeof(float));
            free(lf); free(gnl);
            for (int i = 0; i < nc; i++) ds_image_free(crops[i]);
            free(crops);
            if (ds_verbose >= 1)
                fprintf(stderr, "V3: Multi-crop: %d local + %d global + 1 sep = %d tokens\n", n_lnl, n_gnl, n_encoder_tokens);
        }
    } else {
        /* V1 path (no multi-crop) */
        /* Step 1: Visual tokenizer (SAM) — resize to 1024x1024 */
        int n_visual_tokens;
        float *patch_embeds = NULL;
        double t_sam = ds_time_sec();
        float *visual_tokens = ds_visual_tokenizer_forward(ctx, pixels, width, height, channels,
                                                            &n_visual_tokens, &patch_embeds,
                                                            NULL, NULL, NULL);
        ctx->perf_sam_ms = (ds_time_sec() - t_sam) * 1000.0;
        if (!visual_tokens) {
            fprintf(stderr, "Visual tokenizer failed\n");
            return NULL;
        }

        /* Step 2: Encoder (CLIP V1 or DeepEncoder V2) */
        double t_enc = ds_time_sec();
        if (cfg->enc_type == 1) {
            encoder_output = ds_clip_encoder_forward(ctx,
                                                      NULL, 0, 0, 0,
                                                      visual_tokens, n_visual_tokens,
                                                      &n_encoder_tokens);
        } else {
            encoder_output = ds_encoder_forward_v2(ctx, visual_tokens, n_visual_tokens,
                                                    &n_encoder_tokens,
                                                    cfg->enc_causal_flow_queries,
                                                    ctx->vis_tokenizer.causal_query_embeddings);
        }
        ctx->perf_encoder_ms = (ds_time_sec() - t_enc) * 1000.0;
        free(patch_embeds);
        free(visual_tokens);
        if (!encoder_output) {
            fprintf(stderr, "Encoder forward failed\n");
            return NULL;
        }
    }

    /* Override encoder output with Python reference for decoder quality testing */
    if (getenv("DS_PERFECT_ENCODER")) {
        /* Choose the right .npy file based on token count:
         * 256 tokens -> dump/proj_output.npy (global-only)
         * 1121 tokens -> dump/multicrop/full_proj_output.npy (multi-crop) */
        const char *npy_path = NULL;
        if (n_encoder_tokens == 256) {
            npy_path = "dump/proj_output.npy";
        } else if (n_encoder_tokens == 1121) {
            npy_path = "dump/multicrop/full_proj_output.npy";
        }
        if (npy_path) {
            FILE *f = fopen(npy_path, "rb");
            if (f) {
                /* .npy format: magic(6) + version(2) + header_len(2 or 4) + header + data */
                unsigned char buf[10];
                fread(buf, 1, 10, f);
                uint16_t header_len = *(uint16_t *)(buf + 8);
                fseek(f, 10 + header_len, SEEK_SET);
                fread(encoder_output, sizeof(float), n_encoder_tokens * hidden, f);
                fclose(f);
                if (ds_verbose >= 1)
                    fprintf(stderr, "Override encoder output with Python reference (%d tokens from %s)\n",
                            n_encoder_tokens, npy_path);
            } else {
                fprintf(stderr, "Warning: DS_PERFECT_ENCODER set but %s not found\n", npy_path);
            }
        } else {
            fprintf(stderr, "Warning: DS_PERFECT_ENCODER: unsupported token count %d\n", n_encoder_tokens);
        }
    }

    double encode_end = now_ms();

    /* ── Step 3: 构造解码器输入序列 ──
     * 将编码器输出(视觉token) + prompt文本(token IDs) 拼接为解码器输入
     * 不同版本有不同的prompt格式和token布局:
     *
     * V1: [BOS][img_start][encoder_output(273)][img_end][\nFree OCR.]
     * V2: [BOS][encoder_output(n_img_tokens)][\nFree OCR.]
     * V3: [BOS][image_placeholder(128815)×n][\ndocument parsing.]
     *
     * 每个token通过 tok_embeddings 查表得到嵌入向量 [hidden=1280]
     * 图像token位置用编码器输出覆盖嵌入值
     */

prompt_construction:
    /* Optional: skip SAM+encoder entirely by loading Python's encoder output.
     * DS_LOAD_ENCODER_OUTPUT=dump/py_encoder_output.bin loads [1121, 1280] float32
     * and overrides n_encoder_tokens and encoder_output. */
    {
        const char *load_enc = getenv("DS_LOAD_ENCODER_OUTPUT");
        if (load_enc) {
            FILE *ef = fopen(load_enc, "rb");
            if (ef) {
                fseek(ef, 0, SEEK_END);
                long fsize = ftell(ef);
                fseek(ef, 0, SEEK_SET);
                int n_tokens = (int)(fsize / (sizeof(float) * hidden));
                float *enc_buf = (float *)malloc(n_tokens * hidden * sizeof(float));
                int n_read = (int)fread(enc_buf, sizeof(float), n_tokens * hidden, ef);
                fclose(ef);
                if (n_read == n_tokens * hidden) {
                    free(encoder_output);
                    encoder_output = enc_buf;
                    n_encoder_tokens = n_tokens;
                    if (ds_verbose >= 1)
                        fprintf(stderr, "DS_LOAD_ENCODER_OUTPUT: loaded %d tokens from %s\n",
                                n_tokens, load_enc);
                } else {
                    fprintf(stderr, "Warning: DS_LOAD_ENCODER_OUTPUT size mismatch\n");
                    free(enc_buf);
                }
            } else {
                fprintf(stderr, "Warning: DS_LOAD_ENCODER_OUTPUT=%s not found\n", load_enc);
            }
        }
    }

    /* Load tokenizer early (needed for prompt encoding in V2).
     * Try vocab.json first (standalone format), then tokenizer.json (HuggingFace format).
     * tokenizer.json model.vocab has the same {token:id} structure as vocab.json. */
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/vocab.json", model_dir_str);
    tokenizer = ds_tokenizer_load(tokenizer_path);
    if (!tokenizer) {
        /* Fallback: extract model.vocab from tokenizer.json.
         * ds_tokenizer_load_from_tokenizer_json() parses the nested format. */
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir_str);
        tokenizer = ds_tokenizer_load_from_tokenizer_json(tokenizer_path);
    }
    if (!tokenizer && ds_verbose >= 1)
        fprintf(stderr, "Note: neither vocab.json nor tokenizer.json found, tokenizer not loaded\n");

    float *input_embeds = NULL;
    int prefix_len = 0;

    if (cfg->model_version == 2) {
        /* V2 prompt format matching Python model.infer() with sft_format='plain':
         * Python prompt = '<image>\nFree OCR. '
         * With multi-crop: encoder_output already contains [local, global, view_sep]
         * C equivalent: [BOS] + image_tokens + '\nFree OCR.'(4 tokens)
         * Note: PLAIN template has no role tokens — roles=("", ""), sep=""
         *
         * Python's masked_scatter layout:
         *   images_seq_mask True positions = [global(257), local(W*H*100)]
         *     where global = num_queries_base^2 + 1 = 16*16+1 = 257
         *     and local = (num_queries*W) * (num_queries*H) = 10*W * 10*H per crop
         *   source = cat([local_features(N*144), global_features(256), view_sep(1)])
         *   masked_scatter fills True positions IN ORDER from source[0].
         *   So: global_slots[0:257] ← source[0:257] = local[0:257]
         *       local_slots[0:M]   ← source[257:257+M] = local[257:] + global[0:...]
         */
        int ds_image_size = 640;
        int ds_num_queries = (ds_image_size + 15) / 16 / 4;  /* ceil(640/16/4) = 10 */
        int ds_num_queries_base = (1024 + 15) / 16 / 4;      /* ceil(1024/16/4) = 16 */
        int ds_local_tokens_per_crop = ds_num_queries * ds_num_queries;  /* 100 */
        int n_local_enc = n_encoder_tokens - 257;  /* local features count */
        if (n_local_enc < 0) n_local_enc = 0;
        int n_crops_count = n_local_enc / 144;
        int n_global_slots = ds_num_queries_base * ds_num_queries_base + 1; /* 257: 256 global + 1 sep */
        int n_local_slots = n_crops_count * ds_local_tokens_per_crop;
        int n_img_tokens = n_global_slots + n_local_slots;  /* total True positions */

        int n_text_after = 0;
        int *text_after_ids = NULL;
        if (tokenizer) {
            text_after_ids = ds_tokenizer_encode(tokenizer, "\nFree OCR.", &n_text_after);
            if (!text_after_ids || n_text_after <= 0) {
                static const int fallback_ids[] = {201, 21431, 126041, 16};
                n_text_after = 4;
                text_after_ids = (int *)malloc(n_text_after * sizeof(int));
                memcpy(text_after_ids, fallback_ids, n_text_after * sizeof(int));
            }
            if (ds_verbose >= 1) {
                fprintf(stderr, "V2 Prompt: BOS + img(%d = global%d + local%d) + after(%d) = %d\n",
                        n_img_tokens, n_global_slots, n_local_slots, n_text_after,
                        1 + n_img_tokens + n_text_after);
            }
        }

        if (ds_verbose >= 1) {
            fprintf(stderr, "V2 Layout: image_size=%d, num_queries=%d, "
                   "local_tokens/crop=%d, crops=%d, local_enc=%d, n_enc=%d\n",
                   ds_image_size, ds_num_queries, ds_local_tokens_per_crop,
                   n_crops_count, n_local_enc, n_encoder_tokens);
        }

        prefix_len = 1 + n_img_tokens + n_text_after;
        input_embeds = (float *)malloc(prefix_len * hidden * sizeof(float));
        memset(input_embeds, 0, prefix_len * hidden * sizeof(float));
        int pos = 0;

        /* Helper: embed a single token ID at current pos and advance */
        #define EMBED_TOKEN(tid) do { \
            if (ctx->decoder.tok_embeddings_bf16 && (tid) >= 0 && (tid) < cfg->vocab_size) { \
                const uint16_t *_e = ctx->decoder.tok_embeddings_bf16 + (size_t)(tid) * hidden; \
                for (int _i = 0; _i < hidden; _i++) { \
                    uint32_t _f32 = ((uint32_t)_e[_i]) << 16; \
                    memcpy(&input_embeds[pos * hidden + _i], &_f32, sizeof(float)); \
                } \
            } \
            pos++; \
        } while(0)

        /* 1. BOS token (id=0) */
        EMBED_TOKEN(DS_TOKEN_BOS);

        /* 2. Image tokens — must match Python's masked_scatter layout.
         *
         * Python: images_seq_mask True positions = [global(257), local(n_local_slots)]
         * Source = encoder_output = [local(n_local_enc), global(256), view_sep(1)]
         * masked_scatter fills True positions sequentially from source:
         *   global_slots (257 positions) ← source[0:min(257, n_local_enc)]
         *   local_slots  (n_local_slots) ← source[257:257+n_local_slots] (may wrap into global/sep)
         *
         * We simulate this by iterating through True positions in order and
         * copying from source in order.
         */
        if (getenv("DS_DUMP_ENCODER")) {
            FILE *df = fopen("dump/c_encoder_output.bin", "wb");
            if (df) { fwrite(encoder_output, sizeof(float), n_encoder_tokens * hidden, df); fclose(df); }
            fprintf(stderr, "Dumped C encoder output: %d tokens x %d dim to dump/c_encoder_output.bin\n",
                    n_encoder_tokens, hidden);
        }

        /* Simulate masked_scatter: fill n_img_tokens True positions from source in order.
         * Source = enc = [local(n_local_enc), global(256), view_sep(1)].
         * We copy source[0..n_img_tokens-1] into image_positions (1..n_img_tokens).
         * Since image_positions in input_embeds are contiguous after BOS,
         * this is just a contiguous copy of the first n_img_tokens source elements.
         * Any source elements beyond n_img_tokens are dropped (masked_scatter truncation). */
        int n_copy = n_img_tokens < n_encoder_tokens ? n_img_tokens : n_encoder_tokens;
        memcpy(input_embeds + pos * hidden, encoder_output, n_copy * hidden * sizeof(float));
        pos += n_img_tokens;

        /* 3. Text after image: "\nFree OCR." (4 tokens) */
        for (int t = 0; t < n_text_after; t++) EMBED_TOKEN(text_after_ids[t]);

        #undef EMBED_TOKEN

        /* Dump input_embeds for comparison with Python */
        if (getenv("DS_DUMP_INPUT_EMBEDS")) {
            FILE *df = fopen("dump/c_inputs_embeds.bin", "wb");
            if (df) { fwrite(input_embeds, sizeof(float), prefix_len * hidden, df); fclose(df); }
            if (ds_verbose >= 1)
                fprintf(stderr, "Dumped C input_embeds: %d x %d to dump/c_inputs_embeds.bin\n", prefix_len, hidden);
        }

        free(text_after_ids);
    } else if (cfg->model_version == 3) {
        /* Unlimited-OCR (V3) prompt format:
         * Python: format_messages with sft_format='plain' → "<image>\ndocument parsing."
         * Token layout: [BOS] + image_tokens(128815) + ["\ndocument parsing."]
         *
         * Image token count = n_encoder_tokens (from CLIP+SAM+Projector):
         *   - Small image (≤640): [global(110), view_sep(1)] = 111
         *   - Large image (>640):  [local(N), global(272), view_sep(1)] = N+273
         *     where N = (crop_h*10)*(crop_w*10+1)
         *
         * Python's masked_scatter places all encoder tokens into the
         * image positions (all True in images_seq_mask).
         */
        int n_img_tokens = n_encoder_tokens;  /* Dynamic: matches encoder output */

        int n_text_after = 0;
        int *text_after_ids = NULL;
        if (tokenizer) {
            /* Python tokenizer encodes "\ndocument parsing." as [201, 34030, 76466, 16].
             * C BPE tokenizer may not handle this correctly, so use verified IDs. */
            text_after_ids = ds_tokenizer_encode(tokenizer, "\ndocument parsing.", &n_text_after);
            if (!text_after_ids || n_text_after <= 0 || n_text_after > 10) {
                /* Fallback: verified Python token IDs for "\ndocument parsing." */
                static const int fallback_ids[] = {201, 34030, 76466, 16};
                n_text_after = 4;
                if (text_after_ids) free(text_after_ids);
                text_after_ids = (int *)malloc(n_text_after * sizeof(int));
                memcpy(text_after_ids, fallback_ids, n_text_after * sizeof(int));
            }
            if (ds_verbose >= 1) {
                fprintf(stderr, "V3 Prompt: BOS + img(%d, id=128815) + after(%d) = %d tokens\n",
                        n_img_tokens, n_text_after, 1 + n_img_tokens + n_text_after);
            }
        }

        prefix_len = 1 + n_img_tokens + n_text_after;
        input_embeds = (float *)malloc(prefix_len * hidden * sizeof(float));
        memset(input_embeds, 0, prefix_len * hidden * sizeof(float));
        int pos = 0;

        #define EMBED_TOKEN_V3(tid) do { \
            if (ctx->decoder.tok_embeddings_bf16 && (tid) >= 0 && (tid) < cfg->vocab_size) { \
                const uint16_t *_e = ctx->decoder.tok_embeddings_bf16 + (size_t)(tid) * hidden; \
                for (int _i = 0; _i < hidden; _i++) { \
                    uint32_t _f32 = ((uint32_t)_e[_i]) << 16; \
                    memcpy(&input_embeds[pos * hidden + _i], &_f32, sizeof(float)); \
                } \
            } \
            pos++; \
        } while(0)

        /* 1. BOS token (id=0) */
        EMBED_TOKEN_V3(DS_TOKEN_BOS);

        /* 2. Image tokens — embed all with 128815, then overwrite with encoder output.
         * The encoder returns exactly n_encoder_tokens tokens which should equal
         * n_img_tokens (273). Overwrite all image positions with encoder output. */
        int img_start = pos;
        for (int i = 0; i < n_img_tokens; i++) {
            EMBED_TOKEN_V3(DS_TOKEN_IMAGE_PLACEHOLDER);  /* 128815 */
        }

        /* Overwrite image positions with encoder output */
        int n_copy = n_encoder_tokens < n_img_tokens ? n_encoder_tokens : n_img_tokens;
        memcpy(input_embeds + img_start * hidden, encoder_output, n_copy * hidden * sizeof(float));

        if (ds_verbose >= 1 && n_encoder_tokens != n_img_tokens) {
            fprintf(stderr, "Warning: n_encoder_tokens=%d != n_img_tokens=%d\n",
                    n_encoder_tokens, n_img_tokens);
        }

        /* 3. Text after image */
        for (int t = 0; t < n_text_after; t++) EMBED_TOKEN_V3(text_after_ids[t]);

        #undef EMBED_TOKEN_V3
        free(text_after_ids);
    } else {
        /* V1 prompt format: [BOS][image_start][encoder_output][image_end][\nFree OCR.]
         * Same as V2 but with image_start/end tokens instead of placeholder token. */
        int n_text_after = 0;
        int *text_after_ids = NULL;
        if (tokenizer) {
            text_after_ids = ds_tokenizer_encode(tokenizer, "\nFree OCR.", &n_text_after);
        }
        if (!text_after_ids || n_text_after <= 0) {
            static const int fallback_ids[] = {201, 21431, 126041, 16};
            n_text_after = 4;
            text_after_ids = (int *)malloc(n_text_after * sizeof(int));
            memcpy(text_after_ids, fallback_ids, n_text_after * sizeof(int));
        }

        prefix_len = 1 + 1 + n_encoder_tokens + 1 + n_text_after; /* BOS + img_start + enc + img_end + text */
        input_embeds = (float *)malloc(prefix_len * hidden * sizeof(float));
        memset(input_embeds, 0, prefix_len * hidden * sizeof(float));

        int pos = 0;

        /* BOS token */
        if (ctx->decoder.tok_embeddings_bf16) {
            const uint16_t *emb = ctx->decoder.tok_embeddings_bf16 + (size_t)DS_TOKEN_BOS * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb[i]) << 16;
                memcpy(&input_embeds[pos * hidden + i], &f32_bits, sizeof(float));
            }
        }
        pos++;

        /* image_start token */
        if (ctx->decoder.tok_embeddings_bf16) {
            const uint16_t *emb_start = ctx->decoder.tok_embeddings_bf16 + (size_t)DS_TOKEN_IMAGE_START * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb_start[i]) << 16;
                memcpy(&input_embeds[pos * hidden + i], &f32_bits, sizeof(float));
            }
        }
        pos++;

        /* Encoder output */
        memcpy(input_embeds + pos * hidden, encoder_output, n_encoder_tokens * hidden * sizeof(float));
        pos += n_encoder_tokens;

        /* image_end token */
        if (ctx->decoder.tok_embeddings_bf16) {
            const uint16_t *emb_end = ctx->decoder.tok_embeddings_bf16 + (size_t)DS_TOKEN_IMAGE_END * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb_end[i]) << 16;
                memcpy(&input_embeds[pos * hidden + i], &f32_bits, sizeof(float));
            }
        }
        pos++;

        /* Text after image: "\nFree OCR." */
        for (int t = 0; t < n_text_after; t++) {
            int tid = text_after_ids[t];
            if (ctx->decoder.tok_embeddings_bf16 && tid >= 0 && tid < cfg->vocab_size) {
                const uint16_t *emb = ctx->decoder.tok_embeddings_bf16 + (size_t)tid * hidden;
                for (int i = 0; i < hidden; i++) {
                    uint32_t f32_bits = ((uint32_t)emb[i]) << 16;
                    memcpy(&input_embeds[pos * hidden + i], &f32_bits, sizeof(float));
                }
            }
            pos++;
        }

        free(text_after_ids);
        if (ds_verbose >= 1)
            fprintf(stderr, "V1 prompt: BOS + img_start + enc(%d) + img_end + text(%d) = %d tokens\n",
                    n_encoder_tokens, n_text_after, prefix_len);
    }
    free(encoder_output);

    if (ds_verbose >= 1)
        fprintf(stderr, "Decoder prefix: %d tokens\n", prefix_len);

    /* Optional: override inputs_embeds with Python reference for debugging */
    const char *load_emb = getenv("DS_LOAD_INPUT_EMBEDS");
    if (load_emb) {
        FILE *ef = fopen(load_emb, "rb");
        if (ef) {
            fread(input_embeds, sizeof(float), prefix_len * hidden, ef);
            fclose(ef);
            fprintf(stderr, "Loaded inputs_embeds from %s (%d x %d)\n",
                    load_emb, prefix_len, hidden);
        } else {
            fprintf(stderr, "Warning: DS_LOAD_INPUT_EMBEDS=%s not found\n", load_emb);
        }
    }

    /* ── Step 4: 重置KV缓存 + Prefill ──
     *
     * Prefill: 将所有prefix token(视觉+prompt)一次性送入解码器
     * 使用批量矩阵乘法(sgemm)高效处理整个序列
     * 结果: KV缓存被填满，第一个生成token的logits可用
     *
     * 为什么prefill而非逐token?
     *   逐token: 280 tokens × 12层 × ~23ms/step = ~77s（太慢！）
     *   批量prefill: 一次sgemm处理所有token = ~0.7s（100×加速）
     *
     * Python对应: model.generate() 先prefill整个prompt，再自回归decode
     */
    ctx->kv_cache_len = 0;

    float *dec_input = (float *)malloc(hidden * sizeof(float));

    /* Prefill ALL prefix tokens. Python's model.generate() prefills the entire
     * prompt (including last text token), then samples from the last position's
     * logits. We do the same: prefill prefix_len tokens, then use prefill logits
     * for first generated token, then decode subsequent tokens. */
    int first_token = -1;
    const char *slow_prefill = getenv("DS_SLOW_PREFILL");
    if (prefix_len > 1) {
        if (slow_prefill) {
            /* Token-by-token prefill: process each prefix token individually
             * using the single-token decode path. This is slow but avoids
             * potential bugs in batched prefill. */
            if (ds_verbose >= 1)
                fprintf(stderr, "Slow prefill: %d tokens one-by-one...\n", prefix_len);
            for (int i = 0; i < prefix_len; i++) {
                memcpy(dec_input, input_embeds + i * hidden, hidden * sizeof(float));
                int tok = ds_decoder_forward(ctx, dec_input);
                if (i == prefix_len - 1) {
                    /* Last prefix token — use its logits for first generated token */
                    if (ctx->dec_logits) {
                        float *logits = ctx->dec_logits;
                        int vocab = cfg->vocab_size;
                        float best_val = -1e30f;
                        int best_id = 0;
                        for (int v = 0; v < vocab; v++) {
                            if (logits[v] > best_val) { best_val = logits[v]; best_id = v; }
                        }
                        first_token = best_id;
                    }
                    if (ds_verbose >= 1)
                        fprintf(stderr, "Slow prefill done, first token=%d\n", first_token);
                }
            }
        } else {
            /* Debug: dump input_embeds for comparison with Python */
            {
                const char *dump_ie = getenv("DS_DUMP_INPUT_EMBEDS");
                if (dump_ie) {
                    FILE *f = fopen(dump_ie, "wb");
                    if (f) {
                        fwrite(input_embeds, sizeof(float), prefix_len * hidden, f);
                        fclose(f);
                        fprintf(stderr, "Dumped input_embeds (%d x %d) to %s\n",
                                prefix_len, hidden, dump_ie);
                    }
                }
            }
            ds_decoder_prefill(ctx, input_embeds, prefix_len);
            /* Use prefill logits for first token selection */
            if (ctx->dec_logits) {
                float *logits = ctx->dec_logits;
                int vocab = cfg->vocab_size;
                /* Simple argmax for first token */
                float best_val = -1e30f;
                int best_id = 0;
                for (int i = 0; i < vocab; i++) {
                    if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
                }
                first_token = best_id;
                /* V3 EOS-first fix: suppress EOS and common spurious tokens
                 * for first token selection. Must be OUTSIDE verbose block!
                 * V3 model trained with BF16 autocast tends to output EOS
                 * as top-1 in F32 precision. Suppressing EOS alone picks Ġ(space)
                 * which leads to "10." loops. Also suppress Ġ and <|/det|>
                 * so the model picks a meaningful content/detection token. */
                if (cfg->model_version == 3 && first_token == 1) { /* EOS */
                    /* V3 model in F32 precision outputs EOS as first token.
                     * Ban EOS + Ġ to avoid "10." loops. The model then picks
                     * a content token (e.g., ĠThe) which may produce a short
                     * hallucination prefix before actual OCR content. */
                    logits[1] = -1e30f;   /* Ban EOS */
                    logits[223] = -1e30f;  /* Ban Ġ (space prefix → "10." loops) */
                    best_val = -1e30f; best_id = 0;
                    for (int i = 0; i < vocab; i++) {
                        if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
                    }
                    first_token = best_id;
                    if (ds_verbose >= 1)
                        fprintf(stderr, "V3: EOS+Ġ suppressed, first_token=%d (%.2f)\n",
                                first_token, best_val);
                }
                if (ds_verbose >= 1) {
                    /* Print top-5 prefill logits */
                    int top_ids[5]; float top_vals[5];
                    for (int i = 0; i < 5; i++) { top_ids[i] = -1; top_vals[i] = -1e30f; }
                    for (int i = 0; i < vocab; i++) {
                        for (int j = 0; j < 5; j++) {
                            if (logits[i] > top_vals[j]) {
                                for (int k = 4; k > j; k--) { top_vals[k] = top_vals[k-1]; top_ids[k] = top_ids[k-1]; }
                                top_vals[j] = logits[i]; top_ids[j] = i; break;
                            }
                        }
                    }
                    fprintf(stderr, "Prefill top5: [%d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f]\n",
                            top_ids[0], top_vals[0], top_ids[1], top_vals[1],
                            top_ids[2], top_vals[2], top_ids[3], top_vals[3],
                            top_ids[4], top_vals[4]);
                }
            }
        }
    } else {
        /* Single token: use it directly */
        memcpy(dec_input, input_embeds, hidden * sizeof(float));
    }
    double prefill_end = now_ms();
    free(input_embeds);

    /* Store prefill token count for R-SWA (Unlimited-OCR).
     * After prefill, kv_cache_len = number of visual + prompt tokens.
     * During decode, R-SWA keeps these tokens as "reference" and only
     * attends to reference + last sliding_window_size text tokens. */
    ctx->prefill_token_count = ctx->kv_cache_len;
    if (cfg->sliding_window_size > 0 && ds_verbose >= 1) {
        fprintf(stderr, "R-SWA: prefill_token_count=%d, sliding_window_size=%d\n",
                ctx->prefill_token_count, cfg->sliding_window_size);
    }

    double decode_start = now_ms();

    /* Configure thread pool for decode.
     * Use all available threads — the argmax LM head path benefits from 8T
     * for the 129280-row matvec, and small expert matvecs work fine with 8T. */
    {
        int n_total = ds_get_num_cpus();
        ds_set_threads(n_total);
    }

    /* ── Step 5: 自回归解码循环 ──
     *
     * 核心循环: 每步生成1个token，直到EOS或达到max_new_tokens
     *
     * 每步流程:
     *   1. ds_decoder_forward(ctx, dec_input) → token_id
     *      内部: 12层decoder前向 → argmax(lm_head @ x) → next_token_id
     *   2. 检查EOS → 如果提前到达EOS且<min_new_tokens，抑制并选次优token
     *   3. 记录token到历史（用于重复惩罚）
     *   4. ds_tokenizer_decode(token_id) → 文本片段 → 追加到输出
     *   5. tok_embeddings[token_id] → 下一步的输入嵌入
     *
     * 【V3 特殊处理】
     * - EOS抑制: V3模型F32精度下EOS logit偏高，前32步需禁用EOS+Ġ
     * - R-SWA: 解码器注意力只看reference(视觉)+window(最近128个文本token)
     * - det标签过滤: 流式输出时实时过滤<|det|>...<|/det|>检测标签
     */

    /* Build output string */
    int capacity = 4096;
    char *output = (char *)malloc(capacity);
    int out_len = 0;
    output[0] = '\0';

    int n_generated = 0;
    int eos_token = DS_TOKEN_EOS;

    for (int step = 0; step < ctx->max_new_tokens; step++) {
        /* Check KV cache bounds */
        if (ctx->kv_cache_len >= ctx->kv_cache_max) break;

        int token;
        if (step == 0 && first_token >= 0) {
            /* First token comes from prefill logits — no extra decode needed.
             * Python's generate() prefills all prefix tokens then samples from
             * the last position. We already did that in ds_decoder_prefill. */
            token = first_token;
            if (ds_verbose >= 1)
                fprintf(stderr, "Step 0: token=%d (from prefill logits)\n", token);
        } else {
            /* Decode: process current token, get next token */
            token = ds_decoder_forward(ctx, dec_input);
            if (ds_verbose >= 3 && step < 5) {
                /* Print top-3 logits for first decode steps */
                float *logits = ctx->dec_logits;
                if (logits) {
                    int top_ids[3]; float top_vals[3];
                    for (int i = 0; i < 3; i++) { top_ids[i] = -1; top_vals[i] = -1e30f; }
                    for (int i = 0; i < cfg->vocab_size; i++) {
                        for (int j = 0; j < 3; j++) {
                            if (logits[i] > top_vals[j]) {
                                for (int k = 2; k > j; k--) { top_vals[k] = top_vals[k-1]; top_ids[k] = top_ids[k-1]; }
                                top_vals[j] = logits[i]; top_ids[j] = i; break;
                            }
                        }
                    }
                    fprintf(stderr, "Step %d: token=%d, top3=[%d:%.2f,%d:%.2f,%d:%.2f]\n",
                            step, token, top_ids[0], top_vals[0], top_ids[1], top_vals[1], top_ids[2], top_vals[2]);
                } else {
                    fprintf(stderr, "Step %d: token=%d (no logits)\n", step, token);
                }
            } else if (ds_verbose >= 2) {
                fprintf(stderr, "Step %d: token=%d\n", step, token);
            }
        }

        if (token == eos_token) {
            /* Suppress premature EOS: if we haven't generated enough tokens yet,
             * ban EOS and pick the next-best token instead.
             * This matches Python's NoEOSTextStreamer behavior where EOS is
             * replaced with newline and generation continues. */
            if (step < ctx->min_new_tokens && ctx->dec_logits) {
                float *logits = ctx->dec_logits;
                /* Ban EOS and Ġ during warmup. V3 model tends to oscillate
                 * between EOS and Ġ (space prefix), producing "10." loops.
                 * Banning both forces the model to pick meaningful content tokens. */
                logits[eos_token] = -1e30f;
                if (cfg->model_version == 3) logits[223] = -1e30f; /* Ban Ġ for V3 */
                int best_id = 0;
                float best_val = logits[0];
                for (int i = 1; i < cfg->vocab_size; i++) {
                    if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
                }
                token = best_id;
                if (ds_verbose >= 1) {
                    fprintf(stderr, "EOS suppressed at step %d (< min_new_tokens=%d), using token=%d instead\n",
                            step, ctx->min_new_tokens, token);
                }
            } else {
                if (ds_verbose >= 1) {
                    /* Print top-5 logits at EOS to diagnose termination */
                    float *logits = ctx->dec_logits;
                    if (logits) {
                        int top_ids[5]; float top_vals[5];
                        for (int i = 0; i < 5; i++) { top_ids[i] = -1; top_vals[i] = -1e30f; }
                        for (int i = 0; i < cfg->vocab_size; i++) {
                            for (int j = 0; j < 5; j++) {
                                if (logits[i] > top_vals[j]) {
                                    for (int k = 4; k > j; k--) { top_vals[k] = top_vals[k-1]; top_ids[k] = top_ids[k-1]; }
                                    top_vals[j] = logits[i]; top_ids[j] = i; break;
                                }
                            }
                        }
                        fprintf(stderr, "EOS at step %d, top5=[%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f] (EOS logit=%.4f)\n",
                                step, top_ids[0], top_vals[0], top_ids[1], top_vals[1],
                                top_ids[2], top_vals[2], top_ids[3], top_vals[3], top_ids[4], top_vals[4],
                                logits[eos_token]);
                    } else {
                        fprintf(stderr, "EOS at step %d\n", step);
                    }
                }
                break;
            }
        }

        /* Record token in history for repetition penalty */
        if (ctx->token_history && ctx->token_history_len < ctx->token_history_cap) {
            ctx->token_history[ctx->token_history_len++] = token;
        }
        if (ds_verbose >= 2) fprintf(stderr, "  token[%d] = %d\n", step, token);

        /* Decode token to text */
        if (tokenizer) {
            const char *piece = ds_tokenizer_decode(tokenizer, token);
            if (piece) {
                int piece_len = (int)strlen(piece);
                while (out_len + piece_len + 1 >= capacity) {
                    capacity *= 2;
                    output = (char *)realloc(output, capacity);
                }
                memcpy(output + out_len, piece, piece_len);
                out_len += piece_len;
                output[out_len] = '\0';

                /* Stream token — with V3 det tag filtering for streaming output */
                if (ctx->token_cb) {
                    ds_stream_filter_det(ctx, piece);
                }
            }
        }

        /* Set embedding for next token from tok_embeddings */
        if (ctx->decoder.tok_embeddings_bf16 && token < cfg->vocab_size) {
            /* Convert bf16 embedding to f32 */
            const uint16_t *emb = ctx->decoder.tok_embeddings_bf16 + (size_t)token * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb[i]) << 16;
                memcpy(&dec_input[i], &f32_bits, sizeof(float));
            }
        }

        n_generated++;
    }

    ctx->perf_decode_steps = n_generated;

    free(dec_input);

    double total_end = now_ms();
    ctx->perf_total_ms = total_end - t0;
    ctx->perf_text_tokens = n_generated;
    ctx->perf_encode_ms = encode_end - encode_start;
    ctx->perf_prefill_ms = prefill_end - encode_end;  /* prompt construction + prefill */
    ctx->perf_decode_ms = total_end - decode_start;

    if (ds_verbose >= 1)
        fprintf(stderr, "Recognition: %d tokens generated in %.0f ms\n",
                n_generated, ctx->perf_total_ms);

    /* Trim trailing whitespace (matching Python .strip()) */
    if (output && out_len > 0) {
        while (out_len > 0 && (output[out_len-1] == ' ' || output[out_len-1] == '\n' ||
                                output[out_len-1] == '\r' || output[out_len-1] == '\t')) {
            output[--out_len] = '\0';
        }
    }

    /* V3 (Unlimited-OCR) post-processing: strip detection coordinate tags.
     * Python: det_pattern = r'(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)'
     * and outputs.replace(a_match_other, '') — removes the entire <|det|>...<|/det|> block.
     * Also strips <|ref|>...<|/ref|><|det|>...<|/det|> blocks (ref+det pairs).
     * This removes bounding box coordinates while keeping the descriptive text. */
    if (output && out_len > 0 && cfg->model_version == 3) {
        out_len = ds_strip_det_tags(output, out_len);
        /* V3 hallucination prefix removal: when the model's first token
         * is forced past EOS, it may generate a short English hallucination
         * before the actual OCR content. Common patterns:
         *   "The image contains no text... [No text detected]\n<actual OCR>"
         *   "The image contains no text... Therefore... [No text detected]\n<actual OCR>"
         * Strip everything up to and including "[No text detected]" marker.
         * If no marker found, also check for English-only prefix before CJK text. */
        if (out_len > 20) {
            /* Skip leading whitespace for pattern matching */
            int start = 0;
            while (start < out_len && (output[start] == ' ' || output[start] == '\n'))
                start++;
            const char *marker = "[No text detected]";
            int mlen = 18;
            /* Search for marker in first 400 chars from start */
            int search_len = (out_len - start) < 400 ? (out_len - start) : 400;
            int stripped = 0;
            for (int i = start; i + mlen <= start + search_len; i++) {
                if (memcmp(output + i, marker, mlen) == 0) {
                    int skip = i + mlen;
                    /* Skip whitespace/newlines after marker */
                    while (skip < out_len && (output[skip] == '\n' || output[skip] == ' '))
                        skip++;
                    if (skip < out_len) {
                        memmove(output, output + skip, out_len - skip + 1);
                        out_len -= skip;
                    }
                    stripped = 1;
                    if (ds_verbose >= 1)
                        fprintf(stderr, "V3: stripped hallucination prefix to: [%s]\n", output);
                    break;
                }
            }
            /* If no [No text detected] marker, try stripping English-only prefix
             * before first CJK/meaningful content. Pattern: the model generates
             * a sentence starting with "The image..." before real OCR output.
             * Look for double-newline boundary where actual content starts. */
            if (!stripped && out_len > 50) {
                /* Check if output starts with common hallucination phrases
                 * (after leading whitespace) */
                const char *halluc_starts[] = {
                    "The image contains no text",
                    "There is no text",
                    "No text is present",
                    "The OCR result",
                    NULL
                };
                for (int h = 0; halluc_starts[h]; h++) {
                    int hlen = (int)strlen(halluc_starts[h]);
                    if (start + hlen <= out_len && memcmp(output + start, halluc_starts[h], hlen) == 0) {
                        /* Find end of this English paragraph (double newline or first CJK char) */
                        int skip = start + hlen;
                        /* Skip to end of English sentence(s) — look for \n\n or first CJK */
                        while (skip < out_len) {
                            unsigned char c = (unsigned char)output[skip];
                            /* CJK range: bytes 0xE0+ in UTF-8 */
                            if (c >= 0xE0) break;
                            /* Double newline */
                            if (skip + 1 < out_len && output[skip] == '\n' && output[skip+1] == '\n') {
                                skip += 2;
                                break;
                            }
                            skip++;
                        }
                        /* Skip trailing whitespace/newlines */
                        while (skip < out_len && (output[skip] == '\n' || output[skip] == ' '))
                            skip++;
                        if (skip < out_len && skip > start) {
                            memmove(output, output + skip, out_len - skip + 1);
                            out_len -= skip;
                        }
                        stripped = 1;
                        break;
                    }
                }
            }
        }

        /* Strip broken leading det fragments: when the first det-family token
         * in the output is a CLOSER (<|/det|>), the opening <|det|> was lost
         * (the model sometimes emits doubled closers) — drop each such
         * fragment through its closer. Loop handles consecutive leftovers. */
        {
            int pass = 0;
            while (pass++ < 4) {
                int opener_pos = -1, closer_pos = -1;
                for (int i = 0; i + 8 <= out_len; i++) {
                    if (memcmp(output + i, "<|det|>", 7) == 0) { opener_pos = i; break; }
                    if (memcmp(output + i, "<|/det|>", 8) == 0) { closer_pos = i; break; }
                }
                if (closer_pos < 0 || opener_pos >= 0) break;
                int skip = closer_pos + 8;
                while (skip < out_len && (output[skip] == '\n' || output[skip] == ' ')) skip++;
                if (skip >= out_len) { output[0] = '\0'; out_len = 0; break; }
                memmove(output, output + skip, out_len - skip + 1);
                out_len -= skip;
            }
        }

        /* Strip leading orphaned HTML closing-tag run (e.g. "</td></tr></table>").
         * Same rationale as the streaming suppression: output never begins
         * mid-table, so an opening run of closing tags is structural noise. */
        {
            int p = 0;
            while (p < out_len) {
                char c = output[p];
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { p++; continue; }
                if (c == '<' && p + 1 < out_len && output[p + 1] == '/') {
                    int gt = p + 2;
                    while (gt < out_len && output[gt] != '>') gt++;
                    if (gt >= out_len) break;
                    p = gt + 1;
                    continue;
                }
                break;
            }
            if (p > 0) {
                if (p >= out_len) {
                    /* Entire output was closing tags/whitespace */
                    output[0] = '\0';
                    out_len = 0;
                } else {
                    while (p < out_len && (output[p] == '\n' || output[p] == ' ' ||
                                           output[p] == '\r' || output[p] == '\t'))
                        p++;
                    memmove(output, output + p, out_len - p + 1);
                    out_len -= p;
                }
            }
        }
    }

    return output;
}

/* ds_strip_det_tags — V3后处理：去除检测坐标标签
 *
 * V3 (Unlimited-OCR) 输出格式:
 *   "实际文字<|det|>text [bbox]<|/det|>更多文字<|ref|>...<|/ref|><|det|>...<|/det|>..."
 * 用户只需要文字，不需要检测框坐标，所以需要去除 det/ref 标签。
 *
 * 算法: 读写双指针法
 *   read指针扫描原文，write指针写入过滤后结果
 *   遇到<|det|>...<|/det|> → 跳过整个块
 *   遇到<|ref|>...<|/ref|><|det|>...<|/det|> → 跳过整个ref+det对
 *   其他文本 → 直接复制
 *
 * 后处理: 合并连续多个换行为最多2个
 *
 * Python对应: outputs.replace(a_match_other, '')
 */
int ds_strip_det_tags(char *text, int len) {
    if (!text || len <= 0) return len;

    int read = 0, write = 0;
    while (read < len) {
        /* Check for <|ref|>...<|/ref|><|det|>...<|/det|> pattern (ref+det pair) */
        if (read + 7 < len && memcmp(text + read, "<|ref|>", 7) == 0) {
            /* Find matching </ref> */
            int ref_end = -1;
            for (int i = read + 7; i + 8 <= len; i++) {
                if (memcmp(text + i, "<|/ref|>", 8) == 0) {
                    ref_end = i + 8;
                    break;
                }
            }
            if (ref_end > 0 && ref_end + 6 < len && memcmp(text + ref_end, "<|det|>", 7) == 0) {
                /* Find matching </det> */
                int det_end = -1;
                for (int i = ref_end + 7; i + 8 <= len; i++) {
                    if (memcmp(text + i, "<|/det|>", 8) == 0) {
                        det_end = i + 8;
                        break;
                    }
                }
                if (det_end > 0) {
                    read = det_end;  /* Skip entire ref+det pair */
                    continue;
                }
            }
            /* If ref doesn't form a valid pair, keep it as-is */
            text[write++] = text[read++];
        }
        /* Check for standalone <|det|>...<|/det|> pattern */
        else if (read + 6 < len && memcmp(text + read, "<|det|>", 7) == 0) {
            /* Find matching </det> */
            int det_end = -1;
            for (int i = read + 7; i + 8 <= len; i++) {
                if (memcmp(text + i, "<|/det|>", 8) == 0) {
                    det_end = i + 8;
                    break;
                }
            }
            if (det_end > 0) {
                read = det_end;  /* Skip entire det block */
                continue;
            }
            /* If no matching </det>, keep as-is */
            text[write++] = text[read++];
        } else {
            text[write++] = text[read++];
        }
    }

    text[write] = '\0';

    /* Clean up multiple consecutive newlines left after tag removal */
    int w2 = 0, r2 = 0;
    int prev_newline = 0;
    while (r2 < write) {
        if (text[r2] == '\n') {
            if (prev_newline < 2) {
                text[w2++] = text[r2];
            }
            prev_newline++;
            r2++;
        } else {
            text[w2++] = text[r2++];
            prev_newline = 0;
        }
    }
    text[w2] = '\0';

    return w2;
}
