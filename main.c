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
#include "ds_platform_ocr.h"    /* macOS Vision OCR 后端（不使用模型时的备选方案） */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* usage — 打印帮助信息
 *
 * 各选项的详细说明:
 *   -d <dir>       模型目录，需包含 *.safetensors 和 config.json
 *   -i <file>      输入图像路径
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
    fprintf(stderr, "ds_ocr — DeepSeek-OCR document recognition (pure C)\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> -i <input.png> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with *.safetensors, config.json)\n");
    fprintf(stderr, "  -i <file>     Input image (PNG, JPEG, WebP, BMP, TIFF)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -t <n>        Number of threads (default: all CPUs)\n");
    fprintf(stderr, "  -n <n>        Max new tokens (default: 4096)\n");
    fprintf(stderr, "  --temp <f>    Sampling temperature (default: 0 = greedy)\n");
    fprintf(stderr, "  --rp <f>      Repetition penalty (default: 1.0, try 1.1-1.5)\n");
    fprintf(stderr, "  --ngram <n>   No-repeat ngram size (default: 0, try 20-35 if output is degenerate)\n");
    fprintf(stderr, "  --min-tokens <n> Min tokens before allowing EOS (default: 256, 0 to disable)\n");
    fprintf(stderr, "  --vision      Use macOS Vision OCR backend (no model required)\n");
    fprintf(stderr, "  --vision-fast Use macOS Vision OCR backend in fast mode\n");
    fprintf(stderr, "  --profile     Profile per-layer timing breakdown\n");
    fprintf(stderr, "  --int4        INT8 quantize MoE expert weights (2x less memory bandwidth)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer details)\n");
    fprintf(stderr, "  --silent      No status output (only recognition on stdout)\n");
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    /* ── 命令行参数变量（带默认值）── */
    const char *model_dir = NULL;       /* 模型目录路径 */
    const char *input_image = NULL;     /* 输入图像路径 */
    int verbosity = 1;                  /* 输出详细度: 0=静默, 1=正常, 2=调试 */
    int n_threads = 0;                  /* 线程数: 0=自动(=CPU核心数) */
    int max_new_tokens = 4096;          /* 最大生成token数 */
    float temperature = 0.0f;           /* 采样温度: 0=贪心(确定性) */
    float repeat_penalty = 1.0f;        /* 重复惩罚: 1.0=无惩罚 */
    int no_repeat_ngram_size = 0;       /* N-gram阻断: 0=禁用 */
    int min_new_tokens = -1;            /* 最少token数: -1=使用ctx默认值 */
    int use_platform_ocr = 0;           /* 使用macOS Vision OCR后端 */
    int platform_accurate = 1;          /* Vision OCR精度模式: 1=高精度, 0=快速 */
    int profile = 0;                    /* 启用逐层Profiler */
    int int4 = 0;                       /* 启用INT4/INT8量化 */

    /* ── 解析命令行参数 ── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_image = argv[++i];
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

    /* 参数校验: 至少需要模型目录或Vision模式, 以及输入图像 */
    if ((!model_dir && !use_platform_ocr) || !input_image) {
        usage(argv[0]);
        return 1;
    }

    /* 设置全局verbose级别（影响所有模块的日志输出） */
    ds_verbose = verbosity;
    /* BF16中间截断: 默认开启，模拟Python的BF16计算路径
     * 不开启时F32精度与Python的BF16训练路径不一致，12层MoE后EOS logit偏低，
     * 导致OCR文本结束后产生幻觉内容 */
    ds_bf16_simulate_python = getenv("DS_BF16_SIMULATE_PYTHON") ? 1 : 0;

    /* ── macOS Vision OCR 快捷路径（不使用模型） ── */
    if (use_platform_ocr) {
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
     * 这防止V3在长文档解析时产生重复的n-gram */
    if (ctx->config.model_version == 3 && no_repeat_ngram_size == 0) {
        ctx->no_repeat_ngram_size = 35;
    }

    /* ── 设置流式输出回调 ──
     * V3 禁用流式输出: 因为V3可能产生幻觉前缀（如"[No text detected]"），
     * 需要后处理删除，这无法逐token完成——必须等全部生成完毕后再处理。
     * V1/V2: 启用流式输出，边生成边打印 */
    if (verbosity > 0 && ctx->config.model_version != 3) {
        ds_set_token_callback(ctx, stream_token, NULL);
    }

    /* ── 执行OCR识别 ──
     * ds_recognize() 完成完整的推理流水线:
     *   图像加载 → SAM编码 → 编码器(CLIP/DeepEncoder) → 投影 → 构造Prompt
     *   → Prefill(批量填充KV缓存) → Decode循环(逐token生成) → 文本输出
     */
    char *text = ds_recognize(ctx, input_image);

    if (text) {
        /* V3 或静默模式: 一次性打印完整结果
         * V1/V2 非静默: 结果已通过stream_token逐token打印，此处只打换行 */
        if (verbosity == 0 || ctx->config.model_version == 3) {
            printf("%s\n", text);
        } else {
            printf("\n");
        }
        free(text);
    } else {
        fprintf(stderr, "Recognition failed\n");
        ds_free(ctx);
        return 1;
    }

    /* ── 性能统计输出 ── */
    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_decode_ms > 0) {
            tokens_per_sec = (1000.0 * ctx->perf_text_tokens) / ctx->perf_decode_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (%.2f tok/s decode)\n"
                "  Encoding: %.0f ms | Prefill: %.0f ms | Decode: %.0f ms\n",
                ctx->perf_total_ms, ctx->perf_text_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_prefill_ms, ctx->perf_decode_ms);
    }

    /* ── 逐层Profiler输出（--profile 启用） ──
     *
     * 典型输出格式:
     *   === PROFILER (decode: 5277ms, 226 steps, avg 23.35ms/step) ===
     *     QKV proj:     241.6 ms (  4.6%)    ← QKV投影耗时
     *     Attention:    524.2 ms (  9.9%)    ← 注意力计算耗时
     *     Out proj:     168.2 ms (  3.2%)    ← 输出投影耗时
     *     MLP/MoE:     3890.3 ms ( 73.7%)    ← ★ MoE/FFN是绝对瓶颈
     *     LM head:      365.8 ms (  6.9%)    ← 输出层投影耗时
     *     Sampling:      90.2 ms (  1.7%)    ← 采样+重复惩罚耗时
     *
     * 解读: MLP/MoE占73.7% → 优化MoE可获得最大收益（Amdahl定律）
     * 这正是v0.5→v0.9优化的核心方向：连续expert块+madvise预取+fused gate+up
     */
    if (profile && ctx->profile_enabled) {
        double decode_tok = ctx->perf_decode_ms;
        int n_layers = ctx->config.dec_layers;
        int n_steps = ctx->perf_decode_steps;
        double avg_step = n_steps > 0 ? decode_tok / n_steps : 0;

        fprintf(stderr, "\n=== PROFILER (decode: %.0fms, %d steps, avg %.2fms/step) ===\n",
                decode_tok, n_steps, avg_step);

        /* 汇总各层耗时 — 累加所有层的QKV/Attn/Proj/MLP时间 */
        double total_qkv = 0, total_attn = 0, total_proj = 0, total_mlp = 0;
        for (int l = 0; l < n_layers; l++) {
            total_qkv += ctx->perf_layer_qkv_ms[l];
            total_attn += ctx->perf_layer_attn_ms[l];
            total_proj += ctx->perf_layer_proj_ms[l];
            total_mlp += ctx->perf_layer_mlp_ms[l];
        }

        /* 输出各阶段汇总（百分比 = 阶段耗时/总decode耗时） */
        fprintf(stderr, "  QKV proj:   %7.1f ms (%5.1f%%)\n", total_qkv, decode_tok > 0 ? 100.0 * total_qkv / decode_tok : 0);
        fprintf(stderr, "  Attention:  %7.1f ms (%5.1f%%)\n", total_attn, decode_tok > 0 ? 100.0 * total_attn / decode_tok : 0);
        fprintf(stderr, "  Out proj:   %7.1f ms (%5.1f%%)\n", total_proj, decode_tok > 0 ? 100.0 * total_proj / decode_tok : 0);
        fprintf(stderr, "  MLP/MoE:    %7.1f ms (%5.1f%%)\n", total_mlp, decode_tok > 0 ? 100.0 * total_mlp / decode_tok : 0);
        fprintf(stderr, "  LM head:    %7.1f ms (%5.1f%%)\n", ctx->perf_lm_head_ms, decode_tok > 0 ? 100.0 * ctx->perf_lm_head_ms / decode_tok : 0);
        fprintf(stderr, "  Sampling:   %7.1f ms (%5.1f%%)\n", ctx->perf_sampling_ms, decode_tok > 0 ? 100.0 * ctx->perf_sampling_ms / decode_tok : 0);

        /* 逐层详细分解（每步平均耗时 ms/step） */
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

    /* ── 释放模型资源 ── */
    ds_free(ctx);
    return 0;
}