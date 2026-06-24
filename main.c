/*
 * main.c - CLI entry point for DeepSeek-OCR
 *
 * Usage: ds_ocr -d <model_dir> -i <input.png> [options]
 */

#include "ds_ocr.h"
#include "ds_kernels.h"
#include "ds_platform_ocr.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Token streaming callback: print each piece as it's decoded */
static void stream_token(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);
    fflush(stdout);
}

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
    fprintf(stderr, "  --debug       Debug output (per-layer details)\n");
    fprintf(stderr, "  --silent      No status output (only recognition on stdout)\n");
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_image = NULL;
    int verbosity = 1;
    int n_threads = 0;
    int max_new_tokens = 4096;
    float temperature = 0.0f;
    float repeat_penalty = 1.0f;
    int no_repeat_ngram_size = 0;
    int min_new_tokens = -1;  /* -1 = use default from ctx */
    int use_platform_ocr = 0;
    int platform_accurate = 1;
    int profile = 0;

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

    if ((!model_dir && !use_platform_ocr) || !input_image) {
        usage(argv[0]);
        return 1;
    }

    ds_verbose = verbosity;
    ds_bf16_simulate_python = getenv("DS_BF16_SIMULATE_PYTHON") ? 1 : 0;

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

    /* Initialize thread pool */
    if (n_threads <= 0) n_threads = ds_get_num_cpus();
    ds_set_threads(n_threads);

    /* Load model */
    ds_ctx_t *ctx = ds_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Apply settings */
    ctx->max_new_tokens = max_new_tokens;
    ctx->temperature = temperature;
    ctx->repeat_penalty = repeat_penalty;
    ctx->no_repeat_ngram_size = no_repeat_ngram_size;
    if (min_new_tokens >= 0) ctx->min_new_tokens = min_new_tokens;
    if (profile) ctx->profile_enabled = 1;

    /* Unlimited-OCR (V3) defaults: ngram=35 for repeat suppression.
     * Python uses SlidingWindowNoRepeatNgramProcessor(ngram_size=35, window=128). */
    if (ctx->config.model_version == 3 && no_repeat_ngram_size == 0) {
        ctx->no_repeat_ngram_size = 35;
    }

    /* Set up streaming callback */
    if (verbosity > 0) {
        ds_set_token_callback(ctx, stream_token, NULL);
    }

    /* Recognize */
    char *text = ds_recognize(ctx, input_image);

    if (text) {
        if (verbosity == 0) {
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

    /* Per-layer profiler output */
    if (profile && ctx->profile_enabled) {
        double decode_tok = ctx->perf_decode_ms;
        int n_layers = ctx->config.dec_layers;
        int n_steps = ctx->perf_decode_steps;
        double avg_step = n_steps > 0 ? decode_tok / n_steps : 0;

        fprintf(stderr, "\n=== PROFILER (decode: %.0fms, %d steps, avg %.2fms/step) ===\n",
                decode_tok, n_steps, avg_step);

        /* Aggregate layer timing */
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

    ds_free(ctx);
    return 0;
}
