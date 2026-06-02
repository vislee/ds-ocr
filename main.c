/*
 * main.c - CLI entry point for DeepSeek-OCR
 *
 * Usage: ds_ocr -d <model_dir> -i <input.png> [options]
 */

#include "ds_ocr.h"
#include "ds_kernels.h"
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

    if (!model_dir || !input_image) {
        usage(argv[0]);
        return 1;
    }

    ds_verbose = verbosity;

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
        if (ctx->perf_total_ms > 0) {
            tokens_per_sec = (1000.0 * ctx->perf_text_tokens) / ctx->perf_total_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (%.2f tok/s, encoding: %.0fms, decoding: %.0fms)\n",
                ctx->perf_total_ms, ctx->perf_text_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_decode_ms);
    }

    ds_free(ctx);
    return 0;
}
