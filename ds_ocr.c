/*
 * ds_ocr.c - DeepSeek-OCR Pure C Inference Engine
 *
 * Main coordinator: image → visual tokenizer → encoder → MoE decoder → text
 */

#include "ds_ocr.h"
#include "ds_kernels.h"
#include "ds_safetensors.h"
#include "ds_image.h"
#include "ds_visual_tokenizer.h"
#include "ds_deep_encoder.h"
#include "ds_moe_decoder.h"
#include "ds_tokenizer.h"

#include "ds_dump.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>

int ds_verbose = 1;
int ds_bf16_simulate_python = 1;  /* BF16 intermediate truncation ON by default.
                                     Matches Python BF16 computation path.
                                     Without this, F32 precision diverges from Python
                                     after 12 MoE layers, reducing EOS logit and causing
                                     hallucinated output after OCR text is complete. */
int g_dump_crop_id = -1;  /* Crop ID for per-crop dumps */

/* ========================================================================
 * Timing Helper
 * ======================================================================== */

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ds_time_sec now provided by ds_kernels.h (inline) */

/* ========================================================================
 * Parallel crop encoding (for multi-crop SAM + encoder)
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
 * Configuration Detection
 * ======================================================================== */

static int detect_model_version(const char *model_dir) {
    /* Check for DeepEncoder V2 indicators (Qwen2-based encoder) */
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "detect_model_version: cannot open %s\n", path);
        return -1;
    }

    /* Simple detection: search for "DeepEncoderV2" or "causal_flow" in config */
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
}

/* ========================================================================
 * Weight Loading
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

    /* SAM neck — V1 uses neck.{0,1,2,3}.{weight,bias}, V2 omits Conv biases */
    LOAD_F32("model.sam_model.neck.0.weight", vt->sam_neck_conv1_weight);
    LOAD_F32("model.sam_model.neck.0.bias", vt->sam_neck_conv1_bias);
    LOAD_F32("model.sam_model.neck.1.weight", vt->sam_neck_ln1_weight);
    LOAD_F32("model.sam_model.neck.1.bias", vt->sam_neck_ln1_bias);
    LOAD_F32("model.sam_model.neck.2.weight", vt->sam_neck_conv2_weight);
    LOAD_F32("model.sam_model.neck.2.bias", vt->sam_neck_conv2_bias);
    LOAD_F32("model.sam_model.neck.3.weight", vt->sam_neck_ln2_weight);
    LOAD_F32("model.sam_model.neck.3.bias", vt->sam_neck_ln2_bias);

    /* SAM downsample — V1 uses net_2.0/net_3.0, V2 uses net_2/net_3 */
    LOAD_F32("model.sam_model.net_2.0.weight", vt->sam_net2_weight);
    LOAD_F32("model.sam_model.net_2.0.bias", vt->sam_net2_bias);
    LOAD_F32("model.sam_model.net_3.0.weight", vt->sam_net3_weight);
    LOAD_F32("model.sam_model.net_3.0.bias", vt->sam_net3_bias);
    /* V2 naming (no .0 suffix, no bias) */
    if (!vt->sam_net2_weight)
        LOAD_F32("model.sam_model.net_2.weight", vt->sam_net2_weight);
    if (!vt->sam_net3_weight)
        LOAD_F32("model.sam_model.net_3.weight", vt->sam_net3_weight);

    /* Learnable tokens — image_newline is V1 only, view_seperator is shared */
    LOAD_F32("model.image_newline", vt->image_newline);
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

        LOAD_F32("model.vision_model.post_layernorm.weight", clip->final_norm_weight);
        LOAD_F32("model.vision_model.post_layernorm.bias", clip->final_norm_bias);
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

        DEC_F32(q_norm_weight, "self_attn.q_norm.weight");
        DEC_F32(k_norm_weight, "self_attn.k_norm.weight");
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

    if (ds_verbose >= 1)
        fprintf(stderr, "All weights loaded\n");

    return 0;
}

/* ========================================================================
 * Context Allocation
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
    ctx->min_new_tokens = 0;        /* Don't suppress EOS — let model stop naturally.
                                      With BF16 intermediate truncation (ds_bf16_simulate_python=1),
                                      the numerical path matches Python's, and EOS is output correctly.
                                      Python generate() has no min_new_tokens; it stops at EOS. */

    return 0;
}

/* ========================================================================
 * Public API
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

    if (ds_verbose >= 1)
        fprintf(stderr, "Model loaded successfully (version %d, %d layers)\n",
                version, ctx->config.dec_layers);

    return ctx;
}

void ds_free(ds_ctx_t *ctx) {
    if (!ctx) return;

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
        /* V2 multi-crop path */
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
            crops = ds_dynamic_preprocess(&img, 768, 1, 12, 0, &n_crops);
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
            float *global_sam = ds_sam_forward_image(ctx, global_img, &n_sam_tokens, NULL);
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
            float *global_enc = ds_encoder_forward_v2(ctx, global_sam, n_sam_tokens,
                                                       &n_global_enc, 256,
                                                       ctx->vis_tokenizer.causal_query_embeddings);
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
            free(tasks); free(threads);
            ds_image_free(global_img);

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

    } else {
        /* V1 or single-image V2 path (no cropping) */
        /* Step 1: Visual tokenizer (SAM) — resize to 1024x1024 */
        int n_visual_tokens;
        float *patch_embeds = NULL;
        float *visual_tokens = ds_visual_tokenizer_forward(ctx, pixels, width, height, channels,
                                                            &n_visual_tokens, &patch_embeds);
        if (!visual_tokens) {
            fprintf(stderr, "Visual tokenizer failed\n");
            return NULL;
        }

        /* Step 2: Encoder (CLIP V1 or DeepEncoder V2) */
        if (cfg->enc_type == 1) {
            /* CLIP encoder needs the full-resolution SAM patch_embeds [768, n_full_patches].
             * n_visual_tokens is the downsampled count (256), but patch_embeds has
             * 4096 patches (64x64 grid) from the SAM patch embedding layer.
             * Calculate the actual patch count from patch_embeds dimensions.
             * patch_embeds layout: [768, n_full_patches] in CHW format.
             * For 1024x1024 input: n_full_patches = (1024/16)^2 = 4096. */
            int n_full_patches = 64 * 64; /* Fixed for 1024x1024 SAM input */
            encoder_output = ds_clip_encoder_forward(ctx, patch_embeds,
                                                      n_full_patches, visual_tokens,
                                                      n_visual_tokens, &n_encoder_tokens);
        } else {
            encoder_output = ds_encoder_forward_v2(ctx, visual_tokens, n_visual_tokens,
                                                    &n_encoder_tokens,
                                                    cfg->enc_causal_flow_queries,
                                                    ctx->vis_tokenizer.causal_query_embeddings);
        }
        free(patch_embeds);
        free(visual_tokens);
        if (!encoder_output) {
            fprintf(stderr, "Encoder forward failed\n");
            return NULL;
        }

        /* Debug: dump C encoder output for comparison */
        {
            const char *dump_dir = getenv("DS_DUMP_ENCODER");
            if (dump_dir) {
                char path[512];
                snprintf(path, sizeof(path), "%s/c_encoder_output.bin", dump_dir);
                FILE *f = fopen(path, "wb");
                if (f) {
                    fwrite(encoder_output, sizeof(float), n_encoder_tokens * hidden, f);
                    fclose(f);
                    fprintf(stderr, "Dumped C encoder output (%d x %d) to %s\n",
                            n_encoder_tokens, hidden, path);
                }
            }
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

    /* Step 3: Build decoder input sequence */

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

    /* Load tokenizer early (needed for prompt encoding in V2) */
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/vocab.json", model_dir_str);
    tokenizer = ds_tokenizer_load(tokenizer_path);
    if (!tokenizer && ds_verbose >= 1)
        fprintf(stderr, "Note: vocab.json not found, tokenizer not loaded\n");

    float *input_embeds = NULL;
    int prefix_len = 0;

    if (cfg->model_version == 2) {
        /* V2 prompt format matching Python model.infer() with sft_format='plain':
         * Python prompt = '<image>\nFree OCR. '
         * With multi-crop: encoder_output already contains [local, global, view_sep]
         * C equivalent: [BOS] + encoder_output(n_encoder_tokens) + '\nFree OCR. '(5 tokens)
         * Note: PLAIN template has no role tokens — roles=("", ""), sep=""
         */
        /* Compute image_size=640 token layout (must match Python model.infer() default) */
        int ds_image_size = 640;
        int ds_num_queries = (ds_image_size + 15) / 16 / 4;  /* ceil(640/16/4) = 10 */
        int ds_local_tokens_per_crop = ds_num_queries * ds_num_queries;  /* 100 */
        int n_crops_count = (n_encoder_tokens - 257) / 144;  /* works for 1 or 6 crops */
        if (n_crops_count < 0) n_crops_count = 0;  /* safety */
        int n_local_slots = n_crops_count * ds_local_tokens_per_crop;
        int n_global_slots = 256;
        int n_sep_slots = 1;
        int n_img_tokens = n_global_slots + n_sep_slots + n_local_slots;
        int n_local_enc = n_encoder_tokens - 257;
        if (n_local_enc < 0) n_local_enc = 0;

        int n_text_after = 0;
        int *text_after_ids = NULL;
        if (tokenizer) {
            /* Encode text after image using BPE tokenizer.
             * Python format_messages strips each message's content and the final prompt,
             * so "<image>\nFree OCR. " (with trailing space) becomes "<image>\nFree OCR."
             * after strip(). Text after image = "\nFree OCR." (no trailing space).
             * Expected: [201, 21431, 126041, 16] (4 tokens) */
            text_after_ids = ds_tokenizer_encode(tokenizer, "\nFree OCR.", &n_text_after);
            if (!text_after_ids || n_text_after <= 0) {
                /* Fallback: use verified hardcoded IDs if tokenizer fails */
                static const int fallback_ids[] = {201, 21431, 126041, 16};
                n_text_after = 4;
                text_after_ids = (int *)malloc(n_text_after * sizeof(int));
                memcpy(text_after_ids, fallback_ids, n_text_after * sizeof(int));
            }
            if (ds_verbose >= 1) {
                fprintf(stderr, "Prompt: BOS + img(%d) + after(%d) = %d tokens\n",
                        n_img_tokens, n_text_after, 1 + n_img_tokens + n_text_after);
            }
        }

        /* prefix = BOS(1) + image_tokens(n_img_tokens) + text_after(n_text_after)
         * Note: n_img_tokens may differ from n_encoder_tokens when image_size=640
         * because some encoder tokens are dropped by Python's masked_scatter. */
        /* (Variables already computed above: n_img_tokens, n_local_slots, n_local_enc, etc.) */

        if (ds_verbose >= 1) {
            fprintf(stderr, "Image size config: image_size=%d, num_queries=%d, "
                   "local_tokens/crop=%d, crops=%d\n",
                   ds_image_size, ds_num_queries, ds_local_tokens_per_crop, n_crops_count);
            fprintf(stderr, "Token layout: %d image tokens from local_enc[0:%d] "
                   "(encoder has %d local, dropping %d + global/sep dropped)\n",
                   n_img_tokens, n_img_tokens,
                   n_local_enc, n_local_enc - n_img_tokens);
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

        /* 2. Encoder output tokens — layout must match Python model.forward() behavior.
         *
         * Python constructs: source = cat([local_features(864), global_features(256), view_sep(1)])
         *   → 1121 elements total (after projector, dim=1280)
         *
         * Then masked_scatter_(images_seq_mask, source) fills 857 True positions
         * with source[0:857] = local_features[0:857].
         *
         * With image_size=640: num_queries=10, local_tokens_per_crop=100
         * Image positions = 857 (from image_size*image_size/14/14 = 640*640/196 ≈ 2088?
         *   No — it's image_size/16/4=10, so local_slots = 6*100=600, plus global=256 + sep=1 = 857)
         * 
         * Key insight: ALL 857 image positions are filled with local_enc[0:857].
         * The global features and sep in the source are AFTER local[864], so they
         * land at source[864:1121] which is BEYOND the 857 True positions — they're dropped.
         * Local_enc[857:864] (7 tokens, last ~1.17 per crop) are also dropped.
         *
         * So C simply copies local_enc[0:857] into positions 1-857. */
        /* (Variables already computed above: n_img_tokens, n_local_slots, n_local_enc, etc.) */

        /* Debug: dump encoder_output for Python decoder comparison */
        if (getenv("DS_DUMP_ENCODER")) {
            FILE *df = fopen("dump/c_encoder_output.bin", "wb");
            if (df) { fwrite(encoder_output, sizeof(float), n_encoder_tokens * hidden, df); fclose(df); }
            fprintf(stderr, "Dumped C encoder output: %d tokens x %d dim to dump/c_encoder_output.bin\n",
                    n_encoder_tokens, hidden);
        }

        /* Apply Python model's masked_scatter layout:
         *
         * Python constructs: tokenized_image = [global(256), sep(1), local(N)]
         * These become True positions in images_seq_mask.
         * Source for masked_scatter = cat([local_features, global_features, view_sep])
         * So source = [local(n_local_enc), global(n_global_enc), sep(1)].
         *
         * masked_scatter fills True positions IN ORDER with source elements:
         *   pos 0..255 (global slots)  <- source[0:256]  = local[0:256]
         *   pos 256   (sep slot)       <- source[256]    = local[256]
         *   pos 257..(n_img_tokens-1)  <- source[257:n_img_tokens]
         *     = if n_img_tokens <= n_local_enc: local[257:n_img_tokens]
         *     = if n_img_tokens >  n_local_enc: local[257:n_local_enc] + global[0:n_img_tokens-n_local_enc]
         *
         * Since encoder_output layout = [local, global, sep], we can simply copy
         * encoder_output[0:n_img_tokens] which matches source[0:n_img_tokens]. */
        float *enc = encoder_output;
        int img_pos = pos;  /* starts at 1 (after BOS) */

        /* Copy source[0:n_img_tokens] = enc[0:n_img_tokens] into image positions */
        int n_copy = n_img_tokens < n_encoder_tokens ? n_img_tokens : n_encoder_tokens;
        memcpy(input_embeds + img_pos * hidden, enc, n_copy * hidden * sizeof(float));
        img_pos += n_copy;
        /* If n_img_tokens > n_encoder_tokens (shouldn't happen),
         * remaining positions stay zero. */
        pos = img_pos;

        /* 3. Text after image: "\nFree OCR. " (5 tokens) */
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
         * Image token layout (from Python infer):
         *   num_queries_base = ceil(1024/16/4) = 16
         *   tokenized_image = ([128815]*16 + [128815]) * 16 + [128815] = 273
         *
         * The CLIP+SAM+Projector encoder now returns 273 tokens:
         *   - 256 projected tokens (16x16 grid of CLIP+SAM features)
         *   - 16 image_newline tokens (inserted after each grid row)
         *   - 1 view_seperator token (appended at end)
         *
         * Python's masked_scatter places all 273 encoder tokens into the
         * 273 image positions (all True in images_seq_mask).
         * For C: embed all image positions with 128815, then overwrite
         * ALL 273 with encoder_output (n_encoder_tokens should equal n_img_tokens).
         */
        int base_size = 1024;
        int num_queries_base = (base_size + 15) / 16 / 4;  /* 16 */
        int n_img_tokens = (num_queries_base + 1) * num_queries_base + 1;  /* 273 */

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
        /* V1 prompt format: [image_start][encoder_output][image_end] */
        prefix_len = n_encoder_tokens + 2;
        input_embeds = (float *)malloc(prefix_len * hidden * sizeof(float));
        memset(input_embeds, 0, prefix_len * hidden * sizeof(float));

        /* Embed image_start token */
        if (ctx->decoder.tok_embeddings_bf16) {
            const uint16_t *emb_start = ctx->decoder.tok_embeddings_bf16 + (size_t)DS_TOKEN_IMAGE_START * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb_start[i]) << 16;
                memcpy(&input_embeds[i], &f32_bits, sizeof(float));
            }
        }

        /* Encoder output */
        memcpy(input_embeds + hidden, encoder_output, n_encoder_tokens * hidden * sizeof(float));

        /* Embed image_end token */
        if (ctx->decoder.tok_embeddings_bf16) {
            const uint16_t *emb_end = ctx->decoder.tok_embeddings_bf16 + (size_t)DS_TOKEN_IMAGE_END * hidden;
            float *last_pos = input_embeds + (prefix_len - 1) * hidden;
            for (int i = 0; i < hidden; i++) {
                uint32_t f32_bits = ((uint32_t)emb_end[i]) << 16;
                memcpy(&last_pos[i], &f32_bits, sizeof(float));
            }
        }
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

    /* Step 4: Reset KV cache and prefill (all but last token),
     * then use decoder_forward with last token to get first predicted token */
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
                if (ds_verbose >= 1)
                    fprintf(stderr, "Prefill first token=%d (from prefill logits, %.3f)\n", first_token, best_val);
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

    /* Step 5: Autoregressive decoding */

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
                /* Ban EOS token and find next best */
                logits[eos_token] = -1e30f;
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

                /* Stream token */
                if (ctx->token_cb) ctx->token_cb(piece, ctx->token_cb_userdata);
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

    return output;
}
