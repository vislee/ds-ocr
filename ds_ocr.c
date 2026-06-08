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
#include <sys/time.h>
#include <sys/stat.h>

int ds_verbose = 1;

/* ========================================================================
 * Timing Helper
 * ======================================================================== */

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static double ds_time_sec(void) {
    return now_ms() / 1000.0;
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

    /* KV cache: [layers, max_seq, kv_dim] */
    int max_seq = 4096; /* Max sequence length for KV cache */
    ctx->kv_cache_max = max_seq;
    ctx->kv_cache_len = 0;

    size_t kv_size = (size_t)cfg->dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, kv_size);
    ctx->kv_cache_v = (float *)calloc(1, kv_size);
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;

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
    ctx->no_repeat_ngram_size = 20; /* Match Python default */

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

    /* Free decoder buffers */
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

        if (use_crop) {
            crops = ds_dynamic_preprocess(&img, 768, 1, 12, 0, &n_crops);
            if (!crops || n_crops < 1) {
                fprintf(stderr, "dynamic_preprocess failed\n");
                return NULL;
            }
        }

        if (!use_crop) {
            /* Small image (both dims <= 768): process as 1 local crop + 1 global view.
             * Python always processes both: patches (768x768) and image_ori (1024x1024).
             * For small images, dynamic_preprocess returns 1 crop (768x768 padded).
             * Then the global view is padded to 1024x1024 separately. */
            if (ds_verbose >= 1)
                fprintf(stderr, "Small image %dx%d: 1 crop (768x768) + global (1024x1024)\n",
                        width, height);

            /* Local: pad to 768x768 */
            ds_image_t *local_img = ds_image_pad(&img, 768, 127);
            if (!local_img) return NULL;

            int n_sam_tokens;
            float *sam_tokens = ds_sam_forward_image(ctx, local_img, &n_sam_tokens, NULL);
            ds_image_free(local_img);
            if (!sam_tokens) return NULL;

            int n_enc_tokens;
            float *local_enc = ds_encoder_forward_v2(ctx, sam_tokens, n_sam_tokens,
                                                      &n_enc_tokens, 144,
                                                      ctx->vis_tokenizer.causal_query_768_embeddings);
            free(sam_tokens);
            if (!local_enc) {
                fprintf(stderr, "Encoder forward failed for small local image\n");
                return NULL;
            }

            /* Global: pad to 1024x1024 */
            ds_image_t *global_img = ds_image_pad(&img, 1024, 127);
            if (!global_img) { free(local_enc); return NULL; }

            float *global_sam = ds_sam_forward_image(ctx, global_img, &n_sam_tokens, NULL);
            ds_image_free(global_img);
            if (!global_sam) { free(local_enc); return NULL; }

            int n_global_enc;
            float *global_enc = ds_encoder_forward_v2(ctx, global_sam, n_sam_tokens,
                                                       &n_global_enc, 256,
                                                       ctx->vis_tokenizer.causal_query_embeddings);
            free(global_sam);
            if (!global_enc) { free(local_enc); return NULL; }

            /* Concat: [local(144), global(256), view_sep(1)] */
            n_encoder_tokens = n_enc_tokens + n_global_enc + 1;
            encoder_output = (float *)malloc(n_encoder_tokens * hidden * sizeof(float));
            memcpy(encoder_output, local_enc, n_enc_tokens * hidden * sizeof(float));
            memcpy(encoder_output + n_enc_tokens * hidden, global_enc, n_global_enc * hidden * sizeof(float));
            if (ctx->vis_tokenizer.view_seperator) {
                memcpy(encoder_output + (n_enc_tokens + n_global_enc) * hidden,
                       ctx->vis_tokenizer.view_seperator, hidden * sizeof(float));
            } else {
                memset(encoder_output + (n_enc_tokens + n_global_enc) * hidden, 0, hidden * sizeof(float));
            }
            free(local_enc);
            free(global_enc);
        } else {
            /* Large image: multi-crop encoding */
            if (ds_verbose >= 1)
                fprintf(stderr, "Multi-crop: %d local crops (768x768)\n", n_crops);

            /* Step 2: Encode each local crop (768x768) */
            int tokens_per_crop = 144;  /* 48/4 * 48/4 = 12*12 = 144 */
            float *local_features = NULL;
            int local_token_count = 0;

            for (int i = 0; i < n_crops; i++) {  /* all are 768x768 local crops */
                if (ds_verbose >= 1)
                    fprintf(stderr, "Encoding local crop %d/%d\n", i + 1, n_crops);

                double crop_t0 = ds_time_sec();

                int n_sam_tokens;
                float *sam_tokens = ds_sam_forward_image(ctx, crops[i], &n_sam_tokens, NULL);
                double sam_t = ds_time_sec() - crop_t0;
                if (!sam_tokens) {
                    fprintf(stderr, "SAM forward failed for crop %d\n", i);
                    goto cleanup_crops;
                }

                /* Dump SAM output for first crop if DS_DUMP_TENSORS set */
                if (i == 0 && getenv("DS_DUMP_TENSORS")) {
                    FILE *df = fopen("dump/local_crop0_sam_output.bin", "wb");
                    if (df) { fwrite(sam_tokens, sizeof(float), n_sam_tokens * 896, df); fclose(df); }
                }

                /* Encoder: Qwen2 with 144 causal flow queries (includes projector) */
                double enc_t0 = ds_time_sec();
                int n_enc_tokens;
                float *enc_tokens = ds_encoder_forward_v2(ctx, sam_tokens, n_sam_tokens,
                                                           &n_enc_tokens,
                                                           tokens_per_crop,
                                                           ctx->vis_tokenizer.causal_query_768_embeddings);
                double enc_t = ds_time_sec() - enc_t0;
                free(sam_tokens);
                if (!enc_tokens) {
                    fprintf(stderr, "Encoder forward failed for crop %d\n", i);
                    goto cleanup_crops;
                }

                if (ds_verbose >= 1)
                    fprintf(stderr, "  crop %d: SAM %.2fs + Encoder %.2fs = %.2fs\n",
                            i + 1, sam_t, enc_t, sam_t + enc_t);

                /* Dump crop encoder output for comparison if DS_DUMP_TENSORS set */
                if (getenv("DS_DUMP_TENSORS")) {
                    char dump_path[256];
                    snprintf(dump_path, sizeof(dump_path), "dump/local_crop%d_proj_output.bin", i);
                    FILE *df = fopen(dump_path, "wb");
                    if (df) { fwrite(enc_tokens, sizeof(float), n_enc_tokens * hidden, df); fclose(df); }
                }

                /* Quick debug: after first crop, dump and exit early */
                if (i == 0 && getenv("DS_DUMP_FIRST_CROP")) {
                    FILE *df = fopen("dump/c_crop0_proj_output.bin", "wb");
                    if (df) { fwrite(enc_tokens, sizeof(float), n_enc_tokens * hidden, df); fclose(df); }
                    fprintf(stderr, "DS_DUMP_FIRST_CROP: dumped crop 0, exiting\n");
                    goto cleanup_crops;
                }

                /* Append to local_features */
                int old_count = local_token_count;
                local_token_count += n_enc_tokens;
                local_features = (float *)realloc(local_features, local_token_count * hidden * sizeof(float));
                memcpy(local_features + old_count * hidden, enc_tokens, n_enc_tokens * hidden * sizeof(float));
                free(enc_tokens);
            }

            /* Step 3: Encode global image (1024x1024)
             * Python: global_features_1 = sam_model(image_ori) where image_ori is 1024x1024
             * Note: ds_dynamic_preprocess thumbnail is 768x768, NOT 1024x1024.
             * We need to pad the original image to 1024x1024 separately. */
            if (ds_verbose >= 1)
                fprintf(stderr, "Encoding global image (1024x1024)\n");

            ds_image_t *global_img = ds_image_pad(&img, 1024, 127);
            if (!global_img) {
                free(local_features);
                goto cleanup_crops;
            }

            int n_global_sam_tokens;
            float *global_sam_tokens = ds_sam_forward_image(ctx, global_img, &n_global_sam_tokens, NULL);
            ds_image_free(global_img);
            if (!global_sam_tokens) {
                fprintf(stderr, "SAM forward failed for global image\n");
                free(local_features);
                goto cleanup_crops;
            }

            int n_global_enc_tokens;
            int global_queries = 256;  /* 64/4 * 64/4 = 16*16 = 256 */
            float *global_enc_tokens = ds_encoder_forward_v2(ctx, global_sam_tokens, n_global_sam_tokens,
                                                              &n_global_enc_tokens,
                                                              global_queries,
                                                              ctx->vis_tokenizer.causal_query_embeddings);
            free(global_sam_tokens);

            /* Step 4: Concatenate [local, global, view_separator] — matches Python
             * Python uses masked_scatter_ with source=[local, global, view_sep]
             * into mask positions ordered [global, view_sep, local].
             * The net effect is that embedding positions get filled with
             * [local[0:256], local[256], local[257:864]+global[0:256]+view_sep].
             * Since C code directly places encoder_output into embed positions,
             * we use the same source order: [local, global, view_sep]. */
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
            encoder_output = ds_clip_encoder_forward(ctx, patch_embeds,
                                                      n_visual_tokens, visual_tokens,
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
             * Python format_messages strips the prompt, so "\nFree OCR. " (with
             * trailing space) becomes "\nFree OCR." after strip().
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
            fprintf(stderr, "Token layout: %d global + %d sep + %d local = %d image tokens "
                   "(encoder has %d local, dropping %d)\n",
                   n_global_slots, n_sep_slots, n_local_slots, n_img_tokens,
                   n_local_enc, n_local_enc - n_local_slots);
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
         * CRITICAL: The Python model uses image_size=640 (not 768) for token layout.
         * With image_size=640: num_queries=10, local_tokens_per_crop=100
         * Total image tokens: 256 (global) + 1 (sep) + 600 (local) = 857
         *
         * But the encoder ALWAYS produces 144 tokens per crop (causal_query_768
         * is nn.Embedding(144, 896)). So we have 864 local encoder tokens
         * but only 600 image token positions. Python's masked_scatter silently
         * drops the extra 264 tokens (the last 7 from each crop).
         *
         * Python's masked_scatter fills image token positions with source=[local(864), global(256), sep(1)]:
         *   global slots (pos 1-256)   <- source[0:256]   = local[0:256]
         *   sep slot (pos 257)         <- source[256]      = local[256]
         *   local slots (pos 258-857)  <- source[257:857]  = local[257:600]+partial
         *   (source[857:] = local[857:864] + global + sep are DROPPED)
         *
         * We must replicate this exact layout. */
        /* (Variables already computed above: n_img_tokens, n_local_slots, n_local_enc, etc.) */

        /* Debug: dump encoder_output for Python decoder comparison */
        if (getenv("DS_DUMP_ENCODER")) {
            FILE *df = fopen("dump/c_encoder_output.bin", "wb");
            if (df) { fwrite(encoder_output, sizeof(float), n_encoder_tokens * hidden, df); fclose(df); }
            fprintf(stderr, "Dumped C encoder output: %d tokens x %d dim to dump/c_encoder_output.bin\n",
                    n_encoder_tokens, hidden);
        }

        /* Apply Python model's masked_scatter layout:
         * With image_size=640, only n_local_slots (600) of the n_local_enc (864) 
         * local encoder tokens are used. Global features and sep from encoder
         * are DROPPED (they don't fit in the 857 image token positions).
         * 
         * Python's masked_scatter result:
         *   global slots (1-256)  <- source[0:256] = local_enc[0:256]
         *   sep slot (257)        <- source[256]   = local_enc[256]
         *   local slots (258-857) <- source[257:857] = local_enc[257:600]
         *   (source[857:] dropped = local_enc[600:864] + global + sep)
         */
        float *enc = encoder_output;
        int img_pos = pos;  /* starts at 1 (after BOS) */
        
        /* Global slots (256 positions) <- source[0:256] = local_enc[0:256] */
        memcpy(input_embeds + img_pos * hidden, enc, n_global_slots * hidden * sizeof(float));
        img_pos += n_global_slots;
        /* Sep slot (1 position) <- source[256] = local_enc[256] */
        memcpy(input_embeds + img_pos * hidden, enc + n_global_slots * hidden, hidden * sizeof(float));
        img_pos += 1;
        /* Local slots (n_local_slots positions) <- source[257:257+n_local_slots]
         * = local_enc[257:257+n_local_slots]
         * With n_local_slots=600, this is local_enc[257:857] (600 tokens) */
        memcpy(input_embeds + img_pos * hidden, enc + (n_global_slots + 1) * hidden,
               n_local_slots * hidden * sizeof(float));
        img_pos += n_local_slots;
        /* Note: global features (enc + n_local_enc*hidden) and sep are NOT placed,
         * matching Python's masked_scatter behavior with image_size=640.
         * local_enc[857:864] (7 tokens) are also dropped. */
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

    if (prefix_len > 1) {
        ds_decoder_prefill(ctx, input_embeds, prefix_len - 1);
        /* Last prefix token becomes first decoder_forward input */
        memcpy(dec_input, input_embeds + (prefix_len - 1) * hidden, hidden * sizeof(float));
    } else {
        /* Single token: use it directly */
        memcpy(dec_input, input_embeds, hidden * sizeof(float));
    }
    free(input_embeds);

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

        /* Get embedding for current token and decode */
        int token = ds_decoder_forward(ctx, dec_input);

        /* Debug: print top-5 logits for first token */
        if (step == 0 && ctx->dec_logits) {
            float *logits = ctx->dec_logits;
            int vocab = ctx->config.vocab_size;
            fprintf(stderr, "First token=%d, logit[100088]=%.3f\n", token, logits[100088]);
            fprintf(stderr, "First token top-5: ");
            /* Simple partial sort */
            for (int r = 0; r < 5; r++) {
                int best_i = 0; float best_v = -1e30f;
                for (int i = 0; i < vocab; i++) {
                    if (logits[i] > best_v) { best_v = logits[i]; best_i = i; }
                }
                fprintf(stderr, "[%d:%.3f] ", best_i, best_v);
                logits[best_i] = -1e30f; /* mask for next rank */
            }
            fprintf(stderr, "\n");
        }

        if (token == eos_token) {
            if (ds_verbose >= 2) fprintf(stderr, "EOS at step %d\n", step);
            break;
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

    free(dec_input);

    double total_end = now_ms();
    ctx->perf_total_ms = total_end - t0;
    ctx->perf_text_tokens = n_generated;
    ctx->perf_encode_ms = encode_end - encode_start;
    ctx->perf_decode_ms = total_end - decode_start;

    if (ds_verbose >= 1)
        fprintf(stderr, "Recognition: %d tokens generated in %.0f ms\n",
                n_generated, ctx->perf_total_ms);

    return output;
}
