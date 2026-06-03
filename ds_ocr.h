/*
 * ds_ocr.h - DeepSeek-OCR Pure C Inference Engine
 *
 * Supports DeepSeek-OCR (v1) and DeepSeek-OCR-2 (v2) models.
 * Architecture: SAM Vision Tokenizer + DeepEncoder/DeepEncoderV2 + MoE Decoder
 */

#ifndef DS_OCR_H
#define DS_OCR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

/* Vision tokenizer (SAM ViT-B) */
#define DS_IMAGE_SIZE           1024
#define DS_SAM_PATCH_SIZE       16      /* SAM patch embedding kernel/stride */
#define DS_SAM_EMBED_DIM        768     /* SAM ViT-B embedding dimension */
#define DS_SAM_HEADS            12      /* SAM ViT-B attention heads */
#define DS_SAM_HEAD_DIM         64      /* SAM ViT-B head dimension */
#define DS_SAM_MLP_DIM          3072    /* SAM ViT-B FFN intermediate dim */
#define DS_SAM_WINDOW_SIZE      14      /* SAM window attention window size */
#define DS_SAM_NECK_DIM         256     /* SAM neck output channels */
#define DS_SAM_DS1_DIM          512     /* SAM net_2 downsample channels */
#define DS_SAM_DS2_DIM          1024    /* SAM net_3 downsample channels (V1) */
#define DS_SAM_DS2_DIM_V2       896     /* SAM net_3 downsample channels (V2) */
#define DS_VISUAL_TOKENS_BASE   256     /* Base visual tokens for 1024x1024 */
#define DS_LOCAL_CROP_TOKENS    144     /* Tokens per local crop (768x768) */
#define DS_MAX_LOCAL_CROPS      6       /* Maximum local crop regions */

/* CLIP ViT-L/14 (V1 encoder) */
#define DS_CLIP_LAYERS          24
#define DS_CLIP_HIDDEN          1024
#define DS_CLIP_HEADS           16
#define DS_CLIP_HEAD_DIM        64
#define DS_CLIP_MLP_DIM         4096
#define DS_CLIP_PATCH_SIZE      14

/* DeepEncoder V2 (Qwen2-0.5B based) */
#define DS_ENC_V2_LAYERS        24
#define DS_ENC_V2_HIDDEN        896
#define DS_ENC_V2_HEADS         14
#define DS_ENC_V2_KV_HEADS      2       /* GQA: 2 KV heads for 14 Q heads */
#define DS_ENC_V2_HEAD_DIM      64
#define DS_ENC_V2_INTERMEDIATE  4864

/* Projector (V1: 2048→1280, V2: 896→1280) */
#define DS_PROJECTOR_V1_INPUT   2048    /* CLIP(1024) + SAM(1024) concatenated */
#define DS_PROJECTOR_V2_INPUT   896     /* DeepEncoder V2 output dim */

/* MoE Decoder (DeepSeek3B-MoE-A570M) */
#define DS_DEC_HIDDEN           1280
#define DS_DEC_LAYERS           12
#define DS_DEC_HEADS            10
#define DS_DEC_KV_HEADS         10      /* Standard MHA, NOT GQA (kv_heads = q_heads) */
#define DS_DEC_HEAD_DIM         128
#define DS_DEC_INTERMEDIATE     6848    /* Dense FFN intermediate (layer 0) */
#define DS_DEC_MOE_INTER        896     /* MoE expert intermediate size */
#define DS_DEC_NUM_EXPERTS      64      /* Routed experts per layer */
#define DS_DEC_SHARED_EXPERTS   2       /* Shared experts per layer */
#define DS_DEC_TOP_K            6       /* Experts activated per token */
#define DS_DEC_FIRST_K_DENSE    1       /* First K layers use dense FFN instead of MoE */
#define DS_DEC_VOCAB_SIZE       129280

/* Special token IDs (from config.json) */
#define DS_TOKEN_BOS            0       /* BOS token ID */
#define DS_TOKEN_EOS            1       /* EOS token ID */
#define DS_TOKEN_PAD            2       /* PAD token ID */
#define DS_TOKEN_IMAGE_START    151655
#define DS_TOKEN_IMAGE_END      151656
#define DS_TOKEN_NEWLINE        151657

/* Maximum layer counts (for static array sizing) */
#define DS_MAX_ENC_LAYERS       24
#define DS_MAX_DEC_LAYERS       12
#define DS_MAX_EXPERTS          64

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    int model_version;          /* 1 = DeepSeek-OCR, 2 = DeepSeek-OCR-2 */

    /* Vision tokenizer (SAM ViT-B) */
    int image_size;             /* 1024 */
    int sam_patch_size;         /* 16 */
    int sam_embed_dim;          /* 768 */
    int sam_heads;              /* 12 */
    int sam_head_dim;           /* 64 */
    int sam_mlp_dim;            /* 3072 */
    int sam_window_size;        /* 14 */
    int sam_neck_dim;           /* 256 */
    int sam_ds1_dim;            /* 512 */
    int sam_ds2_dim;            /* 1024 (V1) or 896 (V2) */
    int visual_tokens_base;     /* 256 */
    int max_local_crops;        /* 6 */
    int sam_global_attn_indexes[4]; /* [2, 5, 8, 11] */

    /* Encoder */
    int enc_type;               /* 1 = CLIP (v1), 2 = Qwen2-based (v2) */
    int enc_layers;             /* 24 */
    int enc_hidden;             /* 1024 (CLIP) or 896 (Qwen2) */
    int enc_heads;              /* 16 (CLIP) or 14 (Qwen2) */
    int enc_kv_heads;           /* 16 (CLIP) or 2 (Qwen2 GQA) */
    int enc_head_dim;           /* 64 */
    int enc_intermediate;       /* 4096 (CLIP) or 4864 (Qwen2) */
    int enc_output_dim;         /* 1280 (matches decoder hidden) */
    int enc_causal_flow_queries;/* Number of causal flow query tokens (V2 only) */
    float enc_rope_theta;       /* RoPE theta for Qwen2 encoder */

    /* Projector */
    int proj_input_dim;         /* 2048 (V1: CLIP+SAM concat) or 896 (V2) */

    /* Decoder */
    int dec_hidden;             /* 1280 */
    int dec_layers;             /* 12 */
    int dec_heads;              /* 10 */
    int dec_kv_heads;           /* 10 (standard MHA) */
    int dec_head_dim;           /* 128 */
    int dec_intermediate;       /* 6848 */
    int dec_moe_inter;          /* 896 */
    int dec_n_routed_experts;   /* 64 */
    int dec_n_shared_experts;   /* 2 */
    int dec_top_k;              /* 6 */
    int dec_first_k_dense;      /* 1 (layer 0 uses dense FFN) */
    int vocab_size;             /* 129280 */
    float dec_rms_norm_eps;     /* 1e-6 */
    float dec_rope_theta;       /* 10000.0 */
} ds_config_t;

/* ========================================================================
 * SAM Vision Tokenizer
 * ======================================================================== */

typedef struct {
    /* SAM ViT-B encoder weights */
    float *sam_patch_embed_weight;      /* [768, 3, 16, 16] */
    float *sam_patch_embed_bias;        /* [768] */
    float *sam_pos_embed;               /* [577, 768] */

    /* SAM transformer layers (12 layers) */
    struct {
        float *norm1_weight;            /* [768] LayerNorm1 (pre-attention) */
        float *norm1_bias;              /* [768] */
        float *attn_qkv_weight;         /* [2304, 768] FUSED QKV projection */
        float *attn_qkv_bias;           /* [2304] */
        float *attn_proj_weight;        /* [768, 768] Output projection */
        float *attn_proj_bias;          /* [768] */
        float *rel_pos_h;               /* [heads, head_dim, 2*window_size-1] relative pos emb */
        float *rel_pos_w;               /* [heads, head_dim, 2*window_size-1] */
        float *norm2_weight;            /* [768] LayerNorm2 (pre-FFN) */
        float *norm2_bias;              /* [768] */
        float *mlp_lin1_weight;         /* [3072, 768] */
        float *mlp_lin1_bias;           /* [3072] */
        float *mlp_lin2_weight;         /* [768, 3072] */
        float *mlp_lin2_bias;           /* [768] */
    } sam_layers[12];

    /* SAM neck: Conv2d(768→256) + LayerNorm2d + Conv2d(256→256) + LayerNorm2d */
    float *sam_neck_conv1_weight;       /* [256, 768, 1, 1] */
    float *sam_neck_conv1_bias;         /* [256] */
    float *sam_neck_ln1_weight;         /* [256] */
    float *sam_neck_ln1_bias;           /* [256] */
    float *sam_neck_conv2_weight;       /* [256, 256, 1, 1] */
    float *sam_neck_conv2_bias;         /* [256] */
    float *sam_neck_ln2_weight;         /* [256] */
    float *sam_neck_ln2_bias;           /* [256] */

    /* SAM downsample: net_2 (Conv2d 256→512, k3, s2, p1) + net_3 (Conv2d 512→1024, k3, s2, p1) */
    float *sam_net2_weight;             /* [512, 256, 3, 3] */
    float *sam_net2_bias;               /* [512] */
    float *sam_net3_weight;             /* [1024, 512, 3, 3] */
    float *sam_net3_bias;               /* [1024] */

    /* V1 specific: image_newline and view_seperator learnable tokens */
    float *image_newline;               /* [1280] */
    float *view_seperator;              /* [1280] */

    /* Causal flow query embeddings (V2 only) */
    float *causal_query_embeddings;     /* [n_queries, 896] */
} ds_visual_tokenizer_t;

/* ========================================================================
 * CLIP ViT-L/14 (V1 encoder)
 * ======================================================================== */

typedef struct {
    /* Embeddings */
    float *class_embedding;             /* [1024] CLS token embedding */
    float *patch_embedding_weight;      /* [1024, 3, 14, 14] */
    float *position_embedding;          /* [577, 1024] (1 CLS + 576 patches for 336x336, but varies) */

    /* Pre-LayerNorm */
    float *pre_layernorm_weight;        /* [1024] */
    float *pre_layernorm_bias;          /* [1024] */

    /* Transformer layers (24 layers) */
    struct {
        float *layer_norm1_weight;      /* [1024] */
        float *layer_norm1_bias;        /* [1024] */
        float *qkv_proj_weight;         /* [3072, 1024] FUSED QKV */
        float *qkv_proj_bias;           /* [3072] */
        float *out_proj_weight;         /* [1024, 1024] */
        float *out_proj_bias;           /* [1024] */
        float *layer_norm2_weight;      /* [1024] */
        float *layer_norm2_bias;        /* [1024] */
        float *mlp_fc1_weight;          /* [4096, 1024] */
        float *mlp_fc1_bias;            /* [4096] */
        float *mlp_fc2_weight;          /* [1024, 4096] */
        float *mlp_fc2_bias;            /* [1024] */
    } layers[DS_CLIP_LAYERS];

    /* Post-LayerNorm (not always present) */
    float *final_norm_weight;           /* [1024] */
    float *final_norm_bias;             /* [1024] */
} ds_clip_encoder_t;

/* ========================================================================
 * Projector (V1: 2048→1280, V2: 896→1280)
 * ======================================================================== */

typedef struct {
    float *weight;                      /* [1280, proj_input_dim] */
    float *bias;                        /* [1280] (may be NULL for no bias) */
} ds_projector_t;

/* ========================================================================
 * DeepEncoder V2 (Qwen2-0.5B based)
 * ======================================================================== */

typedef struct {
    /* Transformer layers */
    struct {
        /* Self-attention */
        float *layer_norm1_weight;      /* [896] */
        float *wq_weight;               /* [896, 896] */
        float *wk_weight;               /* [128, 896] (GQA: 2 kv_heads) */
        float *wv_weight;               /* [128, 896] (GQA: 2 kv_heads) */
        float *wo_weight;               /* [896, 896] */
        float *wq_bias;                 /* [896] */
        float *wk_bias;                 /* [128] */
        float *wv_bias;                 /* [128] */

        /* FFN (SwiGLU) */
        float *layer_norm2_weight;      /* [896] */
        float *gate_weight;             /* [4864, 896] */
        float *up_weight;               /* [4864, 896] */
        float *down_weight;             /* [896, 4864] */
    } layers[DS_MAX_ENC_LAYERS];

    /* Final norm */
    float *final_norm_weight;           /* [896] */
} ds_deep_encoder_t;

/* ========================================================================
 * MoE Decoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (NO biases in decoder) */
    uint16_t *wq_weight_bf16;          /* [n_heads*head_dim, hidden] */
    uint16_t *wk_weight_bf16;          /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wv_weight_bf16;          /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wo_weight_bf16;          /* [hidden, n_heads*head_dim] */

    /* Per-head Q/K RMSNorm */
    float *q_norm_weight;              /* [head_dim] = [128] */
    float *k_norm_weight;              /* [head_dim] = [128] */

    /* RMSNorm (no bias) */
    float *input_norm;                 /* [hidden] */
    float *post_attn_norm;             /* [hidden] */

    /* Dense FFN (used when layer_idx < first_k_dense) */
    uint16_t *dense_gate_weight_bf16;  /* [intermediate, hidden] */
    uint16_t *dense_up_weight_bf16;    /* [intermediate, hidden] */
    uint16_t *dense_down_weight_bf16;  /* [hidden, intermediate] */

    /* MoE MLP (used when layer_idx >= first_k_dense) */
    /* Router gate */
    float *gate_weight;                /* [n_experts, hidden] */

    /* Routed experts */
    struct {
        uint16_t *gate_weight_bf16;    /* [moe_inter, hidden] */
        uint16_t *up_weight_bf16;      /* [moe_inter, hidden] */
        uint16_t *down_weight_bf16;    /* [hidden, moe_inter] */
    } experts[DS_MAX_EXPERTS];

    /* Shared experts (always active) */
    uint16_t *shared_gate_weight_bf16; /* [shared_experts * moe_inter, hidden] */
    uint16_t *shared_up_weight_bf16;   /* [shared_experts * moe_inter, hidden] */
    uint16_t *shared_down_weight_bf16; /* [hidden, shared_experts * moe_inter] */

    /* Fused gate+up weight for single-token matvec */
    uint16_t *gate_up_fused_bf16;
} ds_dec_layer_t;

typedef struct {
    /* Token embeddings (tied with lm_head) */
    uint16_t *tok_embeddings_bf16;     /* [vocab_size, hidden] */

    /* Transformer layers */
    ds_dec_layer_t layers[DS_MAX_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;                       /* [hidden] */

    /* LM head (output projection) - can be tied with embeddings */
    uint16_t *lm_head_bf16;            /* [vocab_size, hidden] */
} ds_moe_decoder_t;

/* ========================================================================
 * Token Callback (streaming output)
 * ======================================================================== */

/* Called for each decoded text token during autoregressive generation.
 * 'piece' is the decoded token string (UTF-8). */
typedef void (*ds_token_cb)(const char *piece, void *userdata);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    ds_config_t config;
    ds_visual_tokenizer_t vis_tokenizer;
    ds_clip_encoder_t clip_encoder;     /* V1 only */
    ds_projector_t projector;           /* V1: 2048→1280, V2: 896→1280 */
    ds_deep_encoder_t encoder;          /* V2 only */
    ds_moe_decoder_t decoder;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* KV cache for decoder */
    float *kv_cache_k;         /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_expert_out, *dec_shared_out;
    float *dec_gate_scores;    /* [n_experts] for routing */

    /* Dense FFN buffers (for layer 0) */
    float *dec_dense_gate, *dec_dense_up, *dec_dense_swiglu;
    float *dec_dense_out;

    /* Cached RoPE tables for decoder positions */
    float *rope_cache_cos, *rope_cache_sin;
    float *rope_inv_freq;
    int rope_cache_cap;
    int rope_inv_freq_half;

    /* Encoder intermediate buffers */
    float *enc_output;         /* [max_visual_tokens, dec_hidden] */

    /* Token streaming callback (optional) */
    ds_token_cb token_cb;
    void *token_cb_userdata;

    /* Inference settings */
    int max_new_tokens;        /* Max tokens to generate (default: 4096) */
    float temperature;         /* Sampling temperature (default: 0.0 = greedy) */
    int num_local_crops;       /* Number of local crop regions (0-6) */

    /* Per-run performance stats */
    double perf_total_ms;
    int perf_text_tokens;
    double perf_encode_ms;
    double perf_decode_ms;
} ds_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
ds_ctx_t *ds_load(const char *model_dir);

/* Free all resources */
void ds_free(ds_ctx_t *ctx);

/* Set a callback to receive each decoded token as it's generated.
 * Set cb=NULL to disable. The callback is invoked during recognition. */
void ds_set_token_callback(ds_ctx_t *ctx, ds_token_cb cb, void *userdata);

/* Recognize text from an image file, returns allocated string (caller must free).
 * Supports PNG, JPEG, WebP, BMP, TIFF via stb_image. */
char *ds_recognize(ds_ctx_t *ctx, const char *image_path);

/* Recognize from raw RGB pixel data (3 channels, uint8, width*height) */
char *ds_recognize_image(ds_ctx_t *ctx, const unsigned char *pixels,
                          int width, int height, int channels);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Visual tokenizer forward pass: image pixels -> SAM features + patch_embeds */
float *ds_visual_tokenizer_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                                    int width, int height, int channels,
                                    int *out_n_tokens, float **out_patch_embeds);

/* CLIP encoder forward pass (V1): SAM patch_embeds + SAM features -> projected output */
float *ds_clip_encoder_forward(ds_ctx_t *ctx, const float *patch_embeds,
                                int n_patches, const float *sam_features,
                                int n_sam_tokens, int *out_seq_len);

/* DeepEncoder V2 forward pass: visual tokens -> encoder output */
float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len);

/* Unified encoder forward (dispatches to V1 CLIP or V2 DeepEncoder) */
float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len);

/* Decoder prefill (multiple tokens) */
void ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Decoder forward (single token, uses KV cache, returns greedy token) */
int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed);

/* Global verbose flag */
extern int ds_verbose;

#endif /* DS_OCR_H */
