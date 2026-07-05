/*
 * ds_ocr.h - DeepSeek-OCR Pure C Inference Engine
 * ds_ocr.h — DeepSeek-OCR 纯C推理引擎 公共头文件
 *
 * Supports DeepSeek-OCR (v1), DeepSeek-OCR-2 (v2), and Unlimited-OCR (v3) models.
 * 支持 DeepSeek-OCR (v1)、DeepSeek-OCR-2 (v2) 和 Unlimited-OCR (v3) 三种模型版本。
 *
 * Architecture: SAM Vision Tokenizer + DeepEncoder/DeepEncoderV2 + MoE Decoder
 * 架构: SAM视觉分词器 + 深层编码器(V1:CLIP / V2:Qwen2-0.5B) + MoE解码器
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【整体推理流水线】
 *   图像像素 → SAM ViT-B (视觉分词器) → 视觉token序列
 *                                       ↓
 *                        编码器(CLiP ViT-L/14 或 DeepEncoder V2) → 编码特征
 *                                       ↓
 *                        投影层 (2048→1280 或 896→1280) → 与解码器维度对齐
 *                                       ↓
 *                        MoE解码器 (DeepSeek3B-MoE-A570M) → 自回归生成OCR文本
 *
 * 【关键设计决策】
 *   - 纯C实现: 零外部依赖，可移植到嵌入式/边缘设备
 *   - BF16存储: 权重以BF16格式mmap加载，计算时转F32，节省50%内存
 *   - 在线softmax/fused matvec: 避免大矩阵中间结果的内存分配
 *   - 线程池: 多线程并行处理矩阵运算
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_OCR_H
#define DS_OCR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "ds_metal.h"
#include "ds_quantize.h"

/* ========================================================================
 * Constants — 常量定义
 * 每个常量对应模型架构中的一个具体维度或超参数
 * ======================================================================== */

/* Vision tokenizer (SAM ViT-B)
 * SAM ViT-B 视觉编码器 — 整个OCR流水线的"眼睛"
 * SAM (Segment Anything Model) 将输入图像切分为patch并编码为视觉token序列。
 * 采用 window attention (局部窗口注意力, window_size=14) + global attention (全局注意力, 层2/5/8/11) 的混合策略。
 */
#define DS_IMAGE_SIZE           1024    /* 输入图像尺寸 1024x1024 */
#define DS_SAM_PATCH_SIZE       16      /* SAM patch切分大小：16x16像素/patch，即卷积核和步幅 */
#define DS_SAM_EMBED_DIM        768     /* SAM ViT-B的嵌入维度(隐藏层维度)，ViT-B标准配置 */
#define DS_SAM_HEADS            12      /* SAM注意力头数 */
#define DS_SAM_HEAD_DIM         64      /* 每个头的维度 = 768/12 */
#define DS_SAM_MLP_DIM          3072    /* SAM FFN中间维度 = 768*4 */
#define DS_SAM_WINDOW_SIZE      14      /* SAM窗口注意力的窗口大小(14x14) */
#define DS_SAM_NECK_DIM         256     /* SAM neck输出通道数(768→256通道压缩) */
#define DS_SAM_DS1_DIM          512     /* SAM net_2下采样输出通道(256→512, 2×下采样) */
#define DS_SAM_DS2_DIM          1024    /* SAM net_3下采样输出通道(512→1024, V1/V3最终维度) */
#define DS_SAM_DS2_DIM_V2       896     /* SAM net_3下采样输出通道(512→896, V2最终维度) */
#define DS_VISUAL_TOKENS_BASE   256     /* 1024×1024输入的基础token数 = 64/4 × 64/4 = 16×16 */
#define DS_LOCAL_CROP_TOKENS    144     /* 768×768裁剪的token数 = 48/4 × 48/4 = 12×12 */
#define DS_MAX_LOCAL_CROPS      6       /* 最大局部裁剪数(V2: 动态多裁剪上限) */

/* CLIP ViT-L/14 (V1/V3 encoder)
 * CLIP ViT-L/14 — V1/V3的第二编码器(24层标准ViT)
 * V1/V3中CLIP绕过自身patch_embed，直接接收SAM特征做"二次精炼"
 */
#define DS_CLIP_LAYERS          24      /* CLIP Transformer层数 */
#define DS_CLIP_HIDDEN          1024    /* CLIP隐藏维度 */
#define DS_CLIP_HEADS           16      /* CLIP注意力头数 */
#define DS_CLIP_HEAD_DIM        64      /* 每个头的维度 = 1024/16 */
#define DS_CLIP_MLP_DIM         4096    /* CLIP FFN中间维度 = 1024*4 */
#define DS_CLIP_PATCH_SIZE      14      /* CLIP原始patch大小(V1/V3中不使用，SAM已完成patch化) */

/* DeepEncoder V2 (Qwen2-0.5B based)
 * DeepEncoder V2 — V2的编码器(Qwen2-0.5B架构，混合注意力)
 * 视觉token双向注意力 + 因果流查询因果注意力
 */
#define DS_ENC_V2_LAYERS        24      /* DeepEncoder V2 Transformer层数 */
#define DS_ENC_V2_HIDDEN        896     /* DeepEncoder V2隐藏维度 */
#define DS_ENC_V2_HEADS         14      /* DeepEncoder V2 Q头数 */
#define DS_ENC_V2_KV_HEADS      2       /* GQA: 2个KV头(14个Q头共享2组KV，7:1分组比) */
#define DS_ENC_V2_HEAD_DIM      64      /* 每个头的维度 = 896/14 */
#define DS_ENC_V2_INTERMEDIATE  4864    /* DeepEncoder V2 SwiGLU FFN中间维度 */

/* Projector (投影层: 编码器维度→解码器维度)
 * V1/V3: 2048→1280 (CLIP 1024 + SAM 1024 拼接后投影)
 * V2: 896→1280 (DeepEncoder V2输出直接投影)
 */
#define DS_PROJECTOR_V1_INPUT   2048    /* V1/V3投影输入: CLIP(1024) + SAM(1024)拼接 */
#define DS_PROJECTOR_V2_INPUT   896     /* V2投影输入: DeepEncoder V2输出维度 */

/* MoE Decoder (DeepSeek3B-MoE-A570M)
 * MoE解码器 — 整个推理流程的"嘴巴和大脑"
 * 总参数~3B，但每次只激活~570M(3B知识，570M计算)，这就是MoE的威力
 */
#define DS_DEC_HIDDEN           1280    /* 解码器隐藏维度 */
#define DS_DEC_LAYERS           12      /* 解码器Transformer层数 */
#define DS_DEC_HEADS            10      /* Q注意力头数 */
#define DS_DEC_KV_HEADS         10      /* KV头数(标准MHA，不是GQA；10=10) */
#define DS_DEC_HEAD_DIM         128     /* 每个头的维度 = 1280/10 */
#define DS_DEC_INTERMEDIATE     6848    /* Dense FFN中间维度(仅Layer 0) */
#define DS_DEC_MOE_INTER        896     /* MoE每个专家的中间维度 */
#define DS_DEC_NUM_EXPERTS      64      /* 每层路由专家数(64个专家中每次只激活top-6) */
#define DS_DEC_SHARED_EXPERTS   2       /* 每层共享专家数(始终激活，提供通用知识) */
#define DS_DEC_TOP_K            6       /* 每个token激活的top-K路由专家数 */
#define DS_DEC_FIRST_K_DENSE    1       /* 前K层使用Dense FFN(浅层语义不足，路由效果差) */
#define DS_DEC_VOCAB_SIZE       129280  /* 词表大小(129280个token) */

/* Special token IDs (from config.json) — 特殊token ID */
#define DS_TOKEN_BOS            0       /* BOS token ID — 序列开始标记 */
#define DS_TOKEN_EOS            1       /* EOS token ID — 序列结束标记(解码器遇到此token停止) */
#define DS_TOKEN_PAD            2       /* PAD token ID — 填充标记 */
#define DS_TOKEN_IMAGE_START    151655  /* V1/V2图像开始标记 */
#define DS_TOKEN_IMAGE_END      151656  /* V1/V2图像结束标记 */
#define DS_TOKEN_NEWLINE        151657  /* V1/V3图像行分隔符(每16个token后插入) */
#define DS_TOKEN_IMAGE_PLACEHOLDER 128815  /* Unlimited-OCR: 单个图像占位符token ID */

/* Model version identifiers — 模型版本标识 */
#define DS_MODEL_VERSION_V1     1       /* DeepSeek-OCR (原始版, SAM+CLIP双编码器) */
#define DS_MODEL_VERSION_V2     2       /* DeepSeek-OCR-2 (DeepEncoder V2, 因果流查询) */
#define DS_MODEL_VERSION_UNLIMITED 3    /* Unlimited-OCR (SAM+CLIP+2D网格+R-SWA滑动窗口) */

/* Maximum layer counts (for static array sizing) — 静态数组尺寸上限 */
#define DS_MAX_ENC_LAYERS       24
#define DS_MAX_DEC_LAYERS       12
#define DS_MAX_EXPERTS          64

/* ========================================================================
 * Model Configuration — 模型配置结构体
 * 记录三版本(V1/V2/V3)的所有架构超参数
 * ======================================================================== */

typedef struct {
    int model_version;          /* 1 = DeepSeek-OCR, 2 = DeepSeek-OCR-2, 3 = Unlimited-OCR */
    int sliding_window_size;    /* R-SWA window (0=disabled, 128=Unlimited-OCR) */

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
    int has_qk_norm;            /* 1 = per-head Q/K RMSNorm (V1/V2), 0 = none (Unlimited-OCR) */

    /* Unlimited-OCR specific */
    int image_token_id;         /* 128815 for Unlimited-OCR, unused for V1/V2 */
    int image_size_crop;        /* 640 (crop image size for Unlimited-OCR) */
} ds_config_t;

/* ========================================================================
 * SAM Vision Tokenizer — SAM视觉分词器权重
 * 所有SAM权重以F32格式加载(预转换，适合批量计算)
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
    float *causal_query_embeddings;     /* [256, 896] for 1024x1024 input */
    float *causal_query_768_embeddings; /* [144, 896] for 768x768 input */

    /* Model directory for loading precomputed interpolation files */
    char model_dir[512];
} ds_visual_tokenizer_t;

/* ========================================================================
 * CLIP ViT-L/14 (V1/V3 encoder) — CLIP编码器权重(F32格式)
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
 * Projector (V1: 2048→1280, V2: 896→1280) — 投影层权重(F32)
 * 将编码器输出维度投影到解码器隐藏维度
 * ======================================================================== */

typedef struct {
    float *weight;                      /* [1280, proj_input_dim] */
    float *bias;                        /* [1280] (may be NULL for no bias) */
} ds_projector_t;

/* ========================================================================
 * DeepEncoder V2 (Qwen2-0.5B based) — V2编码器权重(F32格式)
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
    uint16_t *gate_weight_bf16;        /* [n_experts, hidden] BF16 version for precision matching */

    /* Routed experts */
    struct {
        uint16_t *gate_weight_bf16;    /* [moe_inter, hidden] */
        uint16_t *up_weight_bf16;      /* [moe_inter, hidden] */
        uint16_t *down_weight_bf16;    /* [hidden, moe_inter] */
        uint16_t *gate_up_fused_bf16;  /* [2*moe_inter, hidden] gate+up concatenated (decode) */
    } experts[DS_MAX_EXPERTS];

    /* Contiguous expert block: all routed experts' gate_up_fused + shared gate_up_fused
     * in one allocation for better page-in locality during decode.
     * Layout: expert 0 gate_up_fused [2*moe_inter, hidden]
     *         expert 1 gate_up_fused [2*moe_inter, hidden]
     *         ...
     *         expert N-1 gate_up_fused [2*moe_inter, hidden]
     *         shared gate_up_fused [2*n_shared*moe_inter, hidden]
     * experts[e].gate_up_fused_bf16 and shared_gate_up_fused_bf16 point into this block. */
    uint16_t *expert_block_bf16;
    size_t    expert_block_size;        /* bytes */

    /* Shared experts (always active) */
    uint16_t *shared_gate_weight_bf16; /* [shared_experts * moe_inter, hidden] */
    uint16_t *shared_up_weight_bf16;   /* [shared_experts * moe_inter, hidden] */
    uint16_t *shared_down_weight_bf16; /* [hidden, shared_experts * moe_inter] */
    uint16_t *shared_gate_up_fused_bf16; /* [2*shared_experts*moe_inter, hidden] gate+up concatenated (decode) */

    /* INT4 quantized expert weights (optional, enabled by --int4) */
    struct {
        ds_int4_block_t gate_up_fused;  /* [2*moe_inter, hidden] INT4 */
        ds_int4_block_t down_weight;    /* [hidden, moe_inter] INT4 */
    } experts_int4[DS_MAX_EXPERTS];
    ds_int4_block_t shared_gate_up_int4;   /* [2*n_shared*moe_inter, hidden] INT4 */
    ds_int4_block_t shared_down_int4;      /* [hidden, n_shared*moe_inter] INT4 */
    ds_int4_block_t dense_gate_up_int4;    /* [2*intermediate, hidden] INT4 (layer 0 dense) */
    ds_int4_block_t dense_down_int4;       /* [hidden, intermediate] INT4 (layer 0 dense) */
    int int4_enabled;                       /* 1 if INT4 quantization was applied */
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

    /* KV cache for decoder — stored as F32 with cache-line alignment for direct
     * attention access. Previous design used BF16 storage + per-step batch
     * conversion, but that reconverted the ENTIRE cache every decode step
     * (O(seq_len) BF16→F32 work per layer), which dominated attention time.
     * F32 storage doubles memory but eliminates all per-step conversion. */
    float *kv_cache_k;       /* [layers, max_seq, kv_dim] aligned F32 */
    float *kv_cache_v;       /* [layers, max_seq, kv_dim] aligned F32 */
    int kv_cache_len;
    int kv_cache_max;
    int _kv_row_stride;      /* kv_dim rounded up to 16-float alignment for cache line access */

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_expert_out, *dec_shared_out;
    float *dec_gate_scores;    /* [n_experts] for routing */
    float *dec_layer_out;      /* [hidden] temp output for decoder layer forward */

    /* Dense FFN buffers (for layer 0) */
    float *dec_dense_gate, *dec_dense_up, *dec_dense_swiglu;
    float *dec_dense_out;

    /* Pre-allocated MoE scratch buffers (avoid per-token malloc) */
    float *moe_expert_gate_buf, *moe_expert_up_buf;
    float *moe_expert_gate_up_buf, *moe_expert_hidden_buf;
    float *moe_shared_gate_buf, *moe_shared_up_buf;
    float *moe_shared_gate_up_buf, *moe_shared_swiglu_buf;
    float *moe_shared_out_buf;
    float *moe_expert_outputs; /* top_k * hidden */

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
    float repeat_penalty;      /* Repetition penalty (default: 1.0 = none, 1.1-1.5 typical) */
    int num_local_crops;       /* Number of local crop regions (0-6) */

    /* Repetition penalty support */
    float *dec_logits;                 /* [vocab_size] logits buffer */
    int *token_history;                /* generated token IDs for penalty */
    int token_history_len;
    int token_history_cap;
    int no_repeat_ngram_size;          /* n-gram blocking (0=disabled, default=0) */
    int min_new_tokens;               /* Min tokens before allowing EOS (0=disabled, default=256) */
    int prefill_token_count;          /* Token count after prefill (for R-SWA: visual+prompt tokens) */

    /* Per-run performance stats */
    double perf_total_ms;
    int perf_text_tokens;
    double perf_encode_ms;        /* image → encoder output (SAM + DeepEncoder) */
    double perf_decode_ms;        /* autoregressive decode loop */
    double perf_prefill_ms;       /* prefill (KV cache fill from encoder tokens) */
    double perf_sam_ms;           /* SAM vision tokenizer only (for multi-crop, last crop's time) */
    double perf_encoder_ms;       /* DeepEncoder/CLIP only */

    /* Per-layer profiler stats (enabled by --profile) */
    int profile_enabled;                       /* 0=off, 1=on */
    int int4_enabled;                          /* 0=off, 1=INT8 quant for MoE experts (flag name is --int4 for compatibility) */
    double perf_layer_qkv_ms[DS_MAX_DEC_LAYERS];      /* QKV projection time per layer */
    double perf_layer_attn_ms[DS_MAX_DEC_LAYERS];     /* Attention time per layer */
    double perf_layer_proj_ms[DS_MAX_DEC_LAYERS];     /* Output projection time per layer */
    double perf_layer_mlp_ms[DS_MAX_DEC_LAYERS];      /* MLP/MoE time per layer */
    double perf_layer_total_ms[DS_MAX_DEC_LAYERS];    /* Total time per layer */
    double perf_lm_head_ms;                           /* LM head projection time */
    double perf_sampling_ms;                           /* Sampling/repetition penalty time */
    int perf_decode_steps;                            /* Number of decode steps executed */

    /* Pre-converted F32 weight caches for sgemm (BF16→F32, allocated on first use) */
    float *lm_head_f32;             /* [vocab_size, hidden] F32 version of lm_head_bf16 */
    int lm_head_f32_ready;          /* 1 = converted and ready for sgemm */
    float *tok_emb_f32;             /* [vocab_size, hidden] F32 version of tok_embeddings_bf16 (if tied) */
    int tok_emb_f32_ready;

    /* V3 streaming det tag filter state */
    char _det_buf[256];             /* Buffer for accumulating potential det/ref tags */
    int _det_buf_len;               /* Current length of buffered text */

    /* Metal GPU acceleration context (NULL if Metal unavailable) */
    struct ds_metal_ctx *metal_ctx;
    int metal_enabled;              /* 1 = use Metal for MoE/LM head, 0 = CPU only */
} ds_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory */
ds_ctx_t *ds_load(const char *model_dir);

/* INT4 quantize MoE expert weights (call after ds_load, when --int4 is set).
 * Reduces memory bandwidth ~4x for decode, with minimal accuracy loss. */
int ds_quantize_moe_int4(ds_ctx_t *ctx);

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
                                    int *out_n_tokens, float **out_patch_embeds,
                                    unsigned char **out_resized_pixels,
                                    int *out_resized_w, int *out_resized_h);

/* CLIP encoder forward pass (V1/V3):
 * V1: raw RGB pixels -> Conv2d(3→1024) -> CLIP transformer
 * V3: SAM features directly as CLIP patch_embeds */
float *ds_clip_encoder_forward(ds_ctx_t *ctx,
                                const unsigned char *rgb_pixels, int width, int height, int channels,
                                const float *sam_features, int n_sam_tokens,
                                int *out_seq_len);

/* DeepEncoder V2 forward pass: visual tokens -> encoder output */
float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len,
                               int n_causal_queries, const float *causal_queries);

/* Unified encoder forward (dispatches to V1 CLIP or V2 DeepEncoder) */
float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len,
                           int n_causal_queries, const float *causal_queries);

/* Decoder prefill (multiple tokens) */
void ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Decoder forward (single token, uses KV cache, returns greedy token) */
int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed);

/* Decoder forward for N tokens at once (batch decode with causal masking).
 * Each token attends to all prior tokens in the cache + prior tokens in the batch.
 * Returns token IDs for all N tokens. The caller provides N input embeddings.
 * tokens_out must be pre-allocated with space for n_tokens ints.
 * This amortizes KV cache reads across all tokens in the batch. */
void ds_decoder_forward_batch(ds_ctx_t *ctx, const float *input_embeds,
                                int n_tokens, int *tokens_out);

/* Global verbose flag */
extern int ds_verbose;

/* V3 (Unlimited-OCR) post-processing: strip detection coordinate tags.
 * Removes <|det|>label [bbox]<|/det|> and <|ref|>...<|/ref|><|det|>...<|/det|> blocks.
 * Modifies text in-place, returns new length. */
int ds_strip_det_tags(char *text, int len);

#endif /* DS_OCR_H */
