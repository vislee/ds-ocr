/*
 * ds_deep_encoder.h - Encoders for DeepSeek-OCR
 *
 * V1: CLIP ViT-L/14 encoder (takes SAM patch_embeds as input)
 * V2: DeepEncoder V2 (Qwen2-0.5B based) with causal flow queries
 */

#ifndef DS_DEEP_ENCODER_H
#define DS_DEEP_ENCODER_H

#include "ds_ocr.h"

/* V1 CLIP encoder forward pass
 * Takes SAM patch_embeds as input (NOT raw image!)
 * Input: patch_embeds [n_patches, clip_hidden] (from SAM's patch embedding layer)
 * Also takes SAM spatial features [n_sam_tokens, 1024] for concatenation
 * Output: projected tokens [n_output_tokens, dec_hidden] (float32, caller must free)
 * out_seq_len: total sequence length including special tokens
 */
float *ds_clip_encoder_forward(ds_ctx_t *ctx, const float *patch_embeds,
                                int n_patches, const float *sam_features,
                                int n_sam_tokens, int *out_seq_len);

/* V2 DeepEncoder forward pass
 * Input: visual tokens [n_tokens, enc_hidden]
 * Output: encoder output [n_output_tokens, dec_hidden] (caller must free)
 */
float *ds_encoder_forward_v2(ds_ctx_t *ctx, const float *visual_tokens,
                               int n_tokens, int *out_seq_len,
                               int n_causal_queries, const float *causal_queries);

/* Unified encoder forward (dispatches to V1 CLIP or V2 DeepEncoder) */
float *ds_encoder_forward(ds_ctx_t *ctx, const float *visual_tokens,
                           int n_tokens, int *out_seq_len,
                           int n_causal_queries, const float *causal_queries);

/* Load encoder weights from safetensors */
int ds_encoder_load(ds_ctx_t *ctx);

#endif /* DS_DEEP_ENCODER_H */
