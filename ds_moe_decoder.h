/*
 * ds_moe_decoder.h - DeepSeek-V2 MoE Decoder for DeepSeek-OCR
 *
 * 3B parameter MoE decoder with 64 routed experts (top-6) + 2 shared experts.
 * Architecture: 12 layers, hidden=1280, heads=10, kv_heads=10 (MHA), head_dim=128
 * Layer 0 uses dense FFN (intermediate=6848), layers 1-11 use MoE (moe_inter=896)
 */

#ifndef DS_MOE_DECODER_H
#define DS_MOE_DECODER_H

#include "ds_ocr.h"

/* Decoder prefill (multiple tokens at once) */
void ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Decoder forward (single token, uses KV cache, returns token ID) */
int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed);

/* Load decoder weights from safetensors */
int ds_decoder_load(ds_ctx_t *ctx);

#endif /* DS_MOE_DECODER_H */
