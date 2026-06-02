/*
 * ds_visual_tokenizer.h - SAM Vision Tokenizer for DeepSeek-OCR
 *
 * SAM ViT-B (with window attention + relative position embeddings)
 * → SAM neck (Conv2d+LayerNorm2d ×2) → downsample (net_2, net_3)
 * → SAM features [n_sam_tokens, 1024]
 *
 * V1: SAM features + CLIP features → concat → projector
 * V2: SAM features → DeepEncoder V2 → projector
 */

#ifndef DS_VISUAL_TOKENIZER_H
#define DS_VISUAL_TOKENIZER_H

#include "ds_ocr.h"

/* SAM forward pass: image pixels -> SAM spatial features
 * Returns: [n_sam_tokens, sam_ds2_dim] in NCHW format (caller must free)
 * n_sam_tokens = (H/16/4) * (W/16/4) = grid_h/4 * grid_w/4
 * Also sets patch_embeds for CLIP input (V1)
 */
float *ds_sam_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                       int width, int height, int channels,
                       int *out_n_tokens, float **out_patch_embeds);

/* Full visual tokenizer forward pass (calls SAM then returns features) */
float *ds_visual_tokenizer_forward(ds_ctx_t *ctx, const unsigned char *pixels,
                                    int width, int height, int channels,
                                    int *out_n_tokens, float **out_patch_embeds);

/* Load visual tokenizer weights from safetensors */
int ds_visual_tokenizer_load(ds_ctx_t *ctx);

#endif /* DS_VISUAL_TOKENIZER_H */
