/*
 * ds_moe_decoder.h - DeepSeek-V2 MoE Decoder for DeepSeek-OCR
 * ds_moe_decoder.h — DeepSeek-V2 MoE 解码器
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】MoE 解码器是 ds-ocr 的"嘴巴和大脑"
 * ─────────────────────────────────────────────────────────────────────
 * 接收编码器输出的视觉token序列，通过自回归解码逐token生成OCR文本。
 * 这是整个推理流程中计算量最大、耗时最长的阶段（~5秒/图，占总耗时1/3）。
 *
 * 【DeepSeek3B-MoE-A570M 架构】
 *   总参数: ~3B (30亿)
 *   激活参数: ~570M (每次推理只激活1/5的参数，这就是MoE的威力)
 *   隐藏维度: 1280
 *   层数: 12
 *   注意力: MHA (10头×128维 = 1280)
 *   FFN:
 *     Layer 0: Dense SwiGLU (intermediate=6848) — 通用特征提取
 *     Layer 1~11: MoE (64路由专家 top-6 + 2共享专家, moe_inter=896) — 专业化分工
 *   词表: 129280
 *   位置编码: RoPE (theta=10000)
 *   归一化: RMSNorm (eps=1e-6)
 *
 * 【MoE 工作原理】(详见 02-MoE混合专家篇.md)
 *   1. 路由器(Gate): x @ W_gate → softmax → top-6 → 选择6个专家
 *   2. 路由专家: 6个独立的 SwiGLU FFN 各自处理 x，按路由权重加权求和
 *   3. 共享专家: 2个永远激活的 SwiGLU FFN，处理通用知识
 *   4. 最终输出 = 路由专家加权组合 + 共享专家输出
 *
 * 【解码流程】(详见 04-完整推理流程篇.md)
 *   Prefill: 批量处理所有视觉+prompt token → 填充KV缓存
 *   Decode:  逐token生成 → 每步: 嵌入查表 → 12层前向 → argmax → 下一个token
 *
 * 【KV缓存】(详见 03-推理引擎实战篇.md)
 *   存储已计算的 Key 和 Value 向量，避免每步重复计算
 *   V3 使用 R-SWA (Reference Sliding Window Attention)：
 *   KV缓存 = 视觉token(始终保留) + 最近128个文本token(滑动窗口)
 *   这样KV缓存大小恒定，不会随生成长度线性增长
 *
 * 【与 qwen-asr 的对应】
 * qwen-asr 使用 Dense 解码器 (Qwen3, 16头 GQA, Dense SwiGLU FFN)
 * ds-ocr 使用 MoE 解码器 (DeepSeek-V2, 10头 MHA, 64+2 MoE FFN)
 * 这是两个项目最大的架构差异——MoE vs Dense
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_MOE_DECODER_H
#define DS_MOE_DECODER_H

#include "ds_ocr.h"

/* ds_decoder_prefill — 解码器预填充（多token并行处理）
 *
 * 输入: input_embeds [seq_len, hidden] — 视觉token + prompt的嵌入序列
 *       seq_len 通常 = 1(BOS) + 257~273(图像token) + 4(prompt文本) ≈ 280~330
 *
 * 处理流程:
 *   对每一层 (l = 0..11):
 *     1. 批量 RMSNorm: [seq_len, hidden] → x_norm
 *     2. 批量 QKV 投影: [seq_len, hidden] → [seq_len, 3*kv_dim] (用sgemm)
 *     3. 批量注意力: 因果掩码 + 在线softmax → [seq_len, hidden]
 *     4. 写入 KV 缓存: K/V 存入 kv_cache_k/v 的对应位置
 *     5. MoE/FFN:
 *        Layer 0 (Dense): 批量 sgemm 计算 gate+up+swiglu+down
 *        Layer 1~11 (MoE):
 *          a. 批量路由: [seq_len, 64] gate scores (一次sgemm)
 *          b. 逐token top-K 选择
 *          c. 按expert分组: 收集选中同一expert的token → 批量sgemm
 *          d. Scatter: 加权写回对应token位置
 *          e. 共享专家: 一次sgemm处理所有token
 *
 * 【Prefill vs Decode 的关键区别】
 * Prefill: 多token并行，用 sgemm 批量矩阵乘法，计算密集
 * Decode:  单token串行，用 matvec 矩阵×向量，内存带宽密集
 *
 * 【为什么Prefill这么快?】(详见 05-性能优化实战篇.md)
 * 共享专家: 3次sgemm代替860次matvec → 50-100×加速
 * 路由专家: 按expert分组的 Gather-Compute-Scatter 模式 → 30×加速
 * v0.5→v0.9: Prefill从30s降到0.93s
 */
void ds_decoder_prefill(ds_ctx_t *ctx, const float *input_embeds, int seq_len);

/* ds_decoder_forward — 解码器前向传播（单token生成）
 *
 * 输入: input_embed [hidden] — 当前token的嵌入向量（来自tok_embeddings查表）
 * 输出: token ID — 预测的下一个token（贪心解码：argmax of logits）
 *
 * 处理流程 (12层 × 以下步骤):
 *   ┌──────────────────────────────────────────────────────┐
 *   │ 1. RMSNorm: x_norm = rms_norm(x, input_norm)        │  ← Pre-Norm模式
 *   │ 2. QKV投影: q,k,v = Wq*x_norm, Wk*x_norm, Wv*x_norm│  ← BF16→F32 on-the-fly
 *   │ 3. Per-head Q/K RMSNorm (V1/V2):                    │  ← DeepSeek特色: 防止QK点积爆炸
 *   │    q = rms_norm_per_head(q, q_norm)                  │
 *   │    k = rms_norm_per_head(k, k_norm)                  │
 *   │ 4. RoPE: q = apply_rope(q, cos, sin)                │  ← 旋转位置编码
 *   │         k = apply_rope(k, cos, sin)                  │
 *   │ 5. KV缓存写入: cache_k[pos] = k, cache_v[pos] = v  │  ← 存起来，后续步骤复用
 *   │ 6. 因果注意力:                                      │
 *   │    V1/V2: causal_attention(q, cache_k, cache_v)      │  ← 标准因果掩码
 *   │    V3:    rswa_attention(q, ref_k, win_k, ...)       │  ← R-SWA两段注意力
 *   │ 7. 输出投影: proj_out = Wo @ attn_out               │
 *   │ 8. 残差连接: x = x + proj_out                        │  ← 梯度捷径
 *   │ 9. RMSNorm: x_norm = rms_norm(x, post_attn_norm)    │
 *   │ 10. FFN/MoE:                                        │
 *   │    Layer 0: Dense SwiGLU (gate+up+swiglu+down)      │
 *   │    Layer 1~11: MoE (路由+6专家+2共享)               │
 *   │ 11. 残差连接: x = x + ffn_out                       │
 *   └──────────────────────────────────────────────────────┘
 *   最终: x = rms_norm(x, final_norm) → argmax(lm_head @ x) → token_id
 *
 * 【为什么 decode 是瓶颈?】
 * 每步需要读取 ~943MB BF16权重（12层×78.6MB + LM head 331MB）
 * 但只做 ~2.4M FLOPs 计算（单token matvec）
 * → 内存带宽瓶颈: M2 Pro ~200GB/s → 理论极限 ~4.7ms/step
 * → 实测 ~23ms/step（效率 ~20%，受BF16转换+缓存未命中影响）
 */
int ds_decoder_forward(ds_ctx_t *ctx, const float *input_embed);

/* ds_decoder_load — 加载解码器权重
 *
 * 所有权重以 BF16 格式存储（mmap零拷贝），仅 RMSNorm 和 Q/K norm 以 F32 存储
 * 关键权重:
 *   tok_embeddings_bf16: [129280, 1280] — 词嵌入矩阵（与lm_head共享=tied embeddings）
 *   各层 Wq/Wk/Wv/Wo: BF16 — 注意力投影
 *   各层 input_norm/post_attn_norm: F32 — RMSNorm权重
 *   各层 q_norm/k_norm: F32 — Per-head Q/K RMSNorm (V1/V2)
 *   Layer 0 dense gate/up/down: BF16 — 稠密FFN
 *   Layer 1~11 gate_weight: BF16+F32 — MoE路由器（BF16保精度，F32备用）
 *   Layer 1~11 experts[0..63]: BF16 — 64个路由专家的gate/up/down权重
 *   Layer 1~11 shared_gate/up/down: BF16 — 2个共享专家的gate/up/down权重
 *   lm_head_bf16: [129280, 1280] — 输出投影（可能=tok_embeddings共享）
 *
 * 额外构建: gate_up_fused 权重（gate+up拼接为一行，单次matvec代替两次）
 *           连续 expert_block（所有expert+shared在同一块内存，减少page fault）
 */
int ds_decoder_load(ds_ctx_t *ctx);

#endif /* DS_MOE_DECODER_H */
