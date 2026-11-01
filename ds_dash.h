/*
 * ds_dash.h — V4 (Unlimited-OCR-2) DASH 块稀疏注意力
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】V4 解码器专用注意力路径
 * ─────────────────────────────────────────────────────────────────────
 * V4 训练/部署 (baidu/Unlimited-OCR-2) 使用 DASH (Document-Aware Sparse
 * Attention with Head-CLS routing) 块稀疏注意力 + page2d 二维 RoPE，
 * 官方推理时显式关闭 R-SWA (cfg._ring_window=None)，与 V3 数学完全不同。
 *
 * 【与 V3 (R-SWA) 的本质区别】
 *   V3: 解码 query 只看 [参考集(视觉+prompt) + 最近W=128 token]，1D RoPE。
 *   V4: 每 query 由路由器从全部 KV 块中挑 K=12 个"远程块"（加先验偏置），
 *       加上 token 级局部窗 128（无偏置），1D 换成 (tok,page) 二维 RoPE。
 *
 * 【DASH 推理数学】(对照官方 modeling_deepseekv2.py 的 _INFER 路径:
 *   _infer_prefill_attend / _block_scores_infer / _infer_route_select /
 *   _dash_decode_forward / _decode_attend + modeling_unlimitedocr.py 的
 *   _uocr_dash_decode_allow)
 *   配置: block_size=32, local_blocks=4(→局部窗128), K=dash_topk_blocks=12,
 *         topm=16, index_dim=128, idx_scale=128/√128=11.3137, sigma=1.0,
 *         floor=1e-4
 *
 *   1. 索引投影:  idx_q = page2drope(GemmaRMS(q_proj(层输入)))  [S,128]
 *                 idx_k = page2drope(GemmaRMS(k_proj(层输入)))  [S,128]
 *      GemmaRMS: scale = (1 + w) * x / rms(x)，非标准 RMSNorm！
 *      输入是残差流层输入（非 x_norm）；idx_k 按层缓存全量历史
 *      （官方 past_kv._dash_idx_k[layer] 追加语义）。
 *   2. 块分数 (head_cls stage0): k_bar[b] = 块内 softmax(idx_k·hc/√128)
 *      加权平均（与 query 无关，每块算一次）；z[q,b] = clamp(idx_q·k_bar/√128)。
 *   3. remote 掩码: b ≤ q_pos/32 − local_blocks 才算远程，否则 z=−1e9。
 *   4. 路由选择 (官方 _infer_route_select):
 *      cand = remote ∧ 页候选（同页 image 块；无候选→放开全部 remote）；
 *      x = cand ? z·idx_scale : −1e4（哨兵）→ top-M=16 窗口 →
 *      entmax15（α=1.5，τ* 按官方「计数取值」法求，窗口内归一）→
 *      前 K=min(12,M) 槽位；valid = 槽是有效候选（哨兵槽 invalid，
 *      不进 μ 也不进注意力）；bias = (log(clamp(prior,1e-4)) − μ)/σ，
 *      μ 只对 valid 求均值。
 *   5. 注意力: m[q,k] = 0 (k∈(q−128,q] 局部) / bias[b(k)] (k∈选中块) /
 *      −inf (其余)；softmax(QK^T·scale + m) @ V
 *
 * 【page2d 二维 RoPE】(官方 DeepseekV2IndexerRotary2D, page_pairs=1)
 *   维度 128 = 64 频率对: 前 63 对 → tok 轴 (theta=10000, resample:
 *   inv_tok[i]=10000^(-i/63))，第 64 对 → page 轴: inv_page = 256^0 = 1，
 *   角度 = (page%2)·1.0 rad（不是除以 256！page_pairs=1 时指数为 0）。
 *   page 轴折叠 page%2 只保留"同页/跨页"信息；tok 轴是页内 running count，
 *   使页内相对距离与文档总长无关（官方 _build_page_rope_pos 文档）。
 *
 * 【decode 状态机】
 *   prefill 记录 pf_cnt[p] (每页 prefill(≤last_img) token 数)；decode 每步:
 *   新 token == <PAGE> → page=min(page+1, n_pg-1), tok=pf_cnt[新page]；
 *   否则 → tok+=1。生成的 <PAGE> 标记使响应文本与图像页共享页坐标。
 *   decode 候选页 qp = mcum − 1 − pp（响应页 k ↔ 图像页 k），<0 → 放开。
 * ========================================================================
 */

#ifndef DS_DASH_H
#define DS_DASH_H

#include "ds_ocr.h"

/* DASH 超参数（models/Unlimited-OCR-2 config.json → sparse_attention_config）
 * 推理路径参数来自 configure_dash_inference 的 _infer_params 快照:
 *   idx_scale = dash_scale_per_layer/√index_dim = 128/√128,
 *   K = dash_topk_blocks = 12 (读预算槽位数), topm = 16 (entmax 窗口),
 *   floor = dash_prior_floor = 1e-4, sigma = dash_sigma = 1.0 */
#define DS_DASH_BLOCK_SIZE      32       /* sparse_block_size: KV 分块大小 */
#define DS_DASH_LOCAL_BLOCKS    4        /* sparse_local_block: 局部块数 → 局部窗128 */
#define DS_DASH_TOPK_BLOCKS     12       /* dash_topk_blocks: 读预算 K（选中块槽位数） */
#define DS_DASH_TOPM            16       /* DASH_INFER_TOPM: entmax 前的 top-M 窗口 */
#define DS_DASH_INDEX_DIM       128      /* sparse_index_dim: 索引头维度 */
#define DS_DASH_IDX_SCALE       11.3137085f  /* dash_scale_per_layer 128 / sqrt(128) */
#define DS_DASH_SIGMA           1.0f     /* dash_sigma: bias 温度 */
#define DS_DASH_FLOOR           1e-4f    /* dash_prior_floor: log 先验下限 */
#define DS_DASH_TOK_THETA       10000.0f /* sparse_rope_tok_theta: tok 轴基频 */
#define DS_DASH_PAGE_THETA      256.0f   /* sparse_rope_page_theta: page 轴基频率（page_pairs=1
                                             时 inv_page = 256^0 = 1，角度 = page%2 弧度） */
#define DS_DASH_PAGE_MOD        2        /* sparse_rope_page_mod: page 坐标取模 */
#define DS_DASH_MAX_NBLK        256      /* 8192/32: 块数上限 */
#define DS_DASH_MAX_PAGES       128      /* 页桶数上限（decode 状态机 pf_cnt 表） */

/* 每 token id（来自 sparse_attention_config）*/
#define DS_TOKEN_PAGE_MARKER    128827   /* <PAGE>: 多页分隔 + 输出分页标记 */
#define DS_TOKEN_IMAGE_V4       128815   /* <image>: 视觉占位（与 V3 相同） */

/* 每 query 路由结果 */
typedef struct {
    int   n_valid;                       /* 槽位数 K=min(12, M)（M=min(16,nblk)） */
    int   blk[DS_DASH_TOPK_BLOCKS];      /* 选中的 KV 块号（可能含哨兵槽 blk=-1） */
    float bias[DS_DASH_TOPK_BLOCKS];     /* log-prior 偏置（invalid 槽为 0） */
    int   valid[DS_DASH_TOPK_BLOCKS];    /* 该槽是否有效候选（哨兵槽=0，不进注意力） */
} ds_dash_route_t;

/* ── KV cache 行访问（与 ds_moe_decoder.c 共享布局）───────────────────────
 * 布局: cache[layers][max_seq][kv_row_stride], kv_row_stride ≥ kv_dim, 64B 对齐 */
static inline void *ds_kv_k_row(ds_ctx_t *ctx, int layer, int pos) {
    int stride = ctx->_kv_row_stride;
    size_t elem = ctx->kv_bf16 ? sizeof(uint16_t) : sizeof(float);
    return (char *)ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * stride * elem + (size_t)pos * stride * elem;
}
static inline void *ds_kv_v_row(ds_ctx_t *ctx, int layer, int pos) {
    int stride = ctx->_kv_row_stride;
    size_t elem = ctx->kv_bf16 ? sizeof(uint16_t) : sizeof(float);
    return (char *)ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * stride * elem + (size_t)pos * stride * elem;
}

/* ── 生命周期 ─────────────────────────────────────────────────────────── */
int  ds_dash_init(ds_ctx_t *ctx);                        /* 分配运行时缓冲 */
void ds_dash_free(ds_ctx_t *ctx);
/* 跨识别重置: 清 idx_k_hist 历史与 decode 状态机。
 * C 引擎每页独立 decode（kv_cache_len 归零重 prefill），而块分数 k_bar
 * 按全缓存长度读 idx_k_hist——不重置则上一页残留混入路由，随识别次数
 * 累积失真（实测第14页起循环早停）。官方多页单次 decode 无此问题。 */
void ds_dash_reset_run(ds_ctx_t *ctx);

/* ── page2d 坐标构建 ────────────────────────────────────────────────────
 * 由 prefill 的 token id 序列构建 (tok_pos, page_pos) 坐标并预计算 cos/sin。
 * token_ids[0..seq_len) = [BOS, <PAGE>, img..., 指令...]。
 * 同时记录 pf_cnt (每页 prefill 长度) 供 decode 状态机使用。返回 0 成功。*/
int  ds_dash_build_coords(ds_ctx_t *ctx, const int *token_ids, int seq_len);

/* decode 状态机推进: 新 token 写入坐标表 pos=kv_cache_len 处。
 * <PAGE> → 换页(tok 重置为该页 prefill 长度)，否则 tok+1。*/
void ds_dash_advance_coords(ds_ctx_t *ctx, int new_token_id);

/* ── DASH 注意力 ────────────────────────────────────────────────────────
 * 两者都直接读写 F32 KV cache（与 V3 路径共享缓存布局），内部完成:
 * 索引投影 → 块分数 → entmax 路由 → 带偏置的块稀疏 softmax 注意力。
 *
 * layer_input: 残差流 x（层输入）——官方 _router_qk(hidden_states) 用层输入
 *              而非 RMSNorm 后的 x_norm 做路由投影！
 * prefill: Q_rot = 已 page2d 旋转的批量 Q [seq_len, q_dim]；
 *          KV = cache[0..seq_len)，全因果。
 * decode : 单 query（Q_rot = ctx->dec_q），KV = cache[0..kv_seq_len)。*/
int  ds_dash_attention_prefill(ds_ctx_t *ctx, int layer_idx,
                               const float *layer_input, const float *Q_rot,
                               int seq_len, float *out /* [seq_len, q_dim] */);
int  ds_dash_attention_decode(ds_ctx_t *ctx, int layer_idx,
                              const float *layer_input,
                              float *out /* [q_dim] */);

#endif /* DS_DASH_H */
