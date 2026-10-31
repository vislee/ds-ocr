/*
 * ds_dash.c — V4 (Unlimited-OCR-2) DASH 块稀疏注意力实现
 *
 * 逐行对照官方 _INFER 推理路径 (vLLM parity):
 *   modeling_deepseekv2.py  — _dash_attn_forward / _infer_prefill_attend /
 *                             _block_scores_infer / _infer_route_select /
 *                             _dash_decode_forward / _decode_attend /
 *                             DeepseekV2IndexerRotary2D / _main_rope_coords
 *   modeling_unlimitedocr.py— _uocr_dash_decode_allow (decode 候选域)
 *   _sparse_base_impl.py    — _build_page_rope_pos / _page_rope_offsets
 *
 * 所有数值公式与官方一致；路由参数来自 config.json 的 sparse_attention_config:
 *   block_size=32, local_blocks=4(→局部窗128), K=dash_topk_blocks=12,
 *   topm=16, index_dim=128, idx_scale=dash_scale/√128=11.3137,
 *   sigma=1.0, floor=1e-4, page2d RoPE(tok_theta=10000 resample,
 *   page_pairs=1 → page 轴角 = (page%2)·1.0 rad, page_mod=2)。
 */
#include "ds_dash.h"
#include "ds_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── 运行时状态（挂在 ctx 上） ─────────────────────────────────────────── */
typedef struct {
    /* page2d RoPE cos/sin 表: [kv_cache_max, head_dim]（主分支与 router 共用，
     * 官方两处同参数: rotary_emb_2d_main / indexer.rotary_emb_2d） */
    float *cos2d;
    float *sin2d;

    /* 坐标表 (tok_pos, page_pos)：KV cache 每个位置一份 */
    int *tok_pos;
    int *page_pos;
    /* 候选页 id（0-based，官方 query_token_page_ids；-1 = 无页 → 候选域放开） */
    int *cand_page;

    /* decode 状态机 (官方 _main_rope_coords 的 st) */
    int state_active;
    int cur_page;
    int cur_tok;
    int n_pg;                            /* 页桶总数 */
    int pf_cnt[DS_DASH_MAX_PAGES];       /* 每页 prefill(≤last_img) token 数 */
    int max_pages;

    /* 索引投影 scratch（prefill [seq,128] / decode [1,128]，算完即拷入历史） */
    float *idx_q;
    float *idx_k;
    int idx_cap;

    /* 路由结果缓冲 [seq]（prefill 批量 / decode 复用外部单条） */
    ds_dash_route_t *routes;
    int routes_cap;

    /* 按层缓存的历史 router key（官方 past_kv._dash_idx_k[layer] 追加语义）。
     * decode 的块分数必须读全量历史，且各层独立——绝不能共用一份 scratch。 */
    float **idx_k_hist;                  /* [n_layers][kv_cache_max * 128] */
    int n_layers;

    /* image-block 页范围（官方 _uocr_dash_decode_allow 的 bmin/bmax）。
     * 块内 image token 的页连续 → [bmin,bmax] 足够；无 image token → -1。 */
    int *blk_bmin;
    int *blk_bmax;
    int nblk_prompt;                     /* prompt 覆盖的块数 */

    int pp;                              /* prompt 页数（≤last_img 的 <PAGE> 数） */
    int mcum;                            /* 累计 <PAGE> 数（prompt + 生成） */
} ds_dash_state_t;

int ds_dash_init(ds_ctx_t *ctx) {
    if (ctx->dash_state) return 0;
    ds_dash_state_t *st = (ds_dash_state_t *)calloc(1, sizeof(*st));
    if (!st) return -1;
    st->max_pages = DS_DASH_MAX_PAGES;
    st->n_layers = ctx->config.dec_layers > 0 ? ctx->config.dec_layers : 12;
    st->idx_k_hist = (float **)calloc((size_t)st->n_layers, sizeof(float *));
    if (!st->idx_k_hist) { free(st); ctx->dash_state = NULL; return -1; }
    for (int l = 0; l < st->n_layers; l++) {
        st->idx_k_hist[l] = (float *)calloc((size_t)ctx->kv_cache_max * DS_DASH_INDEX_DIM,
                                            sizeof(float));
        if (!st->idx_k_hist[l]) { ds_dash_free(ctx); return -1; }
    }
    ctx->dash_state = st;
    return 0;
}

void ds_dash_free(ds_ctx_t *ctx) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st) return;
    free(st->cos2d); free(st->sin2d);
    free(st->tok_pos); free(st->page_pos); free(st->cand_page);
    if (st->idx_k_hist) {
        for (int l = 0; l < st->n_layers; l++) free(st->idx_k_hist[l]);
        free(st->idx_k_hist);
    }
    free(st->blk_bmin); free(st->blk_bmax);
    free(st->routes); free(st->idx_q); free(st->idx_k);
    free(st);
    ctx->dash_state = NULL;
}

/* ── page2d 二维 RoPE ────────────────────────────────────────────────────
 * 官方 DeepseekV2IndexerRotary2D(tok_freq_mode="resample", page_pairs=1,
 * page_mod=2): half=64, n_tok=63 对 → inv_tok[i]=10000^(-i/63)；
 * page 1 对 → inv_page = 256^(-0/1) = 1，角度 = (page%2)·1.0 rad。
 * ang = cat(a_tok, a_page)，emb = cat(ang, ang)。 */
static void rope2d_cos_sin_row(float *cosr, float *sinr, int tok, int page, int dim) {
    const int n_tok = 63;
    int half = dim / 2;
    int p = page % DS_DASH_PAGE_MOD;
    float ang[64];                       /* [half] 个角度；emb = cat(ang, ang) */
    for (int i = 0; i < n_tok; i++)
        ang[i] = (float)tok * powf(DS_DASH_TOK_THETA, -(float)i / (float)n_tok);
    ang[n_tok] = (float)p;               /* inv_page = 1.0（page_pairs=1） */
    for (int i = 0; i < half; i++) {
        cosr[i] = cosf(ang[i]);          sinr[i] = sinf(ang[i]);
        cosr[i + half] = cosr[i];        sinr[i + half] = sinr[i];
    }
}

/* neox 旋转（cos/sin 已是 cat(ang,ang) 布局: 前 i 与 i+half 配对） */
static void apply_rope2d(float *x, const float *cosr, const float *sinr, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i]        = a * cosr[i] - b * sinr[i];
        x[i + half] = a * sinr[i] + b * cosr[i];
    }
}

/* ── page 坐标 + 候选页构建 (prefill) ────────────────────────────────────
 * 官方 _build_page_rope_pos / _page_rope_offsets 完整规则:
 *   page(rope): <PAGE> 1-based 累计；最后一个图像 token 之后的 token
 *         (指令文本) page = cum − pp → 0 保留桶（BOS 也 page=0）。
 *   tok : 页桶内 running count − 1（对桶内【全部】行累计，含指令文本；
 *         官方 onehot.cumsum 不区分 in_prefill）。
 *   pf_cnt: 仅统计 in_prefill (≤last_img) 的每页 token 数——decode 状态机
 *         <PAGE> 重置 tok 用（官方 pf_cnt = (onehot·in_prefill).sum）。
 *   cand_page(0-based): in_prefill → cum−1；之后 → cum−1−pp；<0 → −1。
 *   pp  = last_img 处的 marker 累计（prompt 页数）。
 * 同时构建 image-block 页范围 bmin/bmax（官方 image_token_page_ids =
 * 图像 token 处 cum−1，其余 −1 → 每块取 min/max）。 */
int ds_dash_build_coords(ds_ctx_t *ctx, const int *token_ids, int seq_len) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st) return -1;
    int hd = ctx->config.dec_head_dim;
    size_t rows = (size_t)ctx->kv_cache_max;

    if (!st->cos2d) {
        st->cos2d   = (float *)malloc(rows * hd * sizeof(float));
        st->sin2d   = (float *)malloc(rows * hd * sizeof(float));
        st->tok_pos = (int *)malloc(rows * sizeof(int));
        st->page_pos= (int *)malloc(rows * sizeof(int));
        st->cand_page=(int *)malloc(rows * sizeof(int));
        int nblk_max = (int)(ctx->kv_cache_max / DS_DASH_BLOCK_SIZE) + 2;
        st->blk_bmin = (int *)malloc((size_t)nblk_max * sizeof(int));
        st->blk_bmax = (int *)malloc((size_t)nblk_max * sizeof(int));
        if (!st->cos2d || !st->sin2d || !st->tok_pos || !st->page_pos ||
            !st->cand_page || !st->blk_bmin || !st->blk_bmax) return -1;
    }

    /* 第一遍: last_img 与 1-based marker 累计 cum1、pp */
    int last_img = -1;
    for (int s = 0; s < seq_len; s++)
        if (token_ids[s] == DS_TOKEN_IMAGE_V4) last_img = s;
    int *cum1 = (int *)malloc((size_t)seq_len * sizeof(int));
    if (!cum1) return -1;
    int mark = 0, pp = 0;
    for (int s = 0; s < seq_len; s++) {
        if (token_ids[s] == DS_TOKEN_PAGE_MARKER && mark < st->max_pages - 1) mark++;
        cum1[s] = mark;
        if (s == last_img) pp = mark;
    }

    /* image-block 页范围 */
    {
        int nblk_prompt = (seq_len + DS_DASH_BLOCK_SIZE - 1) / DS_DASH_BLOCK_SIZE;
        for (int b = 0; b < nblk_prompt; b++) { st->blk_bmin[b] = -1; st->blk_bmax[b] = -1; }
        for (int s = 0; s < seq_len; s++) {
            if (token_ids[s] != DS_TOKEN_IMAGE_V4) continue;
            int b = s / DS_DASH_BLOCK_SIZE;
            int p0 = cum1[s] - 1;            /* 0-based 图像页 */
            if (st->blk_bmin[b] < 0 || p0 < st->blk_bmin[b]) st->blk_bmin[b] = p0;
            if (p0 > st->blk_bmax[b]) st->blk_bmax[b] = p0;
        }
        st->nblk_prompt = nblk_prompt;
    }

    /* 第二遍: 坐标 + 候选页 + pf_cnt + rope 表 */
    memset(st->pf_cnt, 0, sizeof(st->pf_cnt));
    int run[DS_DASH_MAX_PAGES];
    memset(run, 0, sizeof(run));
    int n_pages = pp;
    for (int s = 0; s < seq_len; s++) {
        int in_pf = (s <= last_img);
        int pg = in_pf ? cum1[s] : (cum1[s] - n_pages);
        if (pg < 0) pg = 0;
        st->page_pos[s] = pg;
        st->tok_pos[s]  = run[pg]++;     /* 页桶内 running count（全部行） */
        if (in_pf) st->pf_cnt[pg]++;     /* decode 重置值只数 prefill 区 */
        int cp = in_pf ? (cum1[s] - 1) : (cum1[s] - 1 - pp);
        st->cand_page[s] = cp >= 0 ? cp : -1;
        rope2d_cos_sin_row(st->cos2d + (size_t)s * hd, st->sin2d + (size_t)s * hd,
                           st->tok_pos[s], st->page_pos[s], hd);
    }
    free(cum1);
    /* decode 状态机初值 = 最后一个 prompt token 的坐标（官方 st['page'/'tok']
     * 取 coords 末行）。指令文本在 0 桶；生成的响应文本与指令同桶递增，
     * 遇 <PAGE> 切到下一页桶（tok 重置为该页 pf_cnt）。 */
    st->cur_page = st->page_pos[seq_len - 1];
    st->cur_tok  = st->tok_pos[seq_len - 1];
    st->n_pg     = (last_img >= 0 ? pp : 0) + 1;
    st->pp       = pp;
    st->mcum     = pp;
    st->state_active = 1;
    return 0;
}

/* decode 状态机推进 (官方 _main_rope_coords decode 分支 +
 * _uocr_dash_decode_allow 的 mcum/qp 维护):
 *   <PAGE> → mcum++, page=min(page+1, n_pg-1)，tok=pf_cnt[page]；否则 tok+=1
 *   候选页 qp = mcum − 1 − pp（响应页 k ↔ 图像页 k），<0 → −1（放开） */
void ds_dash_advance_coords(ds_ctx_t *ctx, int new_token_id) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st || !st->state_active) return;
    int pos = ctx->kv_cache_len;
    int hd = ctx->config.dec_head_dim;
    if (pos < 0 || (size_t)pos >= (size_t)ctx->kv_cache_max) return;

    if (new_token_id == DS_TOKEN_PAGE_MARKER) {
        st->mcum++;
        st->cur_page++;
        if (st->cur_page > st->n_pg - 1) st->cur_page = st->n_pg - 1;
        st->cur_tok = st->pf_cnt[st->cur_page];
    } else {
        st->cur_tok++;
    }
    st->tok_pos[pos]  = st->cur_tok;
    st->page_pos[pos] = st->cur_page;
    int qp = st->mcum - 1 - st->pp;
    st->cand_page[pos] = qp >= 0 ? qp : -1;
    rope2d_cos_sin_row(st->cos2d + (size_t)pos * hd, st->sin2d + (size_t)pos * hd,
                       st->cur_tok, st->cur_page, hd);
}

/* ── Gemma 风格 RMSNorm (scale = 1 + weight) ───────────────────────────── */
static void gemma_rms_norm(float *out, const float *x, const float *w,
                           int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.0f + w[i]);
}

/* ── 索引投影 (官方 _router_qk) ──────────────────────────────────────────
 * 输入 = 层输入 hidden_states（残差流，非 x_norm！官方 _infer_prefill_attend
 * 与 _dash_decode_forward 都传 hidden_states）。
 * idx_q = page2drope(GemmaRMS(q_norm(q_proj(x))))   [S,128]
 * idx_k = page2drope(GemmaRMS(k_norm(k_proj(x))))   [S,128]
 * 算完把 idx_k 追加到按层历史（官方 past_kv._dash_idx_k[layer] 的 concat）。 */
static int router_qk(ds_ctx_t *ctx, int layer_idx,
                     const float *layer_input, int seq_len) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    const ds_dec_layer_t *layer = &ctx->decoder.layers[layer_idx];
    const ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden, d = DS_DASH_INDEX_DIM;
    int hd = cfg->dec_head_dim;

    if (!layer->dash_q_proj_bf16 || !layer->dash_k_proj_bf16) return -1;
    if (layer_idx < 0 || layer_idx >= st->n_layers) return -1;
    if (!st->idx_q || st->idx_cap < seq_len) {
        free(st->idx_q); free(st->idx_k);
        st->idx_cap = seq_len > 512 ? seq_len : 512;
        st->idx_q = (float *)malloc((size_t)st->idx_cap * d * sizeof(float));
        st->idx_k = (float *)malloc((size_t)st->idx_cap * d * sizeof(float));
        if (!st->idx_q || !st->idx_k) return -1;
    }
    ds_linear_nobias_bf16(st->idx_q, layer_input, layer->dash_q_proj_bf16, seq_len, hidden, d);
    ds_linear_nobias_bf16(st->idx_k, layer_input, layer->dash_k_proj_bf16, seq_len, hidden, d);

    int base = ctx->kv_cache_len;
    for (int s = 0; s < seq_len; s++) {
        float *q = st->idx_q + (size_t)s * d;
        float *k = st->idx_k + (size_t)s * d;
        gemma_rms_norm(q, q, layer->dash_q_norm, d, cfg->dec_rms_norm_eps);
        gemma_rms_norm(k, k, layer->dash_k_norm, d, cfg->dec_rms_norm_eps);
        const float *cosr = st->cos2d + (size_t)(base + s) * hd;
        const float *sinr = st->sin2d + (size_t)(base + s) * hd;
        apply_rope2d(q, cosr, sinr, d);
        apply_rope2d(k, cosr, sinr, d);
    }
    /* 追加到该层历史缓存（位置 base..base+seq_len-1） */
    memcpy(st->idx_k_hist[layer_idx] + (size_t)base * d, st->idx_k,
           (size_t)seq_len * d * sizeof(float));
    return 0;
}

/* ── 块分数 (官方 _block_scores_infer, stage0="head_cls") ────────────────
 * k_bar[b] = Σ softmax(idx_k·hc/√d)·idx_k   —— 与 query 无关，算一次
 * z[q,b]   = clamp(idx_q[q]·k_bar[b]/√d)；remote 掩码: b ≤ q_blk − local_blocks
 * idx_k 全量读按层历史（官方 past_kv._dash_idx_k[layer]）。 */
static void block_scores_headcls(ds_ctx_t *ctx, ds_dash_state_t *st,
                                 int layer_idx, int seq_len, int nblk,
                                 int total_kv, float *z) {
    const ds_dec_layer_t *layer = &ctx->decoder.layers[layer_idx];
    const float *hc = layer->dash_head_cls;
    const float *idx_k_all = st->idx_k_hist[layer_idx];
    int d = DS_DASH_INDEX_DIM;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    int bs = DS_DASH_BLOCK_SIZE, lb = DS_DASH_LOCAL_BLOCKS;
    int base = ctx->kv_cache_len;

    /* k_bar[nblk, d]：块内 token 对 head_cls 的 softmax 加权平均（每块一次） */
    float *kbar = (float *)malloc((size_t)nblk * d * sizeof(float));
    if (!kbar) { for (size_t i = 0; i < (size_t)seq_len * nblk; i++) z[i] = 0.0f; return; }
    float *aw = (float *)malloc((size_t)bs * sizeof(float));
    if (!aw) { free(kbar); for (size_t i = 0; i < (size_t)seq_len * nblk; i++) z[i] = 0.0f; return; }

    for (int b = 0; b < nblk; b++) {
        int s0 = b * bs, s1 = s0 + bs;
        if (s1 > total_kv) s1 = total_kv;
        int cnt = s1 - s0;
        float *kb = kbar + (size_t)b * d;
        if (cnt <= 0) { memset(kb, 0, (size_t)d * sizeof(float)); continue; }
        float mx = -1e30f;
        for (int t = 0; t < cnt; t++) {
            const float *kt = idx_k_all + (size_t)(s0 + t) * d;
            float a = 0.0f;
            for (int i = 0; i < d; i++) a += kt[i] * hc[i];
            a *= inv_sqrt_d;
            if (a > 1e4f) a = 1e4f; else if (a < -1e4f) a = -1e4f;
            aw[t] = a;
            if (a > mx) mx = a;
        }
        memset(kb, 0, (size_t)d * sizeof(float));
        float esum = 0.0f;
        for (int t = 0; t < cnt; t++) {
            float e = expf(aw[t] - mx);
            esum += e;
            const float *kt = idx_k_all + (size_t)(s0 + t) * d;
            for (int i = 0; i < d; i++) kb[i] += e * kt[i];
        }
        float inv_es = 1.0f / fmaxf(esum, 1e-6f);
        for (int i = 0; i < d; i++) kb[i] *= inv_es;
    }
    free(aw);

    for (int q = 0; q < seq_len; q++) {
        int q_blk = (base + q) / bs;
        const float *iq = st->idx_q + (size_t)q * d;
        float *zr = z + (size_t)q * nblk;
        for (int b = 0; b < nblk; b++) {
            const float *kb = kbar + (size_t)b * d;
            float acc = 0.0f;
            for (int i = 0; i < d; i++) acc += iq[i] * kb[i];
            acc *= inv_sqrt_d;
            if (acc > 1e4f) acc = 1e4f; else if (acc < -1e4f) acc = -1e4f;
            zr[b] = (b <= q_blk - lb) ? acc : -1e9f;
        }
    }
    free(kbar);
}

/* ── entmax15 (α=1.5, 官方 _Entmax15.forward + _entmax_threshold_and_support)
 * Y = clamp((X − max(X))/2 − τ*, 0)²，τ* 由「支持集平方和 = 1」解出，
 * 输出按【原位置】写回（官方对未排序 X 逐位计算）。n ≤ topm(16)。 */
static void entmax15(float *x, int n) {
    float xs[DS_DASH_TOPM + 8];
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) xs[i] = (x[i] - mx) * 0.5f;

    /* 降序插入排序（n ≤ 16）— 只用于求 τ*，输出仍按原位置 */
    for (int i = 1; i < n; i++) {
        float v = xs[i]; int j = i - 1;
        while (j >= 0 && xs[j] < v) { xs[j + 1] = xs[j]; j--; }
        xs[j + 1] = v;
    }
    /* τ* 求解（官方 _entmax_threshold_and_support 精确移植）:
     * 对每个前缀 r 都算 τ_r = mean_r − sqrt(max(0,(1−ss_r)/r))，
     * support = #{r: τ_r ≤ xs[r-1]}（clamp min 1），τ* = τ_{support}。
     * 注意不能在第一次违反时就 break——τ_r − xs[r-1] 非严格单调，
     * 官方按计数取值，个别行会取到更后面的 τ。 */
    float taus[DS_DASH_TOPM + 8];
    float cs = 0.0f, cs2 = 0.0f;
    int support = 0;
    for (int r = 1; r <= n; r++) {
        cs  += xs[r - 1];
        cs2 += xs[r - 1] * xs[r - 1];
        float mean = cs / r;
        float ss   = r * (cs2 / r - mean * mean);
        float delta = (1.0f - ss) / r;
        if (delta < 0.0f) delta = 0.0f;
        taus[r - 1] = mean - sqrtf(delta);
        if (taus[r - 1] <= xs[r - 1]) support++;
    }
    if (support < 1) support = 1;
    float tau = taus[support - 1];
    for (int i = 0; i < n; i++) {
        float v = (x[i] - mx) * 0.5f - tau;
        x[i] = v > 0.0f ? v * v : 0.0f;
    }
}

/* ── 路由选择 (官方 _infer_route_select，vLLM DashRouterMath.route_and_select
 * budget_topk) ────────────────────────────────────────────────────────────
 * 1. 候选域 cand = remote ∧ block_cand（页感知 image-block，per-query；
 *    无候选 → 放开全部 remote，官方 _build_kl_candidate_mask fallback）。
 * 2. x = cand ? z·idx_scale : −1e4（官方 _ENTMAX_NEG）。
 * 3. top-M=16 窗口（按值降序取前 M 个，含哨兵）→ entmax15（窗口内归一）。
 * 4. 读预算 K = min(dash_topk_blocks=12, M) 个槽位；valid = 该槽原候选标记。
 * 5. bias = (log(clamp(prior,floor)) − μ)/σ，μ 只对 valid 求均值；
 *    invalid 槽 bias=0 且注意力侧排除（官方 src=where(valid,bias,_MASK_NEG)）。 */
static void route_select(ds_ctx_t *ctx, const float *z, int nblk, int q, ds_dash_route_t *rt) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    float x[DS_DASH_MAX_NBLK];
    unsigned char cand[DS_DASH_MAX_NBLK];
    int q_abs = ctx->kv_cache_len + q;
    int q_page = st ? st->cand_page[q_abs] : -1;

    /* 候选域（官方 _build_kl_candidate_mask + _uocr_dash_decode_allow）:
     * 同页 image block；query 无页或无同页 image 块 → 全部块放开。
     * 官方 _infer_route_select 的 cand = remote ∧ block_cand——remote 掩码
     * 也在候选集里，未过 remote 的块进 top-M 窗口后必须算 invalid。 */
    int n_same = 0;
    unsigned char eff[DS_DASH_MAX_NBLK];    /* 有效候选 = remote ∧ 页候选 */
    for (int b = 0; b < nblk; b++) cand[b] = 0;
    if (st && q_page >= 0) {
        int npb = st->nblk_prompt < nblk ? st->nblk_prompt : nblk;
        for (int b = 0; b < npb; b++) {
            int bmin = st->blk_bmin[b], bmax = st->blk_bmax[b];
            if (bmin >= 0 && bmin <= q_page && q_page <= bmax) { cand[b] = 1; n_same++; }
        }
    }
    if (n_same == 0)
        for (int b = 0; b < nblk; b++) cand[b] = 1;

    int ncand = 0;
    for (int b = 0; b < nblk; b++) {
        float v = z[(size_t)q * nblk + b];
        int remote = v > -1e9f * 0.5f;
        eff[b] = (unsigned char)(remote && cand[b]);
        if (eff[b]) { x[b] = v * DS_DASH_IDX_SCALE; ncand++; }
        else        { x[b] = -1e4f; }   /* 官方 _ENTMAX_NEG 哨兵 */
    }

    rt->n_valid = 0;
    if (ncand == 0 || nblk <= 0) {
        for (int k = 0; k < DS_DASH_TOPK_BLOCKS; k++) {
            rt->blk[k] = -1; rt->bias[k] = 0.0f; rt->valid[k] = 0;
        }
        return;
    }

    /* top-M 窗口（降序插入，值相等保留先出现者） */
    float xv[DS_DASH_TOPM];
    int   xi[DS_DASH_TOPM];
    int M = nblk < DS_DASH_TOPM ? nblk : DS_DASH_TOPM;
    for (int i = 0; i < M; i++) { xv[i] = -1e30f; xi[i] = -1; }
    for (int b = 0; b < nblk; b++) {
        if (x[b] > xv[M - 1]) {
            int p = M - 1;
            while (p > 0 && xv[p - 1] < x[b]) { xv[p] = xv[p - 1]; xi[p] = xi[p - 1]; p--; }
            xv[p] = x[b]; xi[p] = b;
        }
    }

    /* entmax15 在 top-M 窗口内（官方 prior = _entmax15(x.topk(M))） */
    entmax15(xv, M);

    /* 前 K=min(12,M) 个槽位；valid = 该槽是有效候选（官方 is_cand = cand.gather(xi)，
     * cand 已含 remote——哨兵占据的槽 invalid，不参与 μ 也不进注意力） */
    int K = DS_DASH_TOPK_BLOCKS < M ? DS_DASH_TOPK_BLOCKS : M;
    float logp[DS_DASH_TOPK_BLOCKS];
    int cnt_valid = 0;
    for (int i = 0; i < K; i++) {
        rt->blk[i] = xi[i];
        rt->valid[i] = (xi[i] >= 0 && eff[xi[i]]) ? 1 : 0;
        if (rt->valid[i]) cnt_valid++;
    }
    /* logp 对全部槽位都算（invalid 槽不进 μ、注意力侧排除） */
    for (int i = 0; i < K; i++) {
        float pv = xv[i] > DS_DASH_FLOOR ? xv[i] : DS_DASH_FLOOR;
        logp[i] = logf(pv);
    }
    float mu = 0.0f;
    if (cnt_valid > 0) {
        for (int i = 0; i < K; i++) if (rt->valid[i]) mu += logp[i];
        mu /= (float)cnt_valid;
    }
    /* DS_V4_NOBIAS=1: 诊断开关——选块保留、偏置清零（区分选块错误 vs 偏置数值错误） */
    static int _nobias = -1;
    if (_nobias < 0) _nobias = getenv("DS_V4_NOBIAS") ? 1 : 0;
    for (int i = 0; i < K; i++) {
        rt->bias[i] = (_nobias || !rt->valid[i]) ? 0.0f : (logp[i] - mu) / DS_DASH_SIGMA;
    }
    rt->n_valid = K;
    /* DS_DUMP_ROUTE=1: 诊断——decode（kv_cache_len>0）每 10 步打印 z/选中块/偏置 */
    if (getenv("DS_DUMP_ROUTE") && ctx->config.model_version == 4 &&
        ctx->kv_cache_len > 0 && (ctx->kv_cache_len % 10) == 0) {
        fprintf(stderr, "[route q=%d] z:", ctx->kv_cache_len);
        for (int b = 0; b < nblk; b++)
            fprintf(stderr, " %d:%.1f", b, z[(size_t)q * nblk + b]);
        fprintf(stderr, " | sel:");
        for (int i = 0; i < K; i++)
            fprintf(stderr, " %d(b=%.2f,v=%d)", rt->blk[i], rt->bias[i], rt->valid[i]);
        fprintf(stderr, "\n");
    }
}

/* ── 共享: 带偏置注意力核心 ───────────────────────────────────────────────
 * mask(q,t): 局部窗 (q-lw, q] → 0；valid 选中块 b(t) → bias；其余 → 跳过。
 * Qbuf: 已旋转的 Q [seq_len, q_dim]（prefill 传批量缓冲，decode 传 ctx->dec_q）。
 * 单 query × head 区间 [h0,h1) 一份实现，供串行/两种并行粒度复用。 */
static void attn_head_range(ds_ctx_t *ctx, int layer_idx, const float *Qbuf,
                            const ds_dash_route_t *rt, int q, int h0, int h1,
                            float *O_row) {
    const ds_config_t *cfg = &ctx->config;
    int nkv = cfg->dec_kv_heads, hd = cfg->dec_head_dim;
    int q_dim = cfg->dec_heads * hd;
    (void)q_dim;
    int groups = cfg->dec_heads / nkv;
    int lw = DS_DASH_LOCAL_BLOCKS * DS_DASH_BLOCK_SIZE;
    float scale = 1.0f / sqrtf((float)hd);
    int q_abs = ctx->kv_cache_len + q;
    int nh = h1 - h0;

    /* 只清自己的 head 区间——decode 按 head 并行时各线程写互不相交 */
    memset(O_row + (size_t)h0 * hd, 0, (size_t)nh * hd * sizeof(float));
    float run_m[64], run_s[64];
    for (int i = 0; i < nh; i++) { run_m[i] = -1e30f; run_s[i] = 0.0f; }

    for (int t = 0; t <= q_abs; t++) {
        int tb = t / DS_DASH_BLOCK_SIZE;
        int is_local = (t > q_abs - lw);            /* (q-128, q] */
        float bias = 0.0f;
        if (!is_local) {
            bias = -1e30f;
            for (int k = 0; k < rt->n_valid; k++) {
                if (!rt->valid[k] || rt->blk[k] != tb) continue;
                bias = rt->bias[k]; break;
            }
            if (bias <= -1e29f) continue;           /* 未选中块 → -inf */
        }
        const void *kt = ds_kv_k_row(ctx, layer_idx, t);
        const void *vt = ds_kv_v_row(ctx, layer_idx, t);
        for (int hh = 0; hh < nh; hh++) {
            int h = h0 + hh;
            const float *qh = Qbuf + (size_t)q * q_dim + (size_t)h * hd;
            float s;
            if (ctx->kv_bf16) {
                /* BF16 KV: bf16 行切片 × f32 Q 点积（展开精确，顺序不变；r2 任务2） */
                s = ds_dot_bf16xf32((const uint16_t *)kt + (size_t)(h / groups) * hd, qh, hd) * scale + bias;
            } else {
                const float *kh = (const float *)kt + (size_t)(h / groups) * hd;
                s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
                s = s * scale + bias;
            }
            if (s > run_m[hh]) {
                float f = expf(run_m[hh] - s);
                float *Oh = O_row + (size_t)h * hd;
                for (int i = 0; i < hd; i++) Oh[i] *= f;
                run_s[hh] *= f;
                run_m[hh] = s;
            }
            float e = expf(s - run_m[hh]);
            float *Oh = O_row + (size_t)h * hd;
            if (ctx->kv_bf16) {
                ds_axpy_bf16xf32(Oh, (const uint16_t *)vt + (size_t)(h / groups) * hd, e, hd);
            } else {
                const float *vh = (const float *)vt + (size_t)(h / groups) * hd;
                for (int i = 0; i < hd; i++) Oh[i] += e * vh[i];
            }
            run_s[hh] += e;
        }
    }
    for (int hh = 0; hh < nh; hh++) {
        float inv = run_s[hh] > 0.0f ? 1.0f / run_s[hh] : 0.0f;
        float *Oh = O_row + (size_t)(h0 + hh) * hd;
        for (int i = 0; i < hd; i++) Oh[i] *= inv;
    }
}

/* 并行 worker：decode(seq_len==1) 按 head 切分；prefill 按 query 切分 */
typedef struct {
    ds_ctx_t *ctx;
    int layer_idx;
    const float *Qbuf;
    const ds_dash_route_t *routes;
    int seq_len;
    float *out;
    int by_head;                         /* 1: 单 query 按 head 切；0: 按 query 切 */
} dash_attn_task_t;

static void dash_attn_worker(int tid, int nt, void *arg) {
    dash_attn_task_t *t = (dash_attn_task_t *)arg;
    ds_config_t *cfg = &t->ctx->config;
    int nh = cfg->dec_heads;
    int q_dim = nh * cfg->dec_head_dim;
    if (t->by_head) {
        int chunk = (nh + nt - 1) / nt;
        int h0 = tid * chunk;
        int h1 = h0 + chunk;
        if (h1 > nh) h1 = nh;
        if (h0 < h1)
            attn_head_range(t->ctx, t->layer_idx, t->Qbuf, &t->routes[0], 0,
                            h0, h1, t->out);
    } else {
        int chunk = (t->seq_len + nt - 1) / nt;
        int q0 = tid * chunk;
        int q1 = q0 + chunk;
        if (q1 > t->seq_len) q1 = t->seq_len;
        for (int q = q0; q < q1; q++)
            attn_head_range(t->ctx, t->layer_idx, t->Qbuf, &t->routes[q], q,
                            0, nh, t->out + (size_t)q * q_dim);
    }
}

static void biased_attention(ds_ctx_t *ctx, int layer_idx, const float *Qbuf,
                             const ds_dash_route_t *routes, int seq_len,
                             float *out) {
    int nthreads = ds_get_threads();
    if (nthreads > 1 && (seq_len > 1 || ctx->config.dec_heads >= 2)) {
        dash_attn_task_t task = {
            .ctx = ctx, .layer_idx = layer_idx, .Qbuf = Qbuf,
            .routes = routes, .seq_len = seq_len, .out = out,
            .by_head = (seq_len == 1)
        };
        ds_parallel_for(dash_attn_worker, &task);
        return;
    }
    int nh = ctx->config.dec_heads;
    int q_dim = nh * ctx->config.dec_head_dim;
    for (int q = 0; q < seq_len; q++)
        attn_head_range(ctx, layer_idx, Qbuf, &routes[q], q, 0, nh,
                        out + (size_t)q * q_dim);
}

/* ── prefill: 路由 + 带偏置注意力 ──────────────────────────────────────── */
int ds_dash_attention_prefill(ds_ctx_t *ctx, int layer_idx,
                              const float *x_norm, const float *Q_rot,
                              int seq_len, float *out) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    int total_kv = ctx->kv_cache_len + seq_len;
    int nblk = (total_kv + DS_DASH_BLOCK_SIZE - 1) / DS_DASH_BLOCK_SIZE;

    if (router_qk(ctx, layer_idx, x_norm, seq_len) != 0) return -1;
    if (!st->routes || st->routes_cap < seq_len) {
        free(st->routes);
        st->routes_cap = seq_len > 1024 ? seq_len + 16 : 1024;
        st->routes = (ds_dash_route_t *)malloc((size_t)st->routes_cap * sizeof(ds_dash_route_t));
        if (!st->routes) return -1;
    }
    float *z = (float *)malloc((size_t)seq_len * nblk * sizeof(float));
    if (!z) return -1;
    block_scores_headcls(ctx, st, layer_idx, seq_len, nblk, total_kv, z);
    for (int q = 0; q < seq_len; q++)
        route_select(ctx, z, nblk, q, &st->routes[q]);
    free(z);

    biased_attention(ctx, layer_idx, Q_rot, st->routes, seq_len, out);
    return 0;
}

/* ── decode: 单 query ──────────────────────────────────────────────────── */
int ds_dash_attention_decode(ds_ctx_t *ctx, int layer_idx,
                             const float *x_norm, float *out) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    int total_kv = ctx->kv_cache_len + 1;
    int nblk = (total_kv + DS_DASH_BLOCK_SIZE - 1) / DS_DASH_BLOCK_SIZE;

    if (router_qk(ctx, layer_idx, x_norm, 1) != 0) return -1;
    float *z = (float *)malloc((size_t)nblk * sizeof(float));
    if (!z) return -1;
    block_scores_headcls(ctx, st, layer_idx, 1, nblk, total_kv, z);
    ds_dash_route_t rt;
    route_select(ctx, z, nblk, 0, &rt);
    free(z);

    /* 复用核心: routes 数组长度1, Q = ctx->dec_q (单 token) */
    ds_dash_route_t *one = &rt;
    ds_dash_route_t *save = st->routes;
    st->routes = one;
    biased_attention(ctx, layer_idx, ctx->dec_q, st->routes, 1, out);
    st->routes = save;
    return 0;
}

/* ── V4 main-branch page2d RoPE 表行访问 ─────────────────────────────────
 * ds_moe_decoder.c 的 Q/K 旋转需要按 KV 位置取 cos/sin 行。
 * which: 0=cos, 1=sin。返回 [head_dim] 行指针（内部缓冲，勿释放）。*/
void *ds_dash_rope_row(ds_ctx_t *ctx, int pos, int which) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st || !st->cos2d || pos < 0) {
        /* 状态未就绪时退回 1D 表（不应用于 V4 正常路径） */
        return which == 0 ? ctx->rope_cache_cos + (size_t)pos * ctx->config.dec_head_dim
                          : ctx->rope_cache_sin + (size_t)pos * ctx->config.dec_head_dim;
    }
    return which == 0 ? st->cos2d + (size_t)pos * ctx->config.dec_head_dim
                      : st->sin2d + (size_t)pos * ctx->config.dec_head_dim;
}
