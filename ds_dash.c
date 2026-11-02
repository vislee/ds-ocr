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
#include <assert.h>

/* ── 运行时状态（挂在 ctx 上） ───────────────────────────────────────────
 * ds_dash_state_t - DASH 全部跨调用状态；官方对应 _page_rope_meta +
 * vLLM page_state + past_kv._dash_idx_k[layer]。一次识别的生命周期:
 *   ds_dash_build_coords（prefill 前建表）→ 每层 ds_dash_attention_prefill
 *   → 每步 ds_dash_advance_coords + ds_dash_attention_decode
 *   → ds_dash_reset_run（下一次识别前清历史） */
typedef struct {
    /* page2d RoPE cos/sin 表: [kv_cache_max, head_dim]（主分支与 router 共用，
     * 官方两处同参数: rotary_emb_2d_main / indexer.rotary_emb_2d） */
    float *cos2d;
    float *sin2d;

    /* 坐标表 (tok_pos, page_pos)：KV cache 每个位置一份
     * （官方 _build_page_rope_pos 的输出表；ds_dash_rope_row 的数据源） */
    int *tok_pos;
    int *page_pos;
    /* 候选页 id（0-based，官方 query_token_page_ids；-1 = 无页 → 候选域放开）
     * prefill: 图像 token → cum−1，指令 → cum−1−pp（<0 归 −1）；
     * decode : qp = mcum − 1 − pp（响应页 k ↔ 图像页 k） */
    int *cand_page;

    /* decode 状态机 (官方 _main_rope_coords 的 st)
     * cur_page/cur_tok = 最后一个已写 token 的页坐标；n_pg = 页桶总数
     * （pp 个图像页 + 1 个保留桶 0）；pf_cnt[页] = 每页 in_prefill（≤last_img）
     * token 数——decode 收到 <PAGE> 时 tok 重置为 pf_cnt[新页]，使响应正文
     * 与图像共用同一个页桶、running count 无缝续接 */
    int state_active;
    int cur_page;
    int cur_tok;
    int n_pg;                            /* 页桶总数 */
    int pf_cnt[DS_DASH_MAX_PAGES];       /* 每页 prefill(≤last_img) token 数 */
    int max_pages;

    /* 索引投影 scratch（prefill [seq,128] / decode [1,128]，算完即拷入历史）
     * 【为什么还要历史】块分数要读"全量历史"的 idx_k——见 idx_k_hist */
    float *idx_q;
    float *idx_k;
    int idx_cap;

    /* 路由结果缓冲 [seq]（prefill 批量 / decode 复用外部单条） */
    ds_dash_route_t *routes;
    int routes_cap;

    /* 按层缓存的历史 router key（官方 past_kv._dash_idx_k[layer] 追加语义）。
     * decode 的块分数必须读全量历史，且各层独立——绝不能共用一份 scratch。
     * 【bug ①（最致命）】早期实现让 12 层共用一份 scratch 且 decode 只写第
     * 0 行：层 L 的块分数读到其他层历史 + 垃圾内存；图像 token 离开局部窗
     * （生成约 128 token 后）路由全失准，输出退化为重复空坐标块。
     * 修复: [n_layers][max_seq×128]，prefill 拷入、decode 追加、按层读取。 */
    float **idx_k_hist;                  /* [n_layers][kv_cache_max * 128] */
    int n_layers;

    /* image-block 页范围（官方 _uocr_dash_decode_allow 的 bmin/bmax）。
     * 块内 image token 的页连续 → [bmin,bmax] 足够；无 image token → -1。
     * 路由的页候选判定: 块 b 是候选 ⟺ q_page ∈ [bmin[b], bmax[b]] */
    int *blk_bmin;
    int *blk_bmax;
    int nblk_prompt;                     /* prompt 覆盖的块数 */

    int pp;                              /* prompt 页数（≤last_img 的 <PAGE> 数） */
    int mcum;                            /* 累计 <PAGE> 数（prompt + 生成）。
                                            decode 候选页 qp = mcum − 1 − pp */
} ds_dash_state_t;

/* ds_dash_init - 分配 DASH 运行时状态（幂等：已初始化直接返回 0）
 * 【为什么逐层大块分配 idx_k_hist】块分数读"全量历史"必须按层隔离——这是
 * bug ①（12 层共用 scratch 导致 decode 读到跨层脏数据）的修复结构：
 * [n_layers][kv_cache_max×128] F32，prefill 拷入、decode 逐 token 追加。
 * cos2d/坐标表等大缓冲延迟到 ds_dash_build_coords 首次调用再分配（懒分配）。 */
int ds_dash_init(ds_ctx_t *ctx) {
    if (ctx->dash_state) return 0;
    ds_dash_state_t *st = (ds_dash_state_t *)calloc(1, sizeof(*st));
    if (!st) return -1;
    st->max_pages = DS_DASH_MAX_PAGES;
    st->n_layers = ctx->config.dec_layers > 0 ? ctx->config.dec_layers : 12;
    /* 提前挂到 ctx：逐层 calloc 失败时 ds_dash_free 才能真正释放已分配层。
     * st 由 calloc 清零（free 对 NULL 字段安全），失败路径返回 -1 后
     * ctx 处于一致的无 DASH 状态（free 末尾会把 dash_state 置回 NULL）。 */
    ctx->dash_state = st;
    st->idx_k_hist = (float **)calloc((size_t)st->n_layers, sizeof(float *));
    if (!st->idx_k_hist) { ds_dash_free(ctx); return -1; }
    for (int l = 0; l < st->n_layers; l++) {
        st->idx_k_hist[l] = (float *)calloc((size_t)ctx->kv_cache_max * DS_DASH_INDEX_DIM,
                                            sizeof(float));
        if (!st->idx_k_hist[l]) { ds_dash_free(ctx); return -1; }
    }
    return 0;
}

/* ds_dash_free - 释放全部 DASH 运行时缓冲并把 ctx->dash_state 置空（幂等） */
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
 * rope2d_cos_sin_row - 计算单个 (tok, page) 坐标的 cos/sin 行（dim=128）
 * 【算法】官方 DeepseekV2IndexerRotary2D(tok_freq_mode="resample", page_pairs=1,
 *   page_mod=2): half=64 个角度 = 63 个 tok 角 + 1 个 page 角:
 *     tok 轴: ang[i] = tok · 10000^(-i/63)，i = 0..62 —— resample 模式在 63 对
 *       里重新采样整个频谱（两端波长都保留），tok 是页内 running count，
 *       使页内相对距离与文档总长无关（1D RoPE 会把跨页距离拉爆频谱）；
 *     page 轴: inv_page = 256^(-0/1) = 1 → ang[63] = (page % 2) · 1.0 rad。
 *       【bug ③】这里极易写成 p/256——但 256 只是 base，指数 j/pairs=0/1=0，
 *       inv_page 恒为 1，角度就是 (page%2) 本身；多除一次 base 会把同页/跨页
 *       的相位差抹成 ~0.004 rad，页判别全部失效（docs/14 §3.2）。
 *   emb = cat(ang, ang)（NeoX 拆半配对布局：cos/sin 行 [dim] 前后半相同）。
 * 【为什么 page 取模 2】page%2 只保留"同页/不同页"信息；第几页的区分交给
 *   路由器（route_select 的页候选），位置编码不必承载页序。
 * 【参数】cosr/sinr: [dim] 输出行；tok: 页内计数；page: 页桶号；dim: 128
 */
static void rope2d_cos_sin_row(float *cosr, float *sinr, int tok, int page, int dim) {
    const int n_tok = 63;
    int half = dim / 2;
    int p = page % DS_DASH_PAGE_MOD;     /* 折叠成 0/1 交替：只判"同页/跨页" */
    float ang[64];                       /* [half] 个角度；emb = cat(ang, ang) */
    for (int i = 0; i < n_tok; i++)
        /* tok 轴角度 = tok·10000^(-i/63)：resample 模式把整个频谱重铺进 63 对，
         * 页内相对距离决定旋转差，与文档总长无关 */
        ang[i] = (float)tok * powf(DS_DASH_TOK_THETA, -(float)i / (float)n_tok);
    ang[n_tok] = (float)p;               /* inv_page = 1.0（page_pairs=1）——bug ③ 防回归点 */
    for (int i = 0; i < half; i++) {
        cosr[i] = cosf(ang[i]);          sinr[i] = sinf(ang[i]);
        cosr[i + half] = cosr[i];        sinr[i + half] = sinr[i];   /* 前后半复制 */
    }
}

/* apply_rope2d - NeoX 拆半旋转（cos/sin 已是 cat(ang,ang) 布局: 前 i 与 i+half 配对）
 * 与 ds_apply_rope_neox 数学相同；这里独立小实现是因为索引头（128 维单头）
 * 的调用形态（单向量、行指针）与主分支（批量多头）不同 */
static void apply_rope2d(float *x, const float *cosr, const float *sinr, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i]        = a * cosr[i] - b * sinr[i];
        x[i + half] = a * sinr[i] + b * cosr[i];
    }
}

/* ── page 坐标 + 候选页构建 (prefill) ────────────────────────────────────
 * ds_dash_build_coords - 由 prefill 的 token id 序列构建 page2d 坐标表并初始化
 * decode 状态机（完整页桶规则与算例见 ds_dash.h 同名声明注释）
 *
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
 * 图像 token 处 cum−1，其余 −1 → 每块取 min/max）。
 *
 * 【参数】token_ids: 本次 prefill 的全部 token id（[BOS,<PAGE>,img...,指令...]）；
 *   seq_len: 其长度。两遍扫描: 第一遍求 last_img/cum1/pp，第二遍逐 token
 *   落坐标 + 预计算 rope 表行（避免 decode 期反复算三角函数）。 */
int ds_dash_build_coords(ds_ctx_t *ctx, const int *token_ids, int seq_len) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st) return -1;
    int hd = ctx->config.dec_head_dim;
    size_t rows = (size_t)ctx->kv_cache_max;

    /* 懒分配（逐资源判断）：部分失败只缺缺的那个，下次重试补齐即可。
     * 旧写法七个一起 malloc 后一个 if 全查——任一失败后 cos2d 已非 NULL，
     * 门被永久跳过，下次调用直接解引用 NULL tok_pos（OOM 后重试即崩溃）。
     * rows/nblk_max 每 ctx 恒定，已分配的缓冲不用重分，幂等安全。 */
    int nblk_max = (int)(ctx->kv_cache_max / DS_DASH_BLOCK_SIZE) + 2;
    if (!st->cos2d)     st->cos2d    = (float *)malloc(rows * hd * sizeof(float));
    if (!st->sin2d)     st->sin2d    = (float *)malloc(rows * hd * sizeof(float));
    if (!st->tok_pos)   st->tok_pos  = (int *)malloc(rows * sizeof(int));
    if (!st->page_pos)  st->page_pos = (int *)malloc(rows * sizeof(int));
    if (!st->cand_page) st->cand_page= (int *)malloc(rows * sizeof(int));
    if (!st->blk_bmin)  st->blk_bmin = (int *)malloc((size_t)nblk_max * sizeof(int));
    if (!st->blk_bmax)  st->blk_bmax = (int *)malloc((size_t)nblk_max * sizeof(int));
    if (!st->cos2d || !st->sin2d || !st->tok_pos || !st->page_pos ||
        !st->cand_page || !st->blk_bmin || !st->blk_bmax) return -1;

    /* 第一遍: last_img 与 1-based marker 累计 cum1、pp
     * last_img = 最后一个 <image> 的位置（in_prefill 的边界）；cum1[s] =
     * s 处（含）为止遇到过的 <PAGE> 数——1-based 页计数，第一页数据 = 1 */
    int last_img = -1;
    for (int s = 0; s < seq_len; s++)
        if (token_ids[s] == DS_TOKEN_IMAGE_V4) last_img = s;
    int *cum1 = (int *)malloc((size_t)seq_len * sizeof(int));
    if (!cum1) return -1;
    int mark = 0, pp = 0;
    for (int s = 0; s < seq_len; s++) {
        /* marker 计数夹到 max_pages−1：pf_cnt 表只有 DS_DASH_MAX_PAGES 个桶 */
        if (token_ids[s] == DS_TOKEN_PAGE_MARKER && mark < st->max_pages - 1) mark++;
        cum1[s] = mark;
        if (s == last_img) pp = mark;    /* 图像结束时的页数 = prompt 页数 */
    }

    /* image-block 页范围：块 b 内 image token 的 0-based 页号取 [min,max]，
     * 供 route_select 判定"块 b 是否含 query 所在页的图像内容" */
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
    int run[DS_DASH_MAX_PAGES];          /* 每个页桶的 running count 游标 */
    memset(run, 0, sizeof(run));
    int n_pages = pp;                    /* prompt 页数（指令文本回退到保留桶 0 用） */
    for (int s = 0; s < seq_len; s++) {
        int in_pf = (s <= last_img);
        /* 页桶规则: 图像及之前 → cum1（1-based 页号）；之后（指令文本）→
         * cum1 − pp = 0 保留桶。pg<0 只可能出现在"无 marker 的退化 prompt" */
        int pg = in_pf ? cum1[s] : (cum1[s] - n_pages);
        if (pg < 0) pg = 0;
        st->page_pos[s] = pg;
        /* 页内 running count：对桶内全部行累计（含指令文本行）——官方
         * onehot.cumsum 不区分 in_prefill；段内重启会撞坐标（docs/14 §3.1） */
        st->tok_pos[s]  = run[pg]++;     /* 页桶内 running count（全部行） */
        if (in_pf) st->pf_cnt[pg]++;     /* decode 重置值只数 prefill 区 */
        /* 候选页: 图像 token → 自己的页（0-based）；指令/其他 → 无页 −1
         * （decode 放开全部 remote 的 fallback 条件之一） */
        int cp = in_pf ? (cum1[s] - 1) : (cum1[s] - 1 - pp);
        st->cand_page[s] = cp >= 0 ? cp : -1;
        /* 预计算该位置的 cos/sin 行，decode 与 Q/K 主分支旋转直接查表 */
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

/* ── 跨识别重置 ───────────────────────────────────────────────────────────
 * 【为什么必须重置 idx_k_hist】官方 vLLM 的 _dash_idx_k[layer] 追加语义
 * 服务于「多页单次 decode」：一次 infer() 里 31 个 <PAGE> 共享一个 KV，
 * 块分数读全量历史 = 本页 prompt 的历史，语义自洽。
 * C 引擎是「每页独立 decode」：ds_recognize_image() 每次把 kv_cache_len
 * 归零后重新 prefill。若不重置:
 *   1) router_qk() 会把新页 idx_k 写到 [0..seq_len)，与上一页残留重叠;
 *   2) block_scores_headcls() 的 k_bar[b] 按本页 total_kv 读 [0..total_kv)
 *      ——读到的是"新页前缀 + 上一页残留"的混合，块路由随识别次数累积
 *      失真。31页国标实测: 第14页起 6/6 页内容循环早停（111-1263 tok
 *      截断），单独/序列首位识别同页则完全正常；重置后全部消除。
 * 坐标表 (cos2d/sin2d/tok_pos/page_pos/cand_page) 由每次 prefill 的
 * ds_dash_build_coords() 全量重建，无需在此清理。 */
void ds_dash_reset_run(ds_ctx_t *ctx) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st) return;
    for (int l = 0; l < st->n_layers; l++) {
        memset(st->idx_k_hist[l], 0,
               (size_t)ctx->kv_cache_max * DS_DASH_INDEX_DIM * sizeof(float));
    }
    st->nblk_prompt = 0;
    st->pp = 0;
    st->mcum = 0;
    st->state_active = 0;   /* build_coords 会在本次 prefill 重建 */
}

/* ds_dash_advance_coords - decode 状态机推进一步：为新 token 填坐标表第 pos 行
 * （官方 _main_rope_coords decode 分支 + _uocr_dash_decode_allow 的 mcum/qp 维护）
 * 【状态机伪码】（docs/14 §5）
 *   if token == <PAGE>:
 *       mcum += 1
 *       page = min(page+1, n_pg−1)      # n_pg−1 = pp = 最后一个图像页桶
 *       tok  = pf_cnt[page]             # 响应正文接在该页图像 running count 之后
 *   else:
 *       tok += 1
 *   候选页 qp = mcum − 1 − pp           # 响应页 k ↔ 图像页 k（0-based）
 *   qp < 0 → cand_page = −1 → 候选域放开全部 remote
 * 【为什么 tok 重置为 pf_cnt[page] 而不是 0】页 k 的桶里已有 pf_cnt[k] 个
 *   prefill token（<PAGE> + 图像，单页例 1+273=274），响应正文与图像共用
 *   同一个页桶，tok 必须无缝续接；重置 0 会让两个 token 撞坐标。
 * 【单页算例】prompt [BOS,<PAGE>,img×273,指令×6]，pf_cnt[1]=274。
 *   第一个生成 token 就是 <PAGE>（文档解析输出格式）→ mcum=2 →
 *   page=min(0+1, n_pg−1=1)=1, tok=274——与图像 tok 无缝续接。
 * 【参数】new_token_id: 刚采样出的 token；写入位置 = ctx->kv_cache_len
 *   （该 token 即将占用的 KV 槽位）。 */
void ds_dash_advance_coords(ds_ctx_t *ctx, int new_token_id) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st || !st->state_active) return;
    int pos = ctx->kv_cache_len;
    int hd = ctx->config.dec_head_dim;
    if (pos < 0 || (size_t)pos >= (size_t)ctx->kv_cache_max) return;

    if (new_token_id == DS_TOKEN_PAGE_MARKER) {
        /* 生成出 <PAGE>：换页。page 沿桶号 +1 并夹到最后一个图像页桶
         * （n_pg−1=pp）——多页长文的响应页数不会超过 prompt 图像页数；
         * tok 重置为该页的 prefill 长度（running count 续接，见上） */
        st->mcum++;
        st->cur_page++;
        if (st->cur_page > st->n_pg - 1) st->cur_page = st->n_pg - 1;
        st->cur_tok = st->pf_cnt[st->cur_page];
    } else {
        /* 正文 token：同页页内计数 +1 */
        st->cur_tok++;
    }
    st->tok_pos[pos]  = st->cur_tok;
    st->page_pos[pos] = st->cur_page;
    /* 候选页 = 正在写的响应页（mcum−1 为 1-based 当前页）减去 prompt 页数
     * pp 得 0-based 图像页：响应页 k 的正文应该只路由到自己页的图像块 */
    int qp = st->mcum - 1 - st->pp;
    st->cand_page[pos] = qp >= 0 ? qp : -1;   /* <0（如前置指令阶段）→ 放开 */
    rope2d_cos_sin_row(st->cos2d + (size_t)pos * hd, st->sin2d + (size_t)pos * hd,
                       st->cur_tok, st->cur_page, hd);
}

/* ── Gemma 风格 RMSNorm (scale = 1 + weight) ───────────────────────────── */
/* gemma_rms_norm - Gemma 变体 RMSNorm: out = x·(1+w)/rms(x)（标准 RMSNorm 是 w/rms）
 * 多出的 +1 让 w=0 时退化为普通归一化（可学习缩放围绕 1 而非 0）；
 * 只用于路由索引向量（idx_q/idx_k），主分支 Q/K 仍是标准 RMSNorm。
 * 支持 out==x 原地（先算完 ss/inv 再写回）。eps 沿用 dec_rms_norm_eps。 */
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
 * 算完把 idx_k 追加到按层历史（官方 past_kv._dash_idx_k[layer] 的 concat）。
 *
 * 注意: idx_q/idx_k 只是全层共用的 scratch（[idx_cap,128]，算完即拷走）；
 * 真正按层保留的历史是 idx_k_hist。
 * 【bug ① 关键修复点】idx_k 算完立刻拷入 idx_k_hist[layer_idx]——decode 逐
 * token 追加、prefill 批量拷入；块分数读历史而非本层 scratch。若跨层共用
 * scratch，层 L 的块分数会读到其他层历史 + 垃圾内存（图像 token 离开局部窗
 * 后路由全失准 → 输出退化成重复空坐标块，docs/14 §6 bug ①）。
 */
static int router_qk(ds_ctx_t *ctx, int layer_idx,
                     const float *layer_input, int seq_len) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    const ds_dec_layer_t *layer = &ctx->decoder.layers[layer_idx];
    const ds_config_t *cfg = &ctx->config;
    int hidden = cfg->dec_hidden, d = DS_DASH_INDEX_DIM;
    int hd = cfg->dec_head_dim;

    /* 路由投影权重缺失（非 V4 模型或未加载）直接报错——DASH 路径没有
     * "退化成稠密注意力"的降级模式，静默跳过会让路由读到垃圾 */
    if (!layer->dash_q_proj_bf16 || !layer->dash_k_proj_bf16) return -1;
    if (layer_idx < 0 || layer_idx >= st->n_layers) return -1;
    /* scratch 懒增长（prefill seq_len 可达数千；decode 只需 1 行） */
    if (!st->idx_q || st->idx_cap < seq_len) {
        free(st->idx_q); free(st->idx_k);
        st->idx_cap = seq_len > 512 ? seq_len : 512;
        st->idx_q = (float *)malloc((size_t)st->idx_cap * d * sizeof(float));
        st->idx_k = (float *)malloc((size_t)st->idx_cap * d * sizeof(float));
        if (!st->idx_q || !st->idx_k) return -1;
    }
    /* 无 bias 的 BF16 matvec（路由投影权重也是 BF16 存储、F32 累加） */
    ds_linear_nobias_bf16(st->idx_q, layer_input, layer->dash_q_proj_bf16, seq_len, hidden, d);
    ds_linear_nobias_bf16(st->idx_k, layer_input, layer->dash_k_proj_bf16, seq_len, hidden, d);

    int base = ctx->kv_cache_len;   /* decode 时为已生成 token 数（历史尾部） */
    for (int s = 0; s < seq_len; s++) {
        float *q = st->idx_q + (size_t)s * d;
        float *k = st->idx_k + (size_t)s * d;
        /* 原地 GemmaRMS（q_norm/k_norm 是 (1+w) 缩放向量） */
        gemma_rms_norm(q, q, layer->dash_q_norm, d, cfg->dec_rms_norm_eps);
        gemma_rms_norm(k, k, layer->dash_k_norm, d, cfg->dec_rms_norm_eps);
        /* page2d 旋转：与主分支 Q/K 用同一套 cos/sin 表——index_dim == head_dim
         * == 128 才能共用（不等时官方退回 1D 并告警，本引擎未实现该分支）。
         * 位置 = base+s：decode 的路由 K 也必须带与 KV 相同的页坐标 */
        const float *cosr = st->cos2d + (size_t)(base + s) * hd;
        const float *sinr = st->sin2d + (size_t)(base + s) * hd;
        apply_rope2d(q, cosr, sinr, d);
        apply_rope2d(k, cosr, sinr, d);
    }
    /* 追加到该层历史缓存（位置 base..base+seq_len-1）——bug ① 修复的"写入侧" */
    memcpy(st->idx_k_hist[layer_idx] + (size_t)base * d, st->idx_k,
           (size_t)seq_len * d * sizeof(float));
    return 0;
}

/* ── 块分数 (官方 _block_scores_infer, stage0="head_cls") ────────────────
 * block_scores_headcls - 计算 [seq_len, nblk] 的块分数矩阵 z
 * 【算法】head_cls 摘要（stage0）:
 *   1. 每块算一个摘要向量 k_bar[b] = Σ_t softmax(idx_k[t]·hc/√d)·idx_k[t]
 *      —— 块内 token 对可学习向量 dash_head_cls[128] 的注意力加权平均，
 *      把 32 个 token 压成一个"块内容"代表向量；
 *   2. z[q,b] = clamp(idx_q[q]·k_bar[b]/√d, ±1e4)（query 与块摘要的相容度）；
 *   3. remote 掩码: 只有 b ≤ q_blk − local_blocks(4) 的块才是远程候选，
 *      其余 z = −1e9——局部窗已被注意力侧 token 级掩码覆盖，路由只管远处。
 * 【为什么 k_bar 与 query 无关】k_bar 只依赖 idx_k 与 head_cls，每块算一次、
 * 所有 query 复用。C 实现曾对每个 query 重复计算整张 k_bar，实测白烧
 * ~2.6 GFLOP/层（docs/14 §4.2）——这个结构本身就是那次优化的产物。
 * 【bug ① 读取侧】idx_k 必须按层读全量历史 idx_k_hist[layer_idx]（decode 时
 * 含全部已生成 token），读 scratch 只能拿到本层最近一段。
 * 【参数】seq_len: 本批 query 数；nblk: KV 总块数；total_kv: KV 总 token 数；
 *   z: [seq_len, nblk] 输出。
 */
static int block_scores_headcls(ds_ctx_t *ctx, ds_dash_state_t *st,
                                 int layer_idx, int seq_len, int nblk,
                                 int total_kv, float *z) {
    const ds_dec_layer_t *layer = &ctx->decoder.layers[layer_idx];
    const float *hc = layer->dash_head_cls;
    const float *idx_k_all = st->idx_k_hist[layer_idx];
    int d = DS_DASH_INDEX_DIM;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    int bs = DS_DASH_BLOCK_SIZE, lb = DS_DASH_LOCAL_BLOCKS;
    int base = ctx->kv_cache_len;

    /* k_bar[nblk, d]：块内 token 对 head_cls 的 softmax 加权平均（每块一次）。
     * 两遍法：先算全部分数找块内最大值 mx（数值稳定），再指数累加 k_bar。
     * OOM 时返回 -1 由调用方上报失败，绝不填 0 静默退化——
     * 全零分数经 entmax 后路由貌似合理实则跨页乱选。 */
    float *kbar = (float *)malloc((size_t)nblk * d * sizeof(float));
    if (!kbar) return -1;
    float *aw = (float *)malloc((size_t)bs * sizeof(float));
    if (!aw) { free(kbar); return -1; }

    for (int b = 0; b < nblk; b++) {
        /* 末块可能不满 32 个 token：s1 夹到 total_kv */
        int s0 = b * bs, s1 = s0 + bs;
        if (s1 > total_kv) s1 = total_kv;
        int cnt = s1 - s0;
        float *kb = kbar + (size_t)b * d;
        if (cnt <= 0) { memset(kb, 0, (size_t)d * sizeof(float)); continue; }
        float mx = -1e30f;
        for (int t = 0; t < cnt; t++) {
            const float *kt = idx_k_all + (size_t)(s0 + t) * d;
            /* 分数 = idx_k·head_cls/√d，clamp ±1e4 防 exp 溢出 */
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
        /* 除以指数和 → softmax 权重平均。esum ≥ exp(0)=1 恒成立（减去块内
         * 最大分后必有元素贡献 1），fmaxf 纯防御，非除零所需 */
        float inv_es = 1.0f / fmaxf(esum, 1e-6f);
        for (int i = 0; i < d; i++) kb[i] *= inv_es;
    }
    free(aw);

    /* 第二步：query 与块摘要的相容度 + remote 掩码 */
    for (int q = 0; q < seq_len; q++) {
        int q_blk = (base + q) / bs;     /* 本 query 所在的绝对块号 */
        const float *iq = st->idx_q + (size_t)q * d;
        float *zr = z + (size_t)q * nblk;
        for (int b = 0; b < nblk; b++) {
            const float *kb = kbar + (size_t)b * d;
            float acc = 0.0f;
            for (int i = 0; i < d; i++) acc += iq[i] * kb[i];
            acc *= inv_sqrt_d;
            if (acc > 1e4f) acc = 1e4f; else if (acc < -1e4f) acc = -1e4f;
            /* remote 掩码：b ≤ q_blk − 4 才是"远程"（局部窗 4 块已被 token 级
             * 掩码覆盖，路由不再考虑）；否则 −1e9 哨兵，route_select 会判
             * remote=false（route_select 里以 −1e9·0.5 为界） */
            zr[b] = (b <= q_blk - lb) ? acc : -1e9f;
        }
    }
    free(kbar);
    return 0;
}

/* ── entmax15 (α=1.5, 官方 _Entmax15.forward + _entmax_threshold_and_support)
 * entmax15 - α=1.5 的稀疏 softmax，原地作用于 top-M 窗口（n ≤ 16）
 * 【数学】与 softmax 的本质差异: softmax 用 e^x（永远 >0，输出稠密），entmax15
 *   用 (x−τ*)⁺²（二次门槛，输出可精确为 0——稀疏性的来源）:
 *     Y_i = clamp((X_i − max(X))/2 − τ*, 0)²
 *   τ* 由约束「支持集 ΣY_i = 1」（即平方和 = 1）解出，阈值以下的槽位清零。
 *   效果: 支持集通常远小于 16，prior 分布尖锐——路由只把概率质量集中在
 *   真正相关的块上（docs/13 §2.3 注意力稀疏化背景；docs/14 §4.3）。
 * 【τ* 求解】（官方 _entmax_threshold_and_support 精确移植）
 *   对降序 xs 每个前缀 r 求候选阈值 τ_r = mean_r − sqrt(max(0,(1−ss_r)/r))，
 *   其中 ss_r = 前缀 r 的离差平方和；support = #{r: τ_r ≤ xs[r−1]}（clamp≥1），
 *   τ* = τ_support。数学上等价于在"前 r 个非零"假设下解平方和方程。
 * 【bug ④】输出必须按【原位置】写回——官方对未排序 X 逐位输出
 *   clamp(X−τ*,0)²；C 实现曾把排序副本的值写回原数组，prior 被按大小重排、
 *   偏置与块错配（docs/14 §6 bug ④）。故排序只用于求 τ*（xs 副本），
 *   最终写回时从原 x 逐位重算。
 * 【参数】x: [n] 输入（route_select 的 top-M 窗口值），原地变为 prior 分布
 */
static void entmax15(float *x, int n) {
    float xs[DS_DASH_TOPM + 8];
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    /* α=1.5 的 /2 因子：Y = clamp((X−max)/2 − τ*, 0)²；减 max 保指数级数值稳定 */
    for (int i = 0; i < n; i++) xs[i] = (x[i] - mx) * 0.5f;

    /* 降序插入排序（n ≤ 16）— 只用于求 τ*，输出仍按原位置（bug ④ 防回归） */
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
    float cs = 0.0f, cs2 = 0.0f;         /* 前缀和、前缀平方和（增量维护） */
    int support = 0;
    for (int r = 1; r <= n; r++) {
        cs  += xs[r - 1];
        cs2 += xs[r - 1] * xs[r - 1];
        float mean = cs / r;
        /* 离差平方和: ss = Σ(xs−mean)² = Σxs² − r·mean²（展开式防两遍扫描） */
        float ss   = r * (cs2 / r - mean * mean);
        float delta = (1.0f - ss) / r;   /* 「平方和=1」约束解出的偏移项 */
        if (delta < 0.0f) delta = 0.0f;
        taus[r - 1] = mean - sqrtf(delta);
        if (taus[r - 1] <= xs[r - 1]) support++;   /* 该前缀假设自洽 */
    }
    if (support < 1) support = 1;
    float tau = taus[support - 1];
    /* 写回：按【原位置】逐位计算——绝不能用排序后的 xs（bug ④ 的根因）。
     * sentinel 槽 (x−mx)/2 − τ* ≤ 0 → 精确置 0 → prior≈0 → route_select
     * clamp 到 floor=1e-4（invalid 槽，不进 μ、注意力排除） */
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
 *    invalid 槽 bias=0 且注意力侧排除（官方 src=where(valid,bias,_MASK_NEG)）。
 *
 * ds_dash_route_select - 对单个 query 完整走一遍上述五步
 * 【K=12 的语义】（bug ② 教训）K 是部署路径直接读出的槽位数，不"额外叠加
 * 局部块"——训练路径的 _remote_topk = topk16 − local4 = 12 与部署的
 * dash_topk_blocks=12 数值巧合相等，语义不同；早期按训练语义只选 8 个远程 +
 * entmax 全块做，prior/bias 系统性偏离（docs/14 §6 bug ②）。
 * 【bug ⑤】valid 必须查 eff = remote ∧ 页候选（第 1 步的合取结果），只查页
 * 候选会让 top-M 里的 sentinel 槽拿到 −3.3 偏置参与注意力 → 小前缀输出循环。
 * 【参数】z: [nblk] 本 query 的块分数；q: 批内 query 下标（绝对位置
 *   = kv_cache_len + q）；rt: 输出路由。
 */
static void route_select(ds_ctx_t *ctx, const float *z, int nblk, int q, ds_dash_route_t *rt) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    float x[DS_DASH_MAX_NBLK];
    unsigned char cand[DS_DASH_MAX_NBLK];
    int q_abs = ctx->kv_cache_len + q;
    int q_page = st ? st->cand_page[q_abs] : -1;

    /* 候选域（官方 _build_kl_candidate_mask + _uocr_dash_decode_allow）:
     * 同页 image block；query 无页或无同页 image 块 → 全部块放开。
     * 官方 _infer_route_select 的 cand = remote ∧ block_cand——remote 掩码
     * 也在候选集里，未过 remote 的块进 top-M 窗口后必须算 invalid。
     * 【注意 cand 与 eff 的分工】cand 只含页约束（第 1 步），eff 才是
     * remote ∧ cand 的合取——valid 查 eff 而非 cand（bug ⑤ 的修法）。 */
    int n_same = 0;
    unsigned char eff[DS_DASH_MAX_NBLK];    /* 有效候选 = remote ∧ 页候选 */
    for (int b = 0; b < nblk; b++) cand[b] = 0;
    if (st && q_page >= 0) {
        /* 页候选: 只在 prompt 块里找 [bmin,bmax] 覆盖 q_page 的 image 块——
         * 生成的正文块没有页号，天然不是候选（除非走 fallback） */
        int npb = st->nblk_prompt < nblk ? st->nblk_prompt : nblk;
        for (int b = 0; b < npb; b++) {
            int bmin = st->blk_bmin[b], bmax = st->blk_bmax[b];
            if (bmin >= 0 && bmin <= q_page && q_page <= bmax) { cand[b] = 1; n_same++; }
        }
    }
    if (n_same == 0)
        /* per-query fallback：无页（前置指令）或无同页 image 块 → 不做页
         * 约束，放开全部块由路由自行挑选（官方 _build_kl_candidate_mask） */
        for (int b = 0; b < nblk; b++) cand[b] = 1;

    int ncand = 0;
    for (int b = 0; b < nblk; b++) {
        float v = z[(size_t)q * nblk + b];
        /* remote 判定: block_scores 把非远程块写成 −1e9，以 −5e8 为界区分 */
        int remote = v > -1e9f * 0.5f;
        eff[b] = (unsigned char)(remote && cand[b]);
        /* idx_scale = 128/√128 ≈ 11.31：把块分数放大到 entmax 期望量级 */
        if (eff[b]) { x[b] = v * DS_DASH_IDX_SCALE; ncand++; }
        else        { x[b] = -1e4f; }   /* 官方 _ENTMAX_NEG 哨兵 */
    }

    rt->n_valid = 0;
    if (ncand == 0 || nblk <= 0) {
        /* 连一个候选都没有：全部槽位记为无效哨兵（不进注意力） */
        for (int k = 0; k < DS_DASH_TOPK_BLOCKS; k++) {
            rt->blk[k] = -1; rt->bias[k] = 0.0f; rt->valid[k] = 0;
        }
        return;
    }

    /* top-M 窗口（降序插入，值相等保留先出现者）
     * 只取值最大的 M=min(nblk,16) 个做 entmax——sentinel（−1e4）也可能进窗口
     * 占槽（候选不足 M 时），这正是 valid 机制存在的原因 */
    float xv[DS_DASH_TOPM];
    int   xi[DS_DASH_TOPM];
    int M = nblk < DS_DASH_TOPM ? nblk : DS_DASH_TOPM;
    for (int i = 0; i < M; i++) { xv[i] = -1e30f; xi[i] = -1; }
    for (int b = 0; b < nblk; b++) {
        if (x[b] > xv[M - 1]) {
            /* 严格大于才插入 → 相等值保留先出现者（与官方 topk 的稳定序一致） */
            int p = M - 1;
            while (p > 0 && xv[p - 1] < x[b]) { xv[p] = xv[p - 1]; xi[p] = xi[p - 1]; p--; }
            xv[p] = x[b]; xi[p] = b;
        }
    }

    /* entmax15 在 top-M 窗口内（官方 prior = _entmax15(x.topk(M))）
     * 【bug ② 的一半】不能在全部 nblk 上做——部署语义是窗口内归一，
     * 否则 prior 分布被远处稀疏尾拉偏 */
    entmax15(xv, M);

    /* 前 K=min(12,M) 个槽位；valid = 该槽是有效候选（官方 is_cand = cand.gather(xi)，
     * cand 已含 remote——哨兵占据的槽 invalid，不参与 μ 也不进注意力）
     * 【bug ⑤ 的落点】eff（而非 cand）才是"remote ∧ 页候选"的合取结果 */
    int K = DS_DASH_TOPK_BLOCKS < M ? DS_DASH_TOPK_BLOCKS : M;
    float logp[DS_DASH_TOPK_BLOCKS];
    int cnt_valid = 0;
    for (int i = 0; i < K; i++) {
        rt->blk[i] = xi[i];
        rt->valid[i] = (xi[i] >= 0 && eff[xi[i]]) ? 1 : 0;
        if (rt->valid[i]) cnt_valid++;
    }
    /* logp 对全部槽位都算（invalid 槽不进 μ、注意力侧排除）。
     * sentinel 槽 prior≈0 → clamp 到 floor=1e-4 → logp≈−9.2，但 valid=0
     * ——数字会出现在 logp 里，不影响 μ 也不进注意力（docs/14 §4.3） */
    for (int i = 0; i < K; i++) {
        float pv = xv[i] > DS_DASH_FLOOR ? xv[i] : DS_DASH_FLOOR;
        logp[i] = logf(pv);
    }
    /* μ 只对 valid 槽求均值：sentinel 的 logp≈−9.2 若混入会整体拉偏偏置基准 */
    float mu = 0.0f;
    if (cnt_valid > 0) {
        for (int i = 0; i < K; i++) if (rt->valid[i]) mu += logp[i];
        mu /= (float)cnt_valid;
    }
    /* DS_V4_NOBIAS=1: 诊断开关——选块保留、偏置清零（区分选块错误 vs 偏置数值错误） */
    static int _nobias = -1;
    if (_nobias < 0) _nobias = getenv("DS_V4_NOBIAS") ? 1 : 0;
    /* bias = (logp − μ)/σ（σ=1 即纯中心化）：log-prior 偏置加在选中块的
     * 注意力分数上，让高先验块获得更大的 softmax 权重 */
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
 * attn_head_range - 单 query × head 区间 [h0,h1) 的块稀疏注意力
 * 【算法】可见 KV = 局部窗 ∪ 选中远程块的并集（docs/14 §4.1）:
 *   mask(q,t): 局部窗 (q−128, q] → bias 0（永远可见，无偏置）；
 *   t ∈ valid 选中块 b(t) → bias = log-prior 偏置（加到 QK/√d 上）；
 *   其余 → −∞（代码里直接 continue 跳过，连点积都不算——省算力的关键）。
 *   即 softmax(QK^T/√d + m) @ V，m 只有 0/bias/−∞ 三种取值。
 * 【在线 softmax】逐 KV 流式维护 (run_m, run_s, O) 三元组（docs/13 §2），
 *   遇更大分数先修旧账再写入——两段离散可见域（局部窗 + 远处散块）共用
 *   同一组在线状态，正是 FA"分段扫描、状态合并"的思想。
 * 【稀疏度算账】（块稀疏省 FLOP 的账）
 *   全注意力: 每 query 扫 total_kv 个 KV，每个一次 128 维 QK 点积 → O(total_kv·d)。
 *   DASH 可见域 ≤ 局部窗 128 + K×32 = 12×32 = 384，合计 ≤512 个 KV 位置
 *   （局部窗与最远一档选中块 b=q_blk−4 可能重叠最多 32 个 token，实际略少）
 *   → O(512·d)。total_kv=2048 时约省 4×，kv_cache_max=8192 满载时约 16×；
 *   路由自身开销（k_bar 每块一次、z 每 query 对 nblk 块各一次 128 维点积）
 *   是低阶项——注意力成本不再随长文档线性放大，这是 DASH 的存在意义。
 * 【与全注意力的等价边界】
 *   1. q_abs < 局部窗 128: 全部 KV 天然可见、bias 恒 0、无 continue 跳过
 *      → 与稠密全注意力逐位一致（对照官方 _forward_dense 路径的回归依据）；
 *   2. 一般情形与全注意力的差 = 被掩掉的未选中远程块 + 选中块上的
 *      log-prior 偏置——仅当全部远程块都被选中且偏置为 0 时才退回全注意力；
 *   3. 局部窗与选中块共用一组在线 (m,l,O) 状态，数学上等价于对两段可见域
 *      做同一次 softmax（FA 分段合并公式保证）；浮点下结果随扫描顺序存在
 *      舍入差——C 按 KV 位置递增扫描，与官方实现同序，可复现对齐。
 * 【为什么 V4 的 KV 用 F32】代码保留 BF16 通路（ds_dot_bf16xf32 精确展开），
 *   但 r2 实验中 BF16 KV 在 V4 打红（DASH 对 KV 精度敏感，与 V3 全绿相反）
 *   而回退——config 维持 F32（docs/14 §7 对比表）。
 * Qbuf: 已旋转的 Q [seq_len, q_dim]（prefill 传批量缓冲，decode 传 ctx->dec_q）。
 * 单 query × head 区间 [h0,h1) 一份实现，供串行/两种并行粒度复用。
 * 【参数】rt: 本 query 的路由；O_row: [q_dim] 输出（本函数只写 [h0,h1) 段） */
static void attn_head_range(ds_ctx_t *ctx, int layer_idx, const float *Qbuf,
                            const ds_dash_route_t *rt, int q, int h0, int h1,
                            float *O_row) {
    const ds_config_t *cfg = &ctx->config;
    int nkv = cfg->dec_kv_heads, hd = cfg->dec_head_dim;
    int q_dim = cfg->dec_heads * hd;
    (void)q_dim;
    int groups = cfg->dec_heads / nkv;         /* GQA 分组比（10Q/2KV → 5） */
    int lw = DS_DASH_LOCAL_BLOCKS * DS_DASH_BLOCK_SIZE;   /* 局部窗 = 4×32 = 128 */
    float scale = 1.0f / sqrtf((float)hd);
    int q_abs = ctx->kv_cache_len + q;         /* 本 query 的绝对 KV 位置 */
    int nh = h1 - h0;

    /* 只清自己的 head 区间——decode 按 head 并行时各线程写互不相交 */
    memset(O_row + (size_t)h0 * hd, 0, (size_t)nh * hd * sizeof(float));
    /* 在线 softmax 状态（FA 三元组）: run_m=行最大分、run_s=exp 和；O 在 O_row。
     * 数组按最大 64 头开（V4 实际 10 头）；h1−h0 超限说明 config 头数被改坏，
     * 直接 abort 而非栈溢出。 */
    assert(nh <= 64 && "run_m/run_s sized for max 64 heads; check dec_heads config");
    float run_m[64], run_s[64];
    for (int i = 0; i < nh; i++) { run_m[i] = -1e30f; run_s[i] = 0.0f; }

    /* 扫 [0, q_abs] 全部 KV，逐 token 判定归属（局部窗 / 选中块 / 跳过） */
    for (int t = 0; t <= q_abs; t++) {
        int tb = t / DS_DASH_BLOCK_SIZE;       /* t 所属块号 */
        int is_local = (t > q_abs - lw);            /* (q-128, q] */
        float bias = 0.0f;
        if (!is_local) {
            /* 非局部：默认 −inf（用 −1e30 表示），在选中块里找本块偏置 */
            bias = -1e30f;
            /* 在选中块列表里找本块：命中则整块共用该槽的 log-prior 偏置——
             * 路由粒度是块、注意力粒度是 token，块内部仍是稠密逐 token 计算，
             * 只是每个 token 的分数统一加上同一个 bias */
            for (int k = 0; k < rt->n_valid; k++) {
                if (!rt->valid[k] || rt->blk[k] != tb) continue;
                bias = rt->bias[k]; break;
            }
            /* 未选中块 → -inf：直接 continue，连 QK 点积都省掉
             * （这是块稀疏省算力的来源：远处只算被路由选中的 ~12 块） */
            if (bias <= -1e29f) continue;           /* 未选中块 → -inf */
        }
        /* 取本 token 的 K/V 行（F32/BF16 双宽度，由 ctx->kv_bf16 决定；
         * 布局见 ds_dash.h 的 ds_kv_k_row/ds_kv_v_row 行访问注释） */
        const void *kt = ds_kv_k_row(ctx, layer_idx, t);
        const void *vt = ds_kv_v_row(ctx, layer_idx, t);
        for (int hh = 0; hh < nh; hh++) {
            int h = h0 + hh;
            /* GQA: Q 头 h 读 KV 头 h/groups 的行切片 */
            const float *qh = Qbuf + (size_t)q * q_dim + (size_t)h * hd;
            float s;
            if (ctx->kv_bf16) {
                /* BF16 KV: bf16 行切片 × f32 Q 点积（展开精确，顺序不变；r2 任务2） */
                s = ds_dot_bf16xf32((const uint16_t *)kt + (size_t)(h / groups) * hd, qh, hd) * scale + bias;
            } else {
                const float *kh = (const float *)kt + (size_t)(h / groups) * hd;
                s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
                s = s * scale + bias;   /* log-prior 偏置直接加进分数（局部窗 bias=0） */
            }
            if (s > run_m[hh]) {
                /* 新最大值：先修旧账——O 与 l 都乘 corr = e^{m_old−m_new} */
                float f = expf(run_m[hh] - s);
                float *Oh = O_row + (size_t)h * hd;
                for (int i = 0; i < hd; i++) Oh[i] *= f;
                run_s[hh] *= f;
                run_m[hh] = s;
            }
            /* 以当前 m 为基准累加新项（局部窗与远程块共用同一组在线状态——
             * 两段离散可见域在数学上就是同一次 softmax 的两个分段） */
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
    /* 收尾归一化 O /= l。正常路径下 t=q_abs 恒为局部可见，l>0；
     * 这里兜底防除零（空 head 区间 h0==h1 时直接跳过） */
    for (int hh = 0; hh < nh; hh++) {
        float inv = run_s[hh] > 0.0f ? 1.0f / run_s[hh] : 0.0f;
        float *Oh = O_row + (size_t)(h0 + hh) * hd;
        for (int i = 0; i < hd; i++) Oh[i] *= inv;
    }
}

/* 并行 worker：decode(seq_len==1) 按 head 切分；prefill 按 query 切分
 *
 * dash_attn_task_t - 并行任务描述（by_head 决定切分粒度）：
 *   decode 只有 1 个 query，按 query 切没有并行度 → 沿 head 维切，
 *   各线程写 out 的 head 区间互不相交；
 *   prefill 有 seq_len 个 query、每个 query 都要全 head → 按 query 切，
 *   各线程写 out 的行区间互不相交。
 */
typedef struct {
    ds_ctx_t *ctx;
    int layer_idx;
    const float *Qbuf;
    const ds_dash_route_t *routes;       /* [seq_len] 每 query 一份路由 */
    int seq_len;
    float *out;
    int by_head;                         /* 1: 单 query 按 head 切；0: 按 query 切 */
} dash_attn_task_t;

/* dash_attn_worker - ds_parallel_for 的线程体：tid 号线程处理自己的条带
 * （条带 = ceil(N/nt) 连续块；两端夹进 [0,N) 防越界） */
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

/* biased_attention - 带偏置注意力的调度入口（并行 or 串行）
 * 线程池可用（>1 线程）且任务可切分（prefill 多 query，或 decode 的 10 头
 * ≥2）时走 ds_parallel_for；否则串行逐 query 全 head。
 * 【注意偏置的并行安全性】路由 rt/out 的写区间按切分粒度互斥（见 task 注释），
 * 逐位结果与串行一致。 */
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

/* ── prefill: 路由 + 带偏置注意力 ────────────────────────────────────────
 * ds_dash_attention_prefill - V4 prefill 的一次注意力（对应官方
 * _infer_prefill_attend + _forward_dense；契约见 ds_dash.h 声明注释）
 * 【流程】router_qk（idx_q/idx_k + 拷入层历史）→ block_scores_headcls
 *   （z [seq_len,nblk]）→ 每 query route_select → biased_attention
 * 【参数】x_norm: 形参名叫 x_norm 但实际传的是【层输入 x】（残差流）——
 *   ds_moe_decoder.c 的 prefill 分支传 x；官方 _router_qk(hidden_states)
 *   用层输入而非 RMSNorm 后的 x_norm，参数名沿用历史勿被误导。
 *   Q_rot: 已按 page2d 旋转的批量 Q（ds_moe_decoder.c 用 ds_dash_rope_row
 *   取 cos/sin 行后旋转）；out: [seq_len, q_dim]。
 * 【为什么 total_kv = kv_cache_len + seq_len】C 引擎每次识别从 0 prefill，
 *   但代码支持续写/分块 prefill（base>0）；nblk 随之 ceil(total_kv/32)。 */
int ds_dash_attention_prefill(ds_ctx_t *ctx, int layer_idx,
                              const float *x_norm, const float *Q_rot,
                              int seq_len, float *out) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    int total_kv = ctx->kv_cache_len + seq_len;   /* 含本批 prefill token 的 KV 总数 */
    int nblk = (total_kv + DS_DASH_BLOCK_SIZE - 1) / DS_DASH_BLOCK_SIZE;  /* KV 总块数 */

    if (router_qk(ctx, layer_idx, x_norm, seq_len) != 0) return -1;
    /* 路由结果缓冲懒增长（最小 1024 条，避免小 prefill 反复 malloc） */
    if (!st->routes || st->routes_cap < seq_len) {
        free(st->routes);
        st->routes_cap = seq_len > 1024 ? seq_len + 16 : 1024;
        st->routes = (ds_dash_route_t *)malloc((size_t)st->routes_cap * sizeof(ds_dash_route_t));
        if (!st->routes) return -1;
    }
    float *z = (float *)malloc((size_t)seq_len * nblk * sizeof(float));
    if (!z) return -1;
    if (block_scores_headcls(ctx, st, layer_idx, seq_len, nblk, total_kv, z) != 0) {
        free(z); return -1;   /* OOM：上报失败，不静默降级 */
    }
    /* prefill 的每个 query 独立路由（页候选/远程域都可能不同） */
    for (int q = 0; q < seq_len; q++)
        route_select(ctx, z, nblk, q, &st->routes[q]);
    free(z);

    biased_attention(ctx, layer_idx, Q_rot, st->routes, seq_len, out);
    return 0;
}

/* ── decode: 单 query ────────────────────────────────────────────────────
 * ds_dash_attention_decode - V4 decode 单步注意力（官方 _dash_decode_forward
 * + _decode_attend；契约见 ds_dash.h）
 * 与 prefill 共用同一套路由数学，差别有三（docs/14 §4.4）:
 *   1. idx_k 按层追加历史（router_qk 内 base=kv_cache_len 写入 1 行）；
 *   2. 候选域用状态机的 qp = mcum−1−pp 页匹配（cand_page 已在
 *      ds_dash_advance_coords 算好）；
 *   3. Q/K 的 page2d 角由 ds_dash_rope_row 提供（主分支在 ds_moe_decoder.c 旋转）。
 * 【参数】x_norm: 同 prefill——实为层输入 x；out: [q_dim]（本步注意力输出） */
int ds_dash_attention_decode(ds_ctx_t *ctx, int layer_idx,
                             const float *x_norm, float *out) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    int total_kv = ctx->kv_cache_len + 1;   /* 含本步即将写入的 token */
    int nblk = (total_kv + DS_DASH_BLOCK_SIZE - 1) / DS_DASH_BLOCK_SIZE;  /* KV 总块数 */

    if (router_qk(ctx, layer_idx, x_norm, 1) != 0) return -1;
    /* decode 的 z 只有 1 行（单 query 对全部块的分数） */
    float *z = (float *)malloc((size_t)nblk * sizeof(float));
    if (!z) return -1;
    if (block_scores_headcls(ctx, st, layer_idx, 1, nblk, total_kv, z) != 0) {
        free(z); return -1;   /* OOM：上报失败，不静默降级 */
    }
    ds_dash_route_t rt;
    route_select(ctx, z, nblk, 0, &rt);
    free(z);

    /* 复用核心: routes 数组长度1, Q = ctx->dec_q (单 token)
     * biased_attention 的签名吃"路由数组"，这里把单条 rt 经 st->routes
     * 指针传入（保存/恢复原批量缓冲指针），避免为 1 行复制一份数组 */
    ds_dash_route_t *one = &rt;
    ds_dash_route_t *save = st->routes;
    st->routes = one;
    biased_attention(ctx, layer_idx, ctx->dec_q, st->routes, 1, out);
    st->routes = save;
    return 0;
}

/* ── V4 main-branch page2d RoPE 表行访问 ─────────────────────────────────
 * ds_dash_rope_row - 主分支 Q/K 旋转的 cos/sin 行入口（ds_moe_decoder.c 的
 * decode/prefill V4 分支经 extern 声明调用）
 * 【为什么主分支也用这张表】sparse_index_dim == head_dim == 128，路由器与
 * 主分支共用同一套 page2d cos/sin（官方 rotary_emb_2d_main 与
 * indexer.rotary_emb_2d 同参数）；tok/page 坐标来自本模块的坐标表。
 * 【fallback】状态未就绪（未 init/未 build_coords）时退回 1D rope_cache——
 * 不应用于 V4 正常路径，仅防未初始化崩溃。
 * which: 0=cos, 1=sin。返回 [head_dim] 行只读指针（内部缓冲，勿释放）。
 * 原型在 ds_dash.h（曾被 ds_moe_decoder.c 用局部 extern 调用，签名漂移即静默 UB）。 */
const float *ds_dash_rope_row(ds_ctx_t *ctx, int pos, int which) {
    ds_dash_state_t *st = (ds_dash_state_t *)ctx->dash_state;
    if (!st || !st->cos2d || pos < 0) {
        /* 状态未就绪时退回 1D 表（不应用于 V4 正常路径） */
        return which == 0 ? ctx->rope_cache_cos + (size_t)pos * ctx->config.dec_head_dim
                          : ctx->rope_cache_sin + (size_t)pos * ctx->config.dec_head_dim;
    }
    return which == 0 ? st->cos2d + (size_t)pos * ctx->config.dec_head_dim
                      : st->sin2d + (size_t)pos * ctx->config.dec_head_dim;
}
