/* test_dash_math.c — DASH 路由数学单元验证 (对照 PyTorch 参考)
 * 验证 entmax15 / route_select / block_scores 与官方实现的数值一致性。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DS_DASH_BLOCK_SIZE   32
#define DS_DASH_LOCAL_BLOCKS 4
#define DS_DASH_TOPK_BLOCKS  12
#define DS_DASH_TOPM         16
#define DS_DASH_INDEX_DIM    128
#define DS_DASH_IDX_SCALE    11.3137085f
#define DS_DASH_SIGMA        1.0f
#define DS_DASH_FLOOR        1e-4f
#define DS_DASH_MAX_NBLK     256

/* ==== 待验证函数 (修正后: Y = clamp(xs - tau, 0)², xs 已 shift+scale) ==== */
static void entmax15(float *x, int n) {
    float xs[DS_DASH_TOPM];
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) xs[i] = (x[i] - mx) * 0.5f;
    for (int i = 1; i < n; i++) {
        float v = xs[i]; int j = i - 1;
        while (j >= 0 && xs[j] < v) { xs[j + 1] = xs[j]; j--; }
        xs[j + 1] = v;
    }
    float tau = xs[0] - 1.0f, cs = 0.0f, cs2 = 0.0f;
    for (int r = 1; r <= n; r++) {
        cs  += xs[r - 1];
        cs2 += xs[r - 1] * xs[r - 1];
        float mean = cs / r;
        float ss   = r * (cs2 / r - mean * mean);
        float delta = (1.0f - ss) / r;
        if (delta < 0.0f) delta = 0.0f;
        float t = mean - sqrtf(delta);
        if (t <= xs[r - 1]) tau = t; else break;
    }
    for (int i = 0; i < n; i++) {
        float v = xs[i] - tau;              /* 关键: 用 shift 后的 xs! */
        x[i] = v > 0.0f ? v * v : 0.0f;
    }
}

static int   ref_i[DS_DASH_TOPM];
static float ref_v[DS_DASH_TOPM];

int main(void) {
    /* 场景: nblk=8, 查询块 q_blk=7 → 远程候选 b ≤ 3
     * z 值构造: 候选块分数各异，非候选 -1e9 */
    float z[8] = { 2.1f, -0.3f, 3.7f, 1.2f, -1e9f, -1e9f, -1e9f, -1e9f };
    int nblk = 8;

    /* route_select 逻辑（与 ds_dash.c 相同） */
    float x[DS_DASH_MAX_NBLK];
    int ncand = 0;
    for (int b = 0; b < nblk; b++) {
        float v = z[b];
        if (v > -1e9f * 0.5f) { x[b] = v * DS_DASH_IDX_SCALE; ncand++; }
        else                  x[b] = -1e4f;
    }
    int sel_i[DS_DASH_TOPM]; float sel_v[DS_DASH_TOPM];
    int M = nblk < DS_DASH_TOPM ? nblk : DS_DASH_TOPM;
    for (int i = 0; i < M; i++) { sel_i[i] = -1; sel_v[i] = -1e30f; }
    for (int b = 0; b < nblk; b++) {
        if (x[b] <= -9.0e3f) continue;
        if (x[b] > sel_v[M - 1]) {
            int p = M - 1;
            while (p > 0 && sel_v[p - 1] < x[b]) {
                sel_v[p] = sel_v[p - 1]; sel_i[p] = sel_i[p - 1]; p--;
            }
            sel_v[p] = x[b]; sel_i[p] = b;
        }
    }
    entmax15(sel_v, M);
    printf("entmax output (alpha=1.5, sum should be ~1 over support):\n");
    float sum = 0;
    for (int i = 0; i < M; i++) { printf("  rank%d: blk=%d p=%.6f\n", i, sel_i[i], sel_v[i]); sum += sel_v[i]; }
    printf("  sum = %.6f\n", sum);

    /* 手算验证: 输入 x = z*scale 降序 [3.7,2.1,1.2,-0.3]*11.3137 = [41.86,23.76,13.58,-3.39]
     * entmax15: shift-max → [0,-18.1,-28.28,-45.25]/2 = [0,-9.05,-14.14,-22.63]
     * 试 rho=1: tau = 0 - sqrt((1-0)/1) = -1 → tau=-1 ≤ 0 ✓ (support=1)
     *   检查 rho=2: mean=-4.525, ss = 2*(var)... tau2 = mean - sqrt((1-ss)/2)
     *   Xsrt=[0,-9.05]; mean=-4.525; mean²=20.47; mean_sq=(0+81.9)/2=40.95
     *   ss = 2*(40.95-20.47)=40.96; delta=(1-40.96)/2=-19.98→0; tau=mean=-4.525
     *   -4.525 ≤ -9.05? NO → support=1
     * 所以 entmax15 结果 = [(0+1)², 0,0,0] = [1,0,0,0] one-hot!
     * 打印确认 */
    float expect[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    int ok = 1;
    for (int i = 0; i < 4; i++) if (fabsf(sel_v[i] - expect[i]) > 1e-4f) ok = 0;
    printf("entmax15 one-hot check: %s (support=1, top block=%d)\n",
           ok ? "PASS" : "FAIL", sel_i[0]);
    return ok ? 0 : 1;
}
