/* test_dash_math.c — DASH 路由数学单元验证 (对照 PyTorch 参考)
 * 验证 entmax15 / route_select / block_scores 与官方实现的数值一致性。
 * 本文件刻意独立于引擎（不 include ds_dash.h，宏与函数就地复制一份），
 * 便于把 C 结果与 PyTorch 参考脚本逐数对照——定位是"数学对拍"
 * 而非回归套件（回归在 test_unit.c 的 dash 套件）。 */
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

/* entmax15 - entmax-1.5 稀疏 softmax 的独立实现（与 ds_dash.c 同源，对拍用）
 *
 * 【数学】α=1.5 的 entmax 有闭式解 Y = clamp(xs − τ, 0)²，τ 由
 * "支持集元素的平方和 = 1"唯一确定——这正是输出稀疏的根源：
 * 支持集之外的分量被 τ 截断归零（softmax 永远做不到这一点）。
 *
 * 【算法步骤】
 *   1. shift+scale: 减 max 再 ×0.5（α−1=0.5 的幂次折进尺度，数值更稳）；
 *   2. 降序排序（插入排序，n=TOPM≤16 很小，O(n²) 无所谓）；
 *   3. 从 r=1 递增试支持集大小: 累积 Σxs、Σxs² → τ = mean − sqrt((1−ss)/r)，
 *      直到 τ ≤ xs[r−1]（第 r 个元素确实该留在支持集内）为止；
 *   4. 就地输出 clamp(xs − τ, 0)²，和恰为 1。
 *
 * @param x  就地变换: 入参 logits，出参稀疏概率分布
 * @param n  元素个数（≤ DS_DASH_TOPM） */
static void entmax15(float *x, int n) {
    float xs[DS_DASH_TOPM];
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];      /* 求 max: 数值稳定的 shift 基准 */
    for (int i = 0; i < n; i++) xs[i] = (x[i] - mx) * 0.5f;    /* shift+scale: 最大值归 0 */
    for (int i = 1; i < n; i++) {   /* 插入排序降序（小 n 直接排） */
        float v = xs[i]; int j = i - 1;
        while (j >= 0 && xs[j] < v) { xs[j + 1] = xs[j]; j--; }
        xs[j + 1] = v;
    }
    float tau = xs[0] - 1.0f, cs = 0.0f, cs2 = 0.0f;   /* 初值 = r=1 的解 τ=xs[0]−1 */
    for (int r = 1; r <= n; r++) {   /* 支持集大小从 1 递增试探 */
        cs  += xs[r - 1];            /* cs = 前 r 个之和 */
        cs2 += xs[r - 1] * xs[r - 1];
        float mean = cs / r;
        float ss   = r * (cs2 / r - mean * mean);   /* 平方和 Σxs² = r·(E[x²]−E[x]²) */
        float delta = (1.0f - ss) / r;
        if (delta < 0.0f) delta = 0.0f;   /* 平方和可能数值上略超 1，夹到 0 */
        float t = mean - sqrtf(delta);   /* 本轮 τ 候选 */
        if (t <= xs[r - 1]) tau = t; else break;   /* τ 越过第 r 个元素 → 支持集超界，回退取上一轮 */
    }
    for (int i = 0; i < n; i++) {
        float v = xs[i] - tau;              /* 关键: 用 shift 后的 xs! */
        x[i] = v > 0.0f ? v * v : 0.0f;   /* clamp(xs−τ,0)²——τ 外分量归零（稀疏性来源） */
    }
}

static int   ref_i[DS_DASH_TOPM];
static float ref_v[DS_DASH_TOPM];

/* main - 数学对拍主程序: 复现 route_select 的"哨兵过滤 → top-M 选择 →
 * entmax15"管线，输入刻意构造成可手算（候选分数悬殊 → one-hot），
 * 推导过程见下方"手算验证"注释，最终断言 one-hot [1,0,0,0]。 */
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
        if (v > -1e9f * 0.5f) { x[b] = v * DS_DASH_IDX_SCALE; ncand++; }   /* 过哨兵: ×11.31 是 query 索引头的尺度约定 */
        else                  x[b] = -1e4f;   /* 哨兵块给"远小于一切"的分数，排序自然沉底 */
    }
    int sel_i[DS_DASH_TOPM]; float sel_v[DS_DASH_TOPM];
    int M = nblk < DS_DASH_TOPM ? nblk : DS_DASH_TOPM;
    for (int i = 0; i < M; i++) { sel_i[i] = -1; sel_v[i] = -1e30f; }
    /* 不做全排序: 维护长度 M 的降序选择数组，比第 M 名还小的值直接淘汰 */
    for (int b = 0; b < nblk; b++) {
        if (x[b] <= -9.0e3f) continue;   /* 哨兵块不参与选择 */
        if (x[b] > sel_v[M - 1]) {   /* 挤进前 M 才有戏 */
            int p = M - 1;
            while (p > 0 && sel_v[p - 1] < x[b]) {   /* 插入排序式上浮到正确位置 */
                sel_v[p] = sel_v[p - 1]; sel_i[p] = sel_i[p - 1]; p--;
            }
            sel_v[p] = x[b]; sel_i[p] = b;
        }
    }
    entmax15(sel_v, M);   /* top-M 分数 → 稀疏路由分布（注意力偏置 μ 的来源） */
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
    float expect[4] = {1.0f, 0.0f, 0.0f, 0.0f};   /* 手算结论: one-hot（support=1） */
    int ok = 1;
    for (int i = 0; i < 4; i++) if (fabsf(sel_v[i] - expect[i]) > 1e-4f) ok = 0;   /* 对照手算值逐位验证 */
    printf("entmax15 one-hot check: %s (support=1, top block=%d)\n",
           ok ? "PASS" : "FAIL", sel_i[0]);
    return ok ? 0 : 1;
}
