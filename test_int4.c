/*
 * test_int4.c — Per-Row INT8 Quantization Kernel Correctness Verification
 *
 * Tests:
 *   1. RMS quantization error for typical MoE weight shapes
 *   2. Matvec accuracy: INT8 vs BF16 reference
 *   3. Dequantize round-trip verification
 *   4. matvec_rows sub-row consistency
 *
 * Build:  make test_int4
 * Run:    ./test_int4
 */

#include "ds_kernels.h"
#include "ds_quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* max_abs_diff_f - 两向量逐元素绝对差的最大值（最坏单点误差度量） */
static float max_abs_diff_f(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>mx) mx=d; }
    return mx;
}
/* relative_error_f - 相对 L2 误差 = ||ref−test||₂ / ||ref||₂
 * 用 double 累积平方和，防止大矩阵 float 求和的精度损耗干扰结论；
 * 相比 max_abs 更贴近"能量占比"，是量化验收的常用口径。 */
static float relative_error_f(const float *ref, const float *test, int n) {
    double num=0, den=0;
    for (int i=0;i<n;i++) { num+=(double)(ref[i]-test[i])*(ref[i]-test[i]); den+=(double)ref[i]*ref[i]; }
    return den>0 ? sqrtf((float)(num/den)) : 0.0f;
}
/* bf16_matvec_ref - BF16 权重矩阵向量乘的朴素参考实现（被测内核的"未量化真值"）
 * BF16→F32 无损展开（位型 <<16 即 f32），逐行点积，无任何优化技巧。 */
static void bf16_matvec_ref(float *y, const float *x, const uint16_t *W, int in_dim, int out_dim) {
    for (int o=0;o<out_dim;o++) {
        const uint16_t *wr = W+(size_t)o*in_dim;
        float sum=0;
        for (int i=0;i<in_dim;i++) { uint32_t u=((uint32_t)wr[i])<<16; float w; memcpy(&w,&u,4); sum+=w*x[i]; }
        y[o]=sum;
    }
}
/* lcg_rand - 线性同余伪随机数，输出 [-1,1] 均匀分布（固定种子 → 可复现）。
 * 均匀随机权重对量化是最坏情形之一（无低秩结构可供 scale 利用），
 * 用它验收通过的误差上界对真实权重同样成立。 */
static float lcg_rand(unsigned int *seed, float lo, float hi) {
    *seed=*seed*1103515245+12345; return lo+(hi-lo)*((*seed&0x7fffffff)/(float)0x7fffffff);
}

/* test_rms_error - 量化 RMS 误差验证 (docs/09)
 * 三种真实 MoE 权重形状（gate_up_fused / down_weight / shared_gate_up）
 * 各自量化→反量化，对照原权重算 RMS 误差，验收线 <0.02；
 * 同时打印压缩比（BF16 2B/值 → INT8 1B/值 + 8B/行元数据，约 1.95x）。
 * 期望值依据: 255 级量化的理论 RMS ≈ 1/(255·√3) ≈ 0.23%，
 * 实测 0.4-1.0%，0.02 是留足余量的工程验收线。 */
static int test_rms_error(void) {
    printf("=== Test 1: Per-Row INT8 RMS Error ===\n");
    struct { int od; int id; const char *n; } shapes[] = {
        {1792,1280,"gate_up_fused [1792,1280]"},
        {1280,896,"down_weight  [1280,896]"},
        {3584,1280,"shared_gate_up [3584,1280]"},
        {0,0,NULL}
    };
    int pass=1;
    for (int s=0; shapes[s].n; s++) {
        int od=shapes[s].od, id=shapes[s].id;
        size_t total=(size_t)od*id;
        uint16_t *W=(uint16_t*)malloc(total*2);
        unsigned int seed=42+s;   /* 每个形状换种子，避免共享随机序列 */
        for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }   /* f32→BF16: 截取高 16 位（位型截断） */
        ds_int4_block_t blk;
        ds_int4_quantize_bf16(&blk, W, od, id);
        float rms = ds_int4_quant_error_rms(&blk, W);
        int ok = (rms < 0.02f);
        printf("  %-28s RMS=%.6f %s (%.1fx smaller)\n", shapes[s].n, rms, ok?"PASS":"FAIL", (double)(total*2)/blk.bytes);
        if (!ok) pass=0;
        ds_int4_block_free(&blk);
        free(W);
    }
    return pass;
}

/* test_matvec_accuracy - 量化端到端: INT8 matvec vs BF16 参考对拍 (docs/09)
 * [1792,1280] 量化权重 + 随机输入，对照 bf16_matvec_ref 的相对 L2
 * 误差 <0.02 通过。误差来源 = 量化噪声在 in_dim 次乘加中部分相消，
 * 故整体误差与单值 RMS 同量级——2% 是工程验收线。 */
static int test_matvec_accuracy(void) {
    printf("\n=== Test 2: INT8 Matvec vs BF16 ===\n");
    int od=1792, id=1280;
    size_t total=(size_t)od*id;
    uint16_t *W=(uint16_t*)malloc(total*2);
    float *x=(float*)malloc(id*4), *yr=(float*)malloc(od*4), *yi=(float*)malloc(od*4);
    unsigned int seed=12345;
    for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }
    for (int i=0;i<id;i++) x[i]=lcg_rand(&seed,-1,1);
    bf16_matvec_ref(yr, x, W, id, od);
    ds_int4_block_t blk;
    ds_int4_quantize_bf16(&blk, W, od, id);
    ds_int4_matvec(yi, x, &blk);
    float rel=relative_error_f(yr,yi,od), mad=max_abs_diff_f(yr,yi,od);
    printf("  max_abs=%.4f, rel_err=%.6f %s\n", mad, rel, rel<0.02f?"PASS":"FAIL");
    ds_int4_block_free(&blk);
    free(W); free(x); free(yr); free(yi);
    return rel < 0.02f;
}

/* test_dequantize - 反量化往返: 量化→反量化应还原原权重 (docs/09)
 * 最"纯"的误差测度——不含输入与累加，只测量化本身:
 * Wf（BF16 位型精确展开的原值）对照 Wd（量化再反量化），<0.02 通过。
 * 往返误差 = test_rms_error 的同一物理量，换一种口径互相印证。 */
static int test_dequantize(void) {
    printf("\n=== Test 3: Dequantize Round-Trip ===\n");
    int od=1280, id=896;
    size_t total=(size_t)od*id;
    uint16_t *W=(uint16_t*)malloc(total*2);
    float *Wf=(float*)malloc(total*4), *Wd=(float*)malloc(total*4);
    unsigned int seed=99;
    for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }
    for (size_t i=0;i<total;i++) { uint32_t u=((uint32_t)W[i])<<16; memcpy(&Wf[i],&u,4); }   /* BF16<<16 = 精确 f32 展开（无损真值） */
    ds_int4_block_t blk;
    ds_int4_quantize_bf16(&blk, W, od, id);
    ds_int4_dequantize_f32(Wd, &blk);
    float rel=relative_error_f(Wf,Wd,(int)total);
    printf("  rel_err=%.6f %s\n", rel, rel<0.02f?"PASS":"FAIL");
    ds_int4_block_free(&blk);
    free(W); free(Wf); free(Wd);
    return rel < 0.02f;
}

/* test_matvec_rows - 分片接口一致性: 全量 matvec vs 只算前 896 行 (docs/05)
 * 线程池按行分片时走 ds_int4_matvec_rows；两条路径只应差循环边界，
 * 数学上必须逐位一致（max_diff < 1e-5 容忍打印/比较的浮点噪声）。
 * 分片实现若引入 per-row scale 错位，这里当场暴露。 */
static int test_matvec_rows(void) {
    printf("\n=== Test 4: matvec_rows Consistency ===\n");
    int od=1792, id=1280;
    size_t total=(size_t)od*id;
    uint16_t *W=(uint16_t*)malloc(total*2);
    float *x=(float*)malloc(id*4), *yf=(float*)malloc(od*4), *ys=(float*)malloc(896*4);
    unsigned int seed=77;
    for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }
    for (int i=0;i<id;i++) x[i]=lcg_rand(&seed,-1,1);
    ds_int4_block_t blk;
    ds_int4_quantize_bf16(&blk, W, od, id);
    ds_int4_matvec(yf, x, &blk);
    ds_int4_matvec_rows(ys, x, &blk, 0, 896);   /* 只算前一半输出行（分片语义） */
    float mad=max_abs_diff_f(yf, ys, 896);
    printf("  full vs rows: max_diff=%.10f %s\n", mad, mad<1e-5f?"PASS":"FAIL");
    ds_int4_block_free(&blk);
    free(W); free(x); free(yf); free(ys);
    return mad < 1e-5f;
}

/* main - 依次跑 4 个量化验证，任一失败退出码 1（CI 判据）。
 * 先切单线程，排除线程调度带来的非确定性，保证结果可复现。 */
int main(void) {
    printf("Per-Row INT8 Quantization Kernel Verification\n");
    printf("==============================================\n\n");
    ds_set_threads(1);
    int all=1;
    if (!test_rms_error()) all=0;
    if (!test_matvec_accuracy()) all=0;
    if (!test_dequantize()) all=0;
    if (!test_matvec_rows()) all=0;
    printf("\n==============================================\n");
    printf("OVERALL: %s\n", all?"ALL TESTS PASSED":"SOME TESTS FAILED");
    return all?0:1;
}
