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

static float max_abs_diff_f(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>mx) mx=d; }
    return mx;
}
static float relative_error_f(const float *ref, const float *test, int n) {
    double num=0, den=0;
    for (int i=0;i<n;i++) { num+=(double)(ref[i]-test[i])*(ref[i]-test[i]); den+=(double)ref[i]*ref[i]; }
    return den>0 ? sqrtf((float)(num/den)) : 0.0f;
}
static void bf16_matvec_ref(float *y, const float *x, const uint16_t *W, int in_dim, int out_dim) {
    for (int o=0;o<out_dim;o++) {
        const uint16_t *wr = W+(size_t)o*in_dim;
        float sum=0;
        for (int i=0;i<in_dim;i++) { uint32_t u=((uint32_t)wr[i])<<16; float w; memcpy(&w,&u,4); sum+=w*x[i]; }
        y[o]=sum;
    }
}
static float lcg_rand(unsigned int *seed, float lo, float hi) {
    *seed=*seed*1103515245+12345; return lo+(hi-lo)*((*seed&0x7fffffff)/(float)0x7fffffff);
}

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
        unsigned int seed=42+s;
        for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }
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

static int test_dequantize(void) {
    printf("\n=== Test 3: Dequantize Round-Trip ===\n");
    int od=1280, id=896;
    size_t total=(size_t)od*id;
    uint16_t *W=(uint16_t*)malloc(total*2);
    float *Wf=(float*)malloc(total*4), *Wd=(float*)malloc(total*4);
    unsigned int seed=99;
    for (size_t i=0;i<total;i++) { float v=lcg_rand(&seed,-1,1); uint32_t u; memcpy(&u,&v,4); W[i]=(uint16_t)(u>>16); }
    for (size_t i=0;i<total;i++) { uint32_t u=((uint32_t)W[i])<<16; memcpy(&Wf[i],&u,4); }
    ds_int4_block_t blk;
    ds_int4_quantize_bf16(&blk, W, od, id);
    ds_int4_dequantize_f32(Wd, &blk);
    float rel=relative_error_f(Wf,Wd,(int)total);
    printf("  rel_err=%.6f %s\n", rel, rel<0.02f?"PASS":"FAIL");
    ds_int4_block_free(&blk);
    free(W); free(Wf); free(Wd);
    return rel < 0.02f;
}

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
    ds_int4_matvec_rows(ys, x, &blk, 0, 896);
    float mad=max_abs_diff_f(yf, ys, 896);
    printf("  full vs rows: max_diff=%.10f %s\n", mad, mad<1e-5f?"PASS":"FAIL");
    ds_int4_block_free(&blk);
    free(W); free(x); free(yf); free(ys);
    return mad < 1e-5f;
}

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
