/*
 * ds_metal.m - Metal GPU wrapper for DeepSeek-OCR
 * Zero-copy weight sharing via unified memory (Apple Silicon).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds_metal.h"
#include "ds_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const void *ptr; id<MTLBuffer> buf; size_t sz; } wbuf_t;
#define MAX_WBUFS 4096
typedef struct { float value; int index; } argmax_pair_t;

struct ds_metal_ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLLibrary> lib;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *ps;
    wbuf_t wb[MAX_WBUFS]; int nwb;
    id<MTLBuffer> x_b;  int x_n;
    id<MTLBuffer> y_b;  int y_n;
    id<MTLBuffer> gu_b; int gu_n;
    id<MTLBuffer> sw_b; int sw_n;
    id<MTLBuffer> eo_b; int eo_n;
    id<MTLBuffer> cb_b; int cb_n;
    id<MTLBuffer> ar_b; int ar_n;
    id<MTLBuffer> q_b;  int q_n;
    id<MTLBuffer> k_b;  int kv_n;
    id<MTLBuffer> v_b;  int v_n2;
    int ndispatch;
};

static id<MTLBuffer> mkbuf(id<MTLDevice> d, int n) {
    return [d newBufferWithLength:(size_t)n*sizeof(float)
                          options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> ensure_buf(id<MTLDevice> d, id<MTLBuffer> b, int *cur, int need) {
    if (b && *cur >= need) return b;
    id<MTLBuffer> nb = mkbuf(d, need);
    if (nb) *cur = need;
    return nb;
}

static id<MTLBuffer> get_wbuf(ds_metal_ctx_t *c, const void *p, size_t bytes) {
    for (int i = 0; i < c->nwb; i++)
        if (c->wb[i].ptr == p) return c->wb[i].buf;
    if (c->nwb >= MAX_WBUFS) return nil;
    id<MTLBuffer> b = [c->dev newBufferWithBytesNoCopy:(void*)p length:bytes
            options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) b = [c->dev newBufferWithBytes:p length:bytes
                                    options:MTLResourceStorageModeShared];
    if (!b) return nil;
    c->wb[c->nwb] = (wbuf_t){p, b, bytes};
    c->nwb++;
    return b;
}

static id<MTLComputePipelineState> get_ps(ds_metal_ctx_t *c, const char *name) {
    NSString *ns = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> p = c->ps[ns];
    if (p) return p;
    if (!c->lib) return nil;
    id<MTLFunction> fn = [c->lib newFunctionWithName:ns];
    if (!fn) return nil;
    NSError *e = nil;
    p = [c->dev newComputePipelineStateWithFunction:fn error:&e];
    if (p) c->ps[ns] = p;
    return p;
}

static void dmatvec(ds_metal_ctx_t *c, id<MTLBuffer> yb, id<MTLBuffer> xb,
                     id<MTLBuffer> wb, int in_d, int out_d) {
    id<MTLComputePipelineState> ps = get_ps(c, "bf16_matvec");
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:yb offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:wb offset:0 atIndex:2];
    int z=0;
    [enc setBytes:&in_d  length:4 atIndex:3];
    [enc setBytes:&out_d length:4 atIndex:4];
    [enc setBytes:&z     length:4 atIndex:5];
    NSUInteger sw = ps.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake(out_d,1,1)
          threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

static void dswiglu(ds_metal_ctx_t *c, id<MTLBuffer> ob, id<MTLBuffer> gub, int inter) {
    id<MTLComputePipelineState> ps = get_ps(c, "swiglu");
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:ob  offset:0 atIndex:0];
    [enc setBuffer:gub offset:0 atIndex:1];
    [enc setBytes:&inter length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((inter+63)/64,1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

static void dscale_add(ds_metal_ctx_t *c, id<MTLBuffer> dst, id<MTLBuffer> src,
                        float scale, int n) {
    id<MTLComputePipelineState> ps = get_ps(c, "scale_add");
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBytes:&scale length:4 atIndex:2];
    [enc setBytes:&n     length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n+63)/64,1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

static void dzero(ds_metal_ctx_t *c, id<MTLBuffer> buf, int n) {
    id<MTLComputePipelineState> ps = get_ps(c, "zero_fill");
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&n length:4 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((n+63)/64,1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

ds_metal_ctx_t *ds_metal_init(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return NULL;
    ds_metal_ctx_t *c = (ds_metal_ctx_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->dev = dev;
    c->q = [dev newCommandQueue];
    if (!c->q) { free(c); return NULL; }
    c->ps = [[NSMutableDictionary alloc] init];

    NSString *exe = [[NSBundle mainBundle] executablePath];
    NSString *dir = [exe stringByDeletingLastPathComponent];
    NSString *ml = [dir stringByAppendingPathComponent:@"ds_metal_shaders.metallib"];
    NSError *err = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:ml])
        c->lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:ml] error:&err];

    if (!c->lib) {
        NSString *src = [dir stringByAppendingPathComponent:@"ds_metal_shaders.metal"];
        NSString *code = [NSString stringWithContentsOfFile:src
                                                   encoding:NSUTF8StringEncoding error:nil];
        if (!code) code = [NSString stringWithContentsOfFile:@"ds_metal_shaders.metal"
                                                    encoding:NSUTF8StringEncoding error:nil];
        if (code) {
            MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
            opt.fastMathEnabled = YES;
            c->lib = [dev newLibraryWithSource:code options:opt error:&err];
            if (!c->lib && err) {
                fprintf(stderr, "Metal: compile error: %s\n",
                        [[err localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "Metal: could not find ds_metal_shaders.metal\n");
        }
    }
    if (!c->lib) {
        if (ds_verbose >= 1) fprintf(stderr, "Metal: no shaders, GPU disabled\n");
        ds_metal_free(c); return NULL;
    }

    const char *names[] = {"bf16_matvec","bf16_matvec_bias","bf16_matvec_qkv",
        "swiglu","expert_combine","add_inplace","argmax_partial","zero_fill","scale_add",NULL};
    for (int i = 0; names[i]; i++) {
        if (!get_ps(c, names[i])) {
            fprintf(stderr, "Metal: kernel '%s' missing\n", names[i]);
            ds_metal_free(c); return NULL;
        }
    }
    if (ds_verbose >= 1)
        fprintf(stderr, "Metal: initialized on %s\n", [[dev name] UTF8String]);
    return c;
}

void ds_metal_free(ds_metal_ctx_t *c) {
    if (!c) return;
    c->ps=nil; c->dev=nil; c->q=nil; c->lib=nil;
    c->x_b=nil; c->y_b=nil; c->gu_b=nil; c->sw_b=nil;
    c->eo_b=nil; c->cb_b=nil; c->ar_b=nil;
    c->q_b=nil; c->k_b=nil; c->v_b=nil;
    for (int i = 0; i < c->nwb; i++) c->wb[i].buf = nil;
    free(c);
}

int ds_metal_is_available(const ds_metal_ctx_t *c) {
    return c && c->dev && c->q && c->lib;
}

int ds_metal_register_bf16(ds_metal_ctx_t *c, const uint16_t *d, size_t n) {
    return (c && d && get_wbuf(c, d, n*2)) ? 0 : -1;
}
int ds_metal_register_f32(ds_metal_ctx_t *c, const float *d, size_t n) {
    return (c && d && get_wbuf(c, d, n*4)) ? 0 : -1;
}

void ds_metal_matvec_bf16(ds_metal_ctx_t *c, float *y, const float *x,
                            const uint16_t *W, const float *bias,
                            int in_dim, int out_dim) {
    if (!c||!y||!x||!W) return;
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, in_dim);
    c->y_b = ensure_buf(c->dev, c->y_b, &c->y_n, out_dim);
    memcpy(c->x_b.contents, x, in_dim*sizeof(float));
    id<MTLBuffer> Wb = get_wbuf(c, W, (size_t)out_dim*in_dim*2);
    dmatvec(c, c->y_b, c->x_b, Wb, in_dim, out_dim);
    memcpy(y, c->y_b.contents, out_dim*sizeof(float));
}

void ds_metal_matvec_bf16_qkv(ds_metal_ctx_t *c, float *q, float *k, float *v,
                                const float *x,
                                const uint16_t *Wq, const uint16_t *Wk,
                                const uint16_t *Wv,
                                int in_dim, int q_dim, int kv_dim) {
    if (!c||!q||!k||!v||!x) return;
    id<MTLComputePipelineState> ps = get_ps(c, "bf16_matvec_qkv");
    if (!ps) return;
    int total = q_dim + kv_dim*2;
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, in_dim);
    c->q_b = ensure_buf(c->dev, c->q_b, &c->q_n, q_dim);
    c->k_b = ensure_buf(c->dev, c->k_b, &c->kv_n, kv_dim);
    c->v_b = ensure_buf(c->dev, c->v_b, &c->v_n2, kv_dim);
    memcpy(c->x_b.contents, x, in_dim*sizeof(float));
    id<MTLBuffer> Wq_b = get_wbuf(c, Wq, (size_t)q_dim*in_dim*2);
    id<MTLBuffer> Wk_b = get_wbuf(c, Wk, (size_t)kv_dim*in_dim*2);
    id<MTLBuffer> Wv_b = get_wbuf(c, Wv, (size_t)kv_dim*in_dim*2);

    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:c->q_b offset:0 atIndex:0];
    [enc setBuffer:c->k_b offset:0 atIndex:1];
    [enc setBuffer:c->v_b offset:0 atIndex:2];
    [enc setBuffer:c->x_b offset:0 atIndex:3];
    [enc setBuffer:Wq_b offset:0 atIndex:4];
    [enc setBuffer:Wk_b offset:0 atIndex:5];
    [enc setBuffer:Wv_b offset:0 atIndex:6];
    [enc setBytes:&in_dim length:4 atIndex:7];
    [enc setBytes:&q_dim  length:4 atIndex:8];
    [enc setBytes:&kv_dim length:4 atIndex:9];
    NSUInteger sw = ps.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake(total,1,1)
          threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

    memcpy(q, c->q_b.contents, q_dim*sizeof(float));
    memcpy(k, c->k_b.contents, kv_dim*sizeof(float));
    memcpy(v, c->v_b.contents, kv_dim*sizeof(float));
    c->ndispatch++;
}

void ds_metal_moe_experts(ds_metal_ctx_t *c, const float *x,
                           const uint16_t **gu_ptrs, const uint16_t **dn_ptrs,
                           const float *wts, int top_k,
                           int hidden, int inter, float *output) {
    if (!c||!x||!output) return;
    int gu_dim = 2*inter;
    c->x_b  = ensure_buf(c->dev, c->x_b,  &c->x_n,  hidden);
    c->gu_b = ensure_buf(c->dev, c->gu_b, &c->gu_n, gu_dim);
    c->sw_b = ensure_buf(c->dev, c->sw_b, &c->sw_n, inter);
    c->eo_b = ensure_buf(c->dev, c->eo_b, &c->eo_n, hidden);
    c->cb_b = ensure_buf(c->dev, c->cb_b, &c->cb_n, hidden);

    memcpy(c->x_b.contents, x, hidden*sizeof(float));
    dzero(c, c->cb_b, hidden);

    for (int k = 0; k < top_k; k++) {
        id<MTLBuffer> Wgu = get_wbuf(c, gu_ptrs[k], (size_t)gu_dim*hidden*2);
        dmatvec(c, c->gu_b, c->x_b, Wgu, hidden, gu_dim);
        dswiglu(c, c->sw_b, c->gu_b, inter);
        memcpy(c->x_b.contents, c->sw_b.contents, inter*sizeof(float));
        id<MTLBuffer> Wdn = get_wbuf(c, dn_ptrs[k], (size_t)hidden*inter*2);
        dmatvec(c, c->eo_b, c->x_b, Wdn, inter, hidden);
        dscale_add(c, c->cb_b, c->eo_b, wts[k], hidden);
        if (k+1 < top_k) memcpy(c->x_b.contents, x, hidden*sizeof(float));
    }
    memcpy(output, c->cb_b.contents, hidden*sizeof(float));
}

void ds_metal_shared_experts(ds_metal_ctx_t *c, const float *x,
                              const uint16_t *gu_bf16, const uint16_t *dn_bf16,
                              int hidden, int shared_inter, float *output) {
    if (!c||!x||!output) return;
    int gu_dim = 2*shared_inter;
    c->x_b  = ensure_buf(c->dev, c->x_b,  &c->x_n,  hidden);
    c->gu_b = ensure_buf(c->dev, c->gu_b, &c->gu_n, gu_dim);
    c->sw_b = ensure_buf(c->dev, c->sw_b, &c->sw_n, shared_inter);
    c->eo_b = ensure_buf(c->dev, c->eo_b, &c->eo_n, hidden);

    memcpy(c->x_b.contents, x, hidden*sizeof(float));
    id<MTLBuffer> Wgu = get_wbuf(c, gu_bf16, (size_t)gu_dim*hidden*2);
    dmatvec(c, c->gu_b, c->x_b, Wgu, hidden, gu_dim);
    dswiglu(c, c->sw_b, c->gu_b, shared_inter);
    memcpy(c->x_b.contents, c->sw_b.contents, shared_inter*sizeof(float));
    id<MTLBuffer> Wdn = get_wbuf(c, dn_bf16, (size_t)hidden*shared_inter*2);
    dmatvec(c, c->eo_b, c->x_b, Wdn, shared_inter, hidden);
    float *r = (float*)c->eo_b.contents;
    for (int i = 0; i < hidden; i++) output[i] += r[i];
}

int ds_metal_lm_head_argmax(ds_metal_ctx_t *c, const float *x,
                              const uint16_t *W, int hidden, int vocab) {
    if (!c||!x||!W) return -1;
    id<MTLComputePipelineState> ps = get_ps(c, "argmax_partial");
    if (!ps) return -1;
    int rpc = (int)ps.threadExecutionWidth;
    int nc = (vocab + rpc - 1) / rpc;
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, hidden);
    if (!c->ar_b || c->ar_n < nc) {
        c->ar_b = [c->dev newBufferWithLength:(size_t)nc*sizeof(argmax_pair_t)
                                       options:MTLResourceStorageModeShared];
        c->ar_n = nc;
    }
    memcpy(c->x_b.contents, x, hidden*sizeof(float));
    id<MTLBuffer> Wb = get_wbuf(c, W, (size_t)vocab*hidden*2);
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:c->ar_b offset:0 atIndex:0];
    [enc setBuffer:c->x_b  offset:0 atIndex:1];
    [enc setBuffer:Wb      offset:0 atIndex:2];
    [enc setBytes:&hidden length:4 atIndex:3];
    [enc setBytes:&vocab  length:4 atIndex:4];
    [enc setBytes:&rpc    length:4 atIndex:5];
    NSUInteger sw = ps.threadExecutionWidth;
    NSUInteger tgs = sw * 4;
    NSUInteger ntg = (nc + 3) / 4;
    [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1)
          threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    argmax_pair_t *res = (argmax_pair_t*)c->ar_b.contents;
    float bv = -1e30f; int bi = 0;
    for (int i = 0; i < nc; i++)
        if (res[i].index >= 0 && res[i].value > bv) { bv = res[i].value; bi = res[i].index; }
    c->ndispatch++;
    return bi;
}
