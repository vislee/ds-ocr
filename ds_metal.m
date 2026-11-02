/*
 * ds_metal.m - Metal GPU wrapper for DeepSeek-OCR
 * Zero-copy weight sharing via unified memory (Apple Silicon).
 * ds_metal.m — Apple Metal GPU 后端：把 decode 热路径的 MoE 专家 / 共享专家 /
 * LM head matvec 卸载到 GPU，统一内存下零拷贝共享 mmap 权重。
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】CPU SIMD 内核（ds_kernels*）之上的"可选加速层"
 * ─────────────────────────────────────────────────────────────────────
 * decode 是带宽受限负载（docs/09-量化与部署篇.md §1 roofline：每步只有
 * ~1.15 GFLOP 却要读数 GB 权重）。Apple Silicon 的 unified memory 让
 * CPU 与 GPU 读同一物理页——mmap 的 BF16 权重不拷贝就能被 GPU kernel
 * 直接读。所以本文件的核心工作是三件事：
 *   1. 零拷贝注册：newBufferWithBytesNoCopy 把已有主机指针包成
 *      MTLBuffer（不满足对齐等要求时才退化为 newBufferWithBytes 拷贝）；
 *   2. kernel 派发：查 pipeline state → 建 command buffer → 编码 kernel
 *      参数 → commit + waitUntilCompleted（同步等 GPU 算完再回 CPU）；
 *   3. 回退：初始化任何一步失败都返回 NULL，调用方（ds_moe_decoder.c
 *      等）逐算子回退 CPU 内核，数值语义不变。
 *
 * 【kernel 一览】（shader 源码见 ds_metal_shaders.metal）
 *   bf16_matvec / _bias / _qkv  BF16 权重 × F32 向量（每个 threadgroup 算一行，
 *                               行内线程按 SIMD 宽度分列，simd_shuffle_down 归约）
 *   swiglu / swiglu_offset      SiLU(gate)*up 激活（offset 版服务 MoE 分块输出）
 *   scale_add / _offset         dst += scale*src（top-k 专家输出加权合并）
 *   zero_fill                   清零 buffer
 *   argmax_partial              LM head 分块流式 argmax（docs/05-性能优化实战篇.md §4.1）
 *
 * 【为什么整步只用一个 command buffer】(docs/09 §7.1：本工程的
 * "CUDA Graph" 对应物) 一个 command buffer 可以编码任意多道 kernel，
 * GPU 按序执行、CPU 只同步一次——把逐 kernel 提交的启动开销摊平。
 * ds_metal_moe_experts / ds_metal_shared_experts 都是这个模式。
 *
 * 【派发形状约定】统一用 dispatchThreadgroups:threadsPerThreadgroup: 这对
 * 旧式 API：threadgroup 数 = 输出行数（matvec）或元素数（逐元素 kernel），
 * threadgroup 大小 = threadExecutionWidth（Apple GPU 上为 32，与 shader
 * 内 SIMD_W=32 的归约配套）或 64（逐元素 kernel 的固定块）；越界由
 * shader 内的 row/gid 边界检查兜住。
 *
 * 参考文档: docs/09-量化与部署篇.md §1/§5.3/§7.1；
 *          docs/05-性能优化实战篇.md §4.1/§7；docs/08-解码策略与后处理篇.md
 * ═══════════════════════════════════════════════════════════════════════
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds_metal.h"
#include "ds_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* wbuf_t — 零拷贝权重注册表的一条：主机指针 → 已包好的 Metal buffer。
 * ptr 用作"这个权重注册过没有"的键——mmap 地址在整个推理期间稳定，
 * 同一权重多次派发只需查表，不必重复包装。 */
typedef struct { const void *ptr; id<MTLBuffer> buf; size_t sz; } wbuf_t;
#define MAX_WBUFS 4096  /* 注册表容量上限：模型权重块数量级（几十~几百），留足余量 */

/* argmax_pair_t — argmax_partial kernel 每个 chunk 写回的局部最优（值+下标），
 * 与 ds_metal_shaders.metal 里的 struct argmax_pair 内存布局保持一致 */
typedef struct { float value; int index; } argmax_pair_t;

/* ds_metal_ctx — Metal 后端全部状态（对外不透明，接口见 ds_metal.h） */
struct ds_metal_ctx {
    id<MTLDevice> dev;      /* GPU 设备句柄（MTLCreateSystemDefaultDevice 取默认 GPU） */
    id<MTLCommandQueue> q;  /* 命令队列：所有 command buffer 都提交到它 */
    id<MTLLibrary> lib;     /* 着色器库（预编译 .metallib 或运行时编译的 .metal 源码） */
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *ps;
                            /* kernel 名 → 编译好的管线状态缓存（get_ps 查/建） */
    wbuf_t wb[MAX_WBUFS]; int nwb;  /* 零拷贝权重注册表 + 已用条数 */

    /* ── 可复用的活动 buffer（ensure_buf 按需扩容，跨调用复用免反复分配）──
     * 命名约定：*_b 是 MTLBuffer，配对的 *_n 记当前容量（float 元素数）。
     * x/y 是通用 matvec 的输入/输出；gu/sw/eo/cb 是 MoE 中间结果；
     * ar 是 argmax 的 chunk 结果表；q/k/v 是 QKV 融合投影输出。 */
    id<MTLBuffer> x_b;  int x_n;   /* 通用输入向量 x[hidden] */
    id<MTLBuffer> y_b;  int y_n;   /* 通用 matvec 输出 y[out_dim] */
    id<MTLBuffer> gu_b; int gu_n;  /* gate_up 融合投影输出 [top_k*2*inter] */
    id<MTLBuffer> sw_b; int sw_n;  /* SwiGLU 激活输出 [top_k*inter] */
    id<MTLBuffer> eo_b; int eo_n;  /* 每专家 down 投影输出 [top_k*hidden] */
    id<MTLBuffer> cb_b; int cb_n;  /* combined：top-k 专家加权合并结果 [hidden] */
    id<MTLBuffer> ar_b; int ar_n;  /* argmax 每 chunk 局部最优表 [nc] */
    id<MTLBuffer> q_b;  int q_n;   /* QKV 融合：Q 输出 */
    id<MTLBuffer> k_b;  int kv_n;  /* QKV 融合：K 输出（容量字段沿用 kv_n 命名） */
    id<MTLBuffer> v_b;  int v_n2;  /* QKV 融合：V 输出（容量字段沿用 v_n2 命名） */

    /* Expert block for offset-based weight access */
    /* 专家权重连续块（docs/09 §5.3）：整块只建一个 Metal buffer，各专家
     * 权重靠字节偏移访问——省掉 128+ 个小 buffer 的创建/注册开销 */
    id<MTLBuffer> expert_block_bf16;
    size_t expert_block_size;      /* 块字节数（区间判断用） */
    const void *expert_block_host; /* 块的主机首地址（判断指针是否落在块内） */

    int ndispatch;  /* 已完成派发的 kernel 提交次数（每 commit 一次 +1，统计/调试用） */
};

/* mkbuf — 分配 n 个 float 的 Shared 存储模式 buffer。
 * MTLResourceStorageModeShared：CPU 与 GPU 直接读写同一块内存
 * （unified memory 的关键）——x/y 等活动 buffer 靠它做 CPU→GPU→CPU 搬运，
 * 搬运用 memcpy 进出 .contents 即可，无需显式上传/下载命令。 */
static id<MTLBuffer> mkbuf(id<MTLDevice> d, int n) {
    return [d newBufferWithLength:(size_t)n*sizeof(float)
                          options:MTLResourceStorageModeShared];
}

/* ensure_buf — 活动 buffer 的按需扩容复用。
 * 已有 buffer 容量够（*cur >= need）就直接返回（decode 稳态每步形状
 * 相同，几乎全走这条零分配路径）；不够才重新分配（旧 buffer 由 ARC
 * 自动释放）并记录新容量。 */
static id<MTLBuffer> ensure_buf(id<MTLDevice> d, id<MTLBuffer> b, int *cur, int need) {
    if (b && *cur >= need) return b;    /* 命中：现有 buffer 容量足够，直接复用 */
    id<MTLBuffer> nb = mkbuf(d, need);  /* 首次使用或形状变大：新分配 */
    if (nb) *cur = need;                /* 记录新容量，供后续调用判断 */
    return nb;                          /* 分配失败返回 nil，调用方自行判空 */
}

/* get_wbuf — 取权重指针对应的 Metal buffer（零拷贝优先，失败退拷贝）。
 * 步骤:
 *   1. 线性查注册表：同一 mmap 指针只包装一次，后续 O(nwb) 查表命中；
 *   2. newBufferWithBytesNoCopy 把 [p, p+bytes) 这段已有内存直接包成
 *      Shared buffer —— 不发生任何复制，GPU 读的就是 mmap 的物理页；
 *   3. NoCopy 失败（指针未按页对齐等硬件约束）退化为 newBufferWithBytes
 *      拷贝一份——正确性不变，只是多占一份内存；
 *   4. 注册表满（>= MAX_WBUFS）返回 nil，调用方应回退 CPU 路径。
 * 权重指针来自 mmap 的 safetensors，进程生命周期内地址不变，可安全作键。 */
static id<MTLBuffer> get_wbuf(ds_metal_ctx_t *c, const void *p, size_t bytes) {
    for (int i = 0; i < c->nwb; i++)          /* 先查注册表，命中即复用 */
        if (c->wb[i].ptr == p) return c->wb[i].buf;
    if (c->nwb >= MAX_WBUFS) return nil;      /* 注册表满：放弃（理论到不了） */
    /* 首选零拷贝：包装已有内存（deallocator:nil = Metal 不拥有这段内存，
     * 释放仍归 safetensors 的 mmap 管，因此这里只 nil 掉引用、不 free 内存） */
    id<MTLBuffer> b = [c->dev newBufferWithBytesNoCopy:(void*)p length:bytes
            options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) b = [c->dev newBufferWithBytes:p length:bytes
                                    options:MTLResourceStorageModeShared]; /* 退化：拷贝一份 */
    if (!b) return nil;                       /* 两种方式都失败：GPU 路径放弃 */
    c->wb[c->nwb] = (wbuf_t){p, b, bytes};    /* 登记到注册表，下次直接复用 */
    c->nwb++;
    return b;
}

/* Get weight buffer with byte offset for setBuffer:offset:atIndex:
 * If the pointer falls within the expert block, return the block buffer
 * with appropriate offset. Otherwise, create individual buffer. */
/* get_wbuf_offset — 带偏移版权重 buffer 查询：服务"专家连续块"（docs/09 §5.3）。
 * 64 个路由专家的 gate_up/down 权重如果逐个注册，要建 128 个 Metal buffer；
 * 把整块连续权重一次注册后，这里按指针区间判断专家权重落在块内哪个
 * 位置，返回块 buffer + 块内字节偏移，派发时用 setBuffer:offset: 让
 * kernel 从正确位置读起——大 buffer 复用、零新分配。
 * 不在块内的指针走 get_wbuf 常规注册（offset 恒 0）。 */
static id<MTLBuffer> get_wbuf_offset(ds_metal_ctx_t *c, const void *p, size_t bytes,
                                       NSUInteger *out_offset) {
    if (c->expert_block_host && c->expert_block_bf16) {   /* 块已注册才可能命中 */
        uintptr_t host = (uintptr_t)p;                    /* 指针转整数做区间比较 */
        uintptr_t block_start = (uintptr_t)c->expert_block_host;
        uintptr_t block_end = block_start + c->expert_block_size;
        if (host >= block_start && host + bytes <= block_end) {  /* [p, p+bytes) 完整落块内 */
            *out_offset = (NSUInteger)(host - block_start);      /* 块内字节偏移 */
            return c->expert_block_bf16;                         /* 复用块 buffer */
        }
    }
    *out_offset = 0;                 /* 常规路径：无偏移 */
    return get_wbuf(c, p, bytes);    /* 退回普通零拷贝注册 */
}

/* get_ps — 按名字取（并缓存）compute pipeline state。
 * pipeline state = 编译好的 kernel 可执行体 + 执行配置，创建较贵；
 * 首次 newFunctionWithName + newComputePipelineStateWithFunction 后存入
 * 字典 ps，之后每次派发只剩一次字典查找。返回 nil 表示 kernel 缺失。 */
static id<MTLComputePipelineState> get_ps(ds_metal_ctx_t *c, const char *name) {
    NSString *ns = [NSString stringWithUTF8String:name];   /* C 字符串 → NSString 字典键 */
    id<MTLComputePipelineState> p = c->ps[ns];             /* 先查缓存 */
    if (p) return p;                                       /* 命中：直接返回 */
    if (!c->lib) return nil;                               /* 没着色器库无从查找 */
    id<MTLFunction> fn = [c->lib newFunctionWithName:ns];  /* 按名字取 kernel 函数 */
    if (!fn) return nil;                                   /* 库里没有这个 kernel */
    NSError *e = nil;
    p = [c->dev newComputePipelineStateWithFunction:fn error:&e]; /* 编译成管线（一次性开销） */
    if (p) c->ps[ns] = p;                                  /* 成功才缓存，失败下次重试 */
    return p;
}

/* dmatvec — 派发单个 BF16 matvec（kernel bf16_matvec：y = W·x）。
 * 派发形状: grid = out_d 个 threadgroup（每个输出行一个 TG），每个 TG
 * = threadExecutionWidth 个线程（Apple GPU 上为 32，与 shader 内
 * SIMD_W=32 的 simd_shuffle_down 行内归约配套）。
 * 同步语义: waitUntilCompleted 阻塞到 GPU 算完——decode 每步都立刻要
 * 用结果，此处不做异步流水（对照 ds_metal_moe_experts 的"一次提交"）。 */
static void dmatvec(ds_metal_ctx_t *c, id<MTLBuffer> yb, id<MTLBuffer> xb,
                     id<MTLBuffer> wb, int in_d, int out_d) {
    id<MTLComputePipelineState> ps = get_ps(c, "bf16_matvec"); /* 查缓存好的管线 */
    if (!ps) return;                              /* kernel 缺失：静默跳过，CPU 路径兜底 */
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];               /* 新建命令缓冲 */
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; /* 编码器：往里写 kernel 调用 */
    [enc setComputePipelineState:ps];             /* 本 pass 用哪个 kernel */
    [enc setBuffer:yb offset:0 atIndex:0];        /* buffer(0) = 输出 y */
    [enc setBuffer:xb offset:0 atIndex:1];        /* buffer(1) = 输入 x */
    [enc setBuffer:wb offset:0 atIndex:2];        /* buffer(2) = BF16 权重 W */
    int z=0;                                      /* y_offset = 0：结果从 y[0] 写起 */
    [enc setBytes:&in_d  length:4 atIndex:3];     /* buffer(3) = in_dim（小标量直接内联进命令流） */
    [enc setBytes:&out_d length:4 atIndex:4];     /* buffer(4) = out_dim */
    [enc setBytes:&z     length:4 atIndex:5];     /* buffer(5) = y_offset */
    NSUInteger sw = ps.threadExecutionWidth;      /* SIMD 宽度（=32）：一个 TG 的线程数 */
    [enc dispatchThreadgroups:MTLSizeMake(out_d,1,1)   /* out_d 个 TG，一 TG 算一行 */
          threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; /* 提交 + 同步等 GPU 完成 */
    c->ndispatch++;                               /* 统计：完成一次 GPU 派发 */
}

/* Named-kernel dispatch helpers. ds_metal_moe_experts currently inlines its
 * own encoder passes, so these may be unused in a given build — keep them as
 * the wrapper API for their registered kernels. */
/* dswiglu — 派发 SwiGLU 激活（kernel swiglu）：out[i] = SiLU(gate[i])*up[i]，
 * 输入 gate_up 为融合排布（前 inter 个是 gate，后 inter 个是 up）。
 * 派发形状: ceil(inter/64) 个 TG × 64 线程（逐元素 kernel 的固定块）。
 * 标记 unused：ds_metal_moe_experts 走 swiglu_offset 内联编码，本函数
 * 是 kernel 的独立包装，保留给单步调用/调试用。 */
static void __attribute__((unused)) dswiglu(ds_metal_ctx_t *c, id<MTLBuffer> ob, id<MTLBuffer> gub, int inter) {
    id<MTLComputePipelineState> ps = get_ps(c, "swiglu");   /* 查管线 */
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:ob  offset:0 atIndex:0];   /* buffer(0) = 输出 out[inter] */
    [enc setBuffer:gub offset:0 atIndex:1];   /* buffer(1) = gate_up[2*inter] */
    [enc setBytes:&inter length:4 atIndex:2]; /* buffer(2) = inter（前后半段分界） */
    [enc dispatchThreadgroups:MTLSizeMake((inter+63)/64,1,1)  /* 向上取整到 64 的倍数 */
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

/* dscale_add — 派发加权累加（kernel scale_add）：dst[i] += scale*src[i]。
 * 用途: 把单个专家的输出按路由权重合并进总输出。
 * 派发形状: ceil(n/64) 个 TG × 64 线程。
 * 标记 unused：MoE 主路径用带 offset 的 scale_add_offset 版本。 */
static void __attribute__((unused)) dscale_add(ds_metal_ctx_t *c, id<MTLBuffer> dst, id<MTLBuffer> src,
                        float scale, int n) {
    id<MTLComputePipelineState> ps = get_ps(c, "scale_add");  /* 查管线 */
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dst offset:0 atIndex:0];     /* buffer(0) = 累加目标 dst */
    [enc setBuffer:src offset:0 atIndex:1];     /* buffer(1) = 源 src */
    [enc setBytes:&scale length:4 atIndex:2];   /* buffer(2) = 路由权重 scale */
    [enc setBytes:&n     length:4 atIndex:3];   /* buffer(3) = 元素个数 n */
    [enc dispatchThreadgroups:MTLSizeMake((n+63)/64,1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

/* dzero — 派发清零（kernel zero_fill）：buf[i] = 0，n 个 float。
 * 用途: MoE 合并前清空 combined buffer（等价 CPU memset 的 GPU 版）。
 * 派发形状: ceil(n/64) 个 TG × 64 线程。
 * 标记 unused：独立包装，主路径在 ds_metal_moe_experts 内联编码 zero_fill。 */
static void __attribute__((unused)) dzero(ds_metal_ctx_t *c, id<MTLBuffer> buf, int n) {
    id<MTLComputePipelineState> ps = get_ps(c, "zero_fill");  /* 查管线 */
    if (!ps) return;
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];     /* buffer(0) = 待清零 buffer */
    [enc setBytes:&n length:4 atIndex:1];       /* buffer(1) = 元素个数 */
    [enc dispatchThreadgroups:MTLSizeMake((n+63)/64,1,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    c->ndispatch++;
}

/* ds_metal_init — 初始化 Metal 后端（设备/队列/着色器库/全部管线）。
 * 步骤:
 *   1. 取系统默认 GPU；没有（黑苹果/虚拟机/无 Metal）→ 返回 NULL；
 *   2. 建命令队列 + 管线缓存字典；
 *   3. 加载着色器：优先可执行文件同目录的 ds_metal_shaders.metallib
 *      （预编译产物，加载快）；否则现场编译同目录（或 cwd）的
 *      ds_metal_shaders.metal 源码——macOS 15+ 走 MTLMathModeFast，
 *      旧系统用 fastMathEnabled（二者都是放开浮点变换换速度）；
 *   4. 逐个确认 11 个 kernel 都能建出 pipeline state，缺任何一个即
 *      整体失败——宁可初始化时回退 CPU，也不留运行中才发现的雷。
 * @return 成功的 context；任何一步失败返回 NULL（调用方全程走 CPU）。
 * 性能语境见 docs/09 §1；失败时 stderr 打印具体原因（受 ds_verbose 控制）。 */
ds_metal_ctx_t *ds_metal_init(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();  /* 取默认 GPU 设备 */
    if (!dev) return NULL;                               /* 无 Metal 设备：直接放弃 */
    ds_metal_ctx_t *c = (ds_metal_ctx_t *)calloc(1, sizeof(*c)); /* 上下文清零分配 */
    if (!c) return NULL;
    c->dev = dev;
    c->q = [dev newCommandQueue];                        /* 命令队列：后续所有提交走它 */
    if (!c->q) { free(c); return NULL; }
    c->ps = [[NSMutableDictionary alloc] init];          /* 管线缓存字典（get_ps 填充） */

    /* 加载着色器第一步：找预编译的 .metallib（与可执行文件同目录） */
    NSString *exe = [[NSBundle mainBundle] executablePath];
    NSString *dir = [exe stringByDeletingLastPathComponent];
    NSString *ml = [dir stringByAppendingPathComponent:@"ds_metal_shaders.metallib"];
    NSError *err = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:ml])
        c->lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:ml] error:&err];

    if (!c->lib) {
        /* 第二步：没有 metallib 就现场编译 .metal 源码（教学部署的兜底） */
        NSString *src = [dir stringByAppendingPathComponent:@"ds_metal_shaders.metal"];
        NSString *code = [NSString stringWithContentsOfFile:src
                                                   encoding:NSUTF8StringEncoding error:nil];
        if (!code) code = [NSString stringWithContentsOfFile:@"ds_metal_shaders.metal"
                                                    encoding:NSUTF8StringEncoding error:nil]; /* 再试 cwd */
        if (code) {
            MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) {
                opt.mathMode = MTLMathModeFast;          /* 新 API：快速数学模式 */
            } else
#endif
            {
                /* fastMathEnabled is deprecated in favor of mathMode on the
                 * macOS 15 SDK, but is the only option on older runtimes. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opt.fastMathEnabled = YES;               /* 旧 API：同效，旧系统唯一选择 */
#pragma clang diagnostic pop
            }
            c->lib = [dev newLibraryWithSource:code options:opt error:&err]; /* 运行时编译 */
            if (!c->lib && err) {
                fprintf(stderr, "Metal: compile error: %s\n",
                        [[err localizedDescription] UTF8String]); /* 编译失败打印原因 */
            }
        } else {
            fprintf(stderr, "Metal: could not find ds_metal_shaders.metal\n");
        }
    }
    if (!c->lib) {                                       /* 两条路都没拿到着色器库 */
        if (ds_verbose >= 1) fprintf(stderr, "Metal: no shaders, GPU disabled\n");
        ds_metal_free(c); return NULL;                   /* 释放半成品并报告不可用 */
    }

    /* 第三步：验收清单——11 个 kernel 一个都不能少 */
    const char *names[] = {"bf16_matvec","bf16_matvec_bias","bf16_matvec_qkv",
        "swiglu","swiglu_offset","expert_combine","add_inplace",
        "argmax_partial","zero_fill","scale_add","scale_add_offset",NULL};
    for (int i = 0; names[i]; i++) {
        if (!get_ps(c, names[i])) {                      /* 缺一个即整体失败 */
            fprintf(stderr, "Metal: kernel '%s' missing\n", names[i]);
            ds_metal_free(c); return NULL;
        }
    }
    if (ds_verbose >= 1)
        fprintf(stderr, "Metal: initialized on %s\n", [[dev name] UTF8String]); /* 打印 GPU 型号 */
    return c;
}

/* ds_metal_free — 释放 Metal 资源（NULL 安全，可对半初始化的 ctx 调用）。
 * 把各 ObjC 引用置 nil 触发 ARC 回收；权重内存本身不在这里释放——
 * NoCopy 包装不拥有 mmap 内存，拷贝版随 buffer 对象一起回收。 */
void ds_metal_free(ds_metal_ctx_t *c) {
    if (!c) return;
    c->ps=nil; c->dev=nil; c->q=nil; c->lib=nil;     /* 核心对象：管线缓存/设备/队列/库 */
    c->x_b=nil; c->y_b=nil; c->gu_b=nil; c->sw_b=nil;/* 活动 buffer 逐个置 nil */
    c->eo_b=nil; c->cb_b=nil; c->ar_b=nil;
    c->q_b=nil; c->k_b=nil; c->v_b=nil;
    c->expert_block_bf16=nil;                        /* 专家块 buffer（不拥有底层 mmap） */
    for (int i = 0; i < c->nwb; i++) c->wb[i].buf = nil; /* 注册表里的权重 buffer 同理 */
    free(c);                                         /* 最后释放上下文结构体本身 */
}

/* ds_metal_is_available — 设备/队列/着色器库三要素齐备才算可用。
 * 调用方在每个 GPU 算子入口先问它，决定走 GPU 还是回退 CPU 内核。 */
int ds_metal_is_available(const ds_metal_ctx_t *c) {
    return c && c->dev && c->q && c->lib;
}

/* ds_metal_register_bf16 — 把 BF16 权重数组（mmap 的）注册成可被 GPU 读的
 * buffer（零拷贝优先，见 get_wbuf）。n 为元素个数（字节 = n*2）。
 * @return 0 成功；-1 失败（ctx/指针为空或包装失败）。 */
int ds_metal_register_bf16(ds_metal_ctx_t *c, const uint16_t *d, size_t n) {
    return (c && d && get_wbuf(c, d, n*2)) ? 0 : -1;   /* BF16 每元素 2 字节 */
}
/* ds_metal_register_f32 — F32 版注册（字节 = n*4），语义同上。 */
int ds_metal_register_f32(ds_metal_ctx_t *c, const float *d, size_t n) {
    return (c && d && get_wbuf(c, d, n*4)) ? 0 : -1;   /* F32 每元素 4 字节 */
}

/* ds_metal_register_expert_block — 注册专家权重连续块（零拷贝，docs/09 §5.3）。
 * 块内布局：64 个路由专家的 gate_up_fused + down 依序连续存放。
 * 整块只建一个 MTLBuffer；之后各专家权重经 get_wbuf_offset 以块内
 * 字节偏移访问。NoCopy 失败（对齐不满足）退化为整块拷贝一份。 */
void ds_metal_register_expert_block(ds_metal_ctx_t *c, const void *block_ptr, size_t block_bytes) {
    if (!c || !block_ptr) return;
    c->expert_block_host = block_ptr;      /* 记主机首地址：区间判断的基准 */
    c->expert_block_size = block_bytes;    /* 记块大小：越界判断用 */
    /* 首选 NoCopy 零拷贝包装；失败退化为整块拷贝 */
    c->expert_block_bf16 = [c->dev newBufferWithBytesNoCopy:(void*)block_ptr
                                                      length:block_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!c->expert_block_bf16)
        c->expert_block_bf16 = [c->dev newBufferWithBytes:block_ptr
                                                   length:block_bytes
                                                  options:MTLResourceStorageModeShared];
    if (ds_verbose >= 2 && c->expert_block_bf16)
        fprintf(stderr, "Metal: registered expert block %p (%zu bytes)\n", block_ptr, block_bytes);
}

/* ds_metal_matvec_bf16 — 通用 BF16 matvec 的 GPU 入口（y = W·x）。
 * 数据流: x 拷入 GPU → 派发 bf16_matvec → y 拷回 CPU。
 * 三段都是 Shared buffer 的 .contents memcpy（unified memory 下就是
 * 普通内存拷贝）；权重 W 经 get_wbuf 零拷贝包装（首次后走注册表）。
 * 注意: 当前走 bf16_matvec kernel，无 bias 输入——bias 被显式忽略；
 * 带 bias 的 kernel（bf16_matvec_bias）已就绪但此入口未接。 */
void ds_metal_matvec_bf16(ds_metal_ctx_t *c, float *y, const float *x,
                            const uint16_t *W, const float *bias,
                            int in_dim, int out_dim) {
    (void)bias;  /* bf16_matvec kernel 无 bias 输入；带 bias 请用 bf16_matvec_bias 路径 */
    if (!c||!y||!x||!W) return;
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, in_dim);    /* 输入 buffer 复用/扩容 */
    c->y_b = ensure_buf(c->dev, c->y_b, &c->y_n, out_dim);   /* 输出 buffer 复用/扩容 */
    memcpy(c->x_b.contents, x, in_dim*sizeof(float));        /* CPU→GPU（同一内存，纯拷贝） */
    id<MTLBuffer> Wb = get_wbuf(c, W, (size_t)out_dim*in_dim*2); /* 权重零拷贝注册 */
    dmatvec(c, c->y_b, c->x_b, Wb, in_dim, out_dim);         /* GPU 上算 y = W·x */
    memcpy(y, c->y_b.contents, out_dim*sizeof(float));       /* GPU→CPU 拷回结果 */
}

/* ds_metal_matvec_bf16_qkv — 融合 QKV 投影：q=Wq·x, k=Wk·x, v=Wv·x。
 * 一次派发算完三个投影（kernel bf16_matvec_qkv：总行数 q_dim+2*kv_dim，
 * shader 内按行号分段选权重矩阵和输出 buffer），省两次 kernel 启动
 * 和一次 x 的重复拷贝——注意力投影是 decode 每步的必经之路。 */
void ds_metal_matvec_bf16_qkv(ds_metal_ctx_t *c, float *q, float *k, float *v,
                                const float *x,
                                const uint16_t *Wq, const uint16_t *Wk,
                                const uint16_t *Wv,
                                int in_dim, int q_dim, int kv_dim) {
    if (!c||!q||!k||!v||!x) return;
    id<MTLComputePipelineState> ps = get_ps(c, "bf16_matvec_qkv"); /* 查融合 kernel 管线 */
    if (!ps) return;
    int total = q_dim + kv_dim*2;                 /* 输出总行数 = Q + K + V */
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, in_dim);  /* 共享输入 x */
    c->q_b = ensure_buf(c->dev, c->q_b, &c->q_n, q_dim);   /* Q 输出 buffer */
    c->k_b = ensure_buf(c->dev, c->k_b, &c->kv_n, kv_dim); /* K 输出 buffer */
    c->v_b = ensure_buf(c->dev, c->v_b, &c->v_n2, kv_dim); /* V 输出 buffer */
    memcpy(c->x_b.contents, x, in_dim*sizeof(float));      /* x 只拷一次，三投影共用 */
    id<MTLBuffer> Wq_b = get_wbuf(c, Wq, (size_t)q_dim*in_dim*2);  /* 三个权重矩阵 */
    id<MTLBuffer> Wk_b = get_wbuf(c, Wk, (size_t)kv_dim*in_dim*2); /* 各自零拷贝注册 */
    id<MTLBuffer> Wv_b = get_wbuf(c, Wv, (size_t)kv_dim*in_dim*2);

    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:c->q_b offset:0 atIndex:0];   /* buffer(0) = Q 输出 */
    [enc setBuffer:c->k_b offset:0 atIndex:1];   /* buffer(1) = K 输出 */
    [enc setBuffer:c->v_b offset:0 atIndex:2];   /* buffer(2) = V 输出 */
    [enc setBuffer:c->x_b offset:0 atIndex:3];   /* buffer(3) = 输入 x */
    [enc setBuffer:Wq_b offset:0 atIndex:4];     /* buffer(4) = Wq */
    [enc setBuffer:Wk_b offset:0 atIndex:5];     /* buffer(5) = Wk */
    [enc setBuffer:Wv_b offset:0 atIndex:6];     /* buffer(6) = Wv */
    [enc setBytes:&in_dim length:4 atIndex:7];   /* buffer(7) = in_dim */
    [enc setBytes:&q_dim  length:4 atIndex:8];   /* buffer(8) = q_dim（shader 按行号分段） */
    [enc setBytes:&kv_dim length:4 atIndex:9];   /* buffer(9) = kv_dim */
    NSUInteger sw = ps.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake(total,1,1)   /* total 个 TG：一行 Q/K/V 各一个 */
          threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

    memcpy(q, c->q_b.contents, q_dim*sizeof(float));   /* 三个结果一次性拷回 CPU */
    memcpy(k, c->k_b.contents, kv_dim*sizeof(float));
    memcpy(v, c->v_b.contents, kv_dim*sizeof(float));
    c->ndispatch++;
}

/* ds_metal_moe_experts — MoE 路由专家批前向（decode 第一热路径，
 * docs/05 §4：MoE MLP 占 decode 时间 74-77%，是 GPU 卸载的首要目标）。
 * 数学: output = Σ_k wts[k] · Down_k(SwiGLU(GateUp_k·x))，k=0..top_k-1
 * 实现要点: top_k 个专家的 4 类 kernel（matvec→swiglu→matvec→scale_add）
 * 全部编码进同一个 command buffer——CPU 一次 commit，GPU 按序执行，
 * 中途零同步（docs/09 §7.1 "CUDA Graph" 对应物）。
 * 权重访问: 专家指针经 get_wbuf_offset 命中专家连续块（+块内偏移）；
 * 缓冲区复用: gu/sw/eo/cb 按 top_k×形状分配一次，之后每步复用。
 * 中间布局: gu_b/sw_b/eo_b 都按专家 k 分段（第 k 段 = [k*dim, (k+1)*dim)），
 * 段内偏移靠 setBuffer:offset: 与 shader 的 *_offset 参数共同完成。 */
void ds_metal_moe_experts(ds_metal_ctx_t *c, const float *x,
                           const uint16_t **gu_ptrs, const uint16_t **dn_ptrs,
                           const float *wts, int top_k,
                           int hidden, int inter, float *output) {
    if (!c||!x||!output) return;

    /* 一次取齐 4 个 kernel 的管线；任何一个缺失就整体放弃（CPU 兜底） */
    id<MTLComputePipelineState> matvec_ps = get_ps(c, "bf16_matvec");      /* 两级投影 */
    id<MTLComputePipelineState> swiglu_ps = get_ps(c, "swiglu_offset");    /* SwiGLU（带段偏移） */
    id<MTLComputePipelineState> scale_ps  = get_ps(c, "scale_add_offset"); /* 加权合并（带段偏移） */
    id<MTLComputePipelineState> zero_ps   = get_ps(c, "zero_fill");        /* 清零 */
    if (!matvec_ps || !swiglu_ps || !scale_ps || !zero_ps) return;

    int gu_dim = 2 * inter;                     /* gate_up 融合输出 = gate + up 两段 */
    NSUInteger sw = matvec_ps.threadExecutionWidth;  /* matvec 的 TG 大小（SIMD 宽度） */

    /* Allocate GPU scratch buffers */
    /* 中间缓冲区按 top_k 段一次分配到位，decode 稳态每步零分配 */
    c->x_b  = ensure_buf(c->dev, c->x_b,  &c->x_n,  hidden);        /* 输入 x[hidden] */
    c->gu_b = ensure_buf(c->dev, c->gu_b, &c->gu_n, top_k * gu_dim);/* 各专家 gate_up 输出 */
    c->sw_b = ensure_buf(c->dev, c->sw_b, &c->sw_n, top_k * inter); /* 各专家 SwiGLU 输出 */
    c->eo_b = ensure_buf(c->dev, c->eo_b, &c->eo_n, top_k * hidden);/* 各专家 down 输出 */
    c->cb_b = ensure_buf(c->dev, c->cb_b, &c->cb_n, hidden);        /* 合并结果[hidden] */

    /* Copy input to GPU */
    memcpy(c->x_b.contents, x, hidden * sizeof(float));  /* 唯一一次 CPU→GPU 输入拷贝 */

    /* ── Single command buffer for all MoE operations ── */
    /* 之后所有 kernel 都编码进这一个 cmd：commit 之前 GPU 不会开始执行 */
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    /* Step 0: Clear combined output buffer */
    {   /* 合并 buffer 清零（scale_add 是 += 语义，必须先归零） */
        [enc setComputePipelineState:zero_ps];
        [enc setBuffer:c->cb_b offset:0 atIndex:0];
        int n = hidden;
        [enc setBytes:&n length:4 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((hidden+63)/64,1,1)  /* ceil(hidden/64) 个 TG */
              threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    }

    /* Process each expert: gate_up matvec -> swiglu -> down matvec -> scale_add
     * All in one command buffer — GPU executes sequentially, zero CPU sync. */
    for (int k = 0; k < top_k; k++) {
        /* 第 k 个专家在各 scratch buffer 里的段起点（float 元素偏移） */
        int gu_off = k * gu_dim;     /* float offset in gu_b */
        int sw_off = k * inter;      /* float offset in sw_b */
        int eo_off = k * hidden;     /* float offset in eo_b */

        /* Step 1: gate_up_fused matvec: x[hidden] -> gate_up[k*2*inter .. (k+1)*2*inter] */
        {   /* 路由专家 k 的融合投影：结果写进 gu_b 的第 k 段 */
            NSUInteger w_off = 0;    /* 权重在块 buffer 内的字节偏移（get_wbuf_offset 填） */
            size_t w_bytes = (size_t)gu_dim * hidden * 2; /* BF16 bytes */
            id<MTLBuffer> Wgu = get_wbuf_offset(c, gu_ptrs[k], w_bytes, &w_off);
            [enc setComputePipelineState:matvec_ps];
            [enc setBuffer:c->gu_b offset:gu_off*sizeof(float) atIndex:0]; /* y with offset */
            [enc setBuffer:c->x_b  offset:0 atIndex:1];                    /* x */
            [enc setBuffer:Wgu    offset:w_off atIndex:2];                  /* W with offset */
            int y_off = 0; /* y offset is handled by buffer offset above */
            [enc setBytes:&hidden length:4 atIndex:3];   /* in_dim = hidden */
            [enc setBytes:&gu_dim length:4 atIndex:4];   /* out_dim = 2*inter */
            [enc setBytes:&y_off  length:4 atIndex:5];   /* y_offset = 0（段偏移走 buffer offset） */
            [enc dispatchThreadgroups:MTLSizeMake(gu_dim,1,1)  /* gu_dim 个 TG：一 TG 算一行 */
                  threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
        }

        /* Step 2: SwiGLU: gate_up[k*2*inter] -> swiglu[k*inter] */
        {   /* 对第 k 段 gate_up 做激活，结果写 sw_b 第 k 段 */
            [enc setComputePipelineState:swiglu_ps];
            [enc setBuffer:c->sw_b offset:sw_off*sizeof(float) atIndex:0]; /* 输出段偏移 */
            [enc setBuffer:c->gu_b offset:gu_off*sizeof(float) atIndex:1]; /* 输入段偏移 */
            [enc setBytes:&inter   length:4 atIndex:2];   /* 前后半段分界 */
            [enc setBytes:&sw_off  length:4 atIndex:3]; /* out_offset in shader */
            int in_off = 0; /* input starts at gu_off, no extra offset needed */
            [enc setBytes:&in_off  length:4 atIndex:4];   /* 输入段内再偏移 = 0 */
            [enc dispatchThreadgroups:MTLSizeMake((inter+63)/64,1,1)  /* ceil(inter/64) */
                  threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        }

        /* Step 3: down matvec: swiglu[k*inter] -> expert_out[k*hidden] */
        {   /* 第 k 段 SwiGLU 结果过 down 投影，写 eo_b 第 k 段 */
            NSUInteger w_off = 0;
            size_t w_bytes = (size_t)hidden * inter * 2;
            id<MTLBuffer> Wdn = get_wbuf_offset(c, dn_ptrs[k], w_bytes, &w_off);
            [enc setComputePipelineState:matvec_ps];
            [enc setBuffer:c->eo_b offset:eo_off*sizeof(float) atIndex:0]; /* 输出段偏移 */
            [enc setBuffer:c->sw_b offset:sw_off*sizeof(float) atIndex:1]; /* 输入 = 第 k 段 */
            [enc setBuffer:Wdn    offset:w_off atIndex:2];   /* down 权重（块内偏移） */
            int y_off2 = 0;
            [enc setBytes:&inter  length:4 atIndex:3];   /* in_dim = inter */
            [enc setBytes:&hidden length:4 atIndex:4];   /* out_dim = hidden */
            [enc setBytes:&y_off2 length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(hidden,1,1)
                  threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
        }

        /* Step 4: scale_add: combined += wts[k] * expert_out[k*hidden] */
        {   /* 按路由概率加权累加进 cb_b（Step 0 已清零，+= 安全） */
            [enc setComputePipelineState:scale_ps];
            [enc setBuffer:c->cb_b offset:0 atIndex:0];                      /* 累加目标（整段） */
            [enc setBuffer:c->eo_b offset:eo_off*sizeof(float) atIndex:1];   /* 源 = 第 k 段 */
            float w = wts[k];          /* 该专家的路由权重（softmax 后） */
            int n = hidden;
            int dst = 0;               /* dst 偏移 = 0（cb_b 整体就是目标） */
            int src = 0;               /* src 偏移 = 0（段偏移走 buffer offset） */
            [enc setBytes:&w    length:4 atIndex:2];
            [enc setBytes:&n    length:4 atIndex:3];
            [enc setBytes:&dst  length:4 atIndex:4];
            [enc setBytes:&src  length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((hidden+63)/64,1,1)
                  threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        }
    }

    [enc endEncoding];         /* 编码完毕：4×top_k 道 kernel 都在队列里 */
    [cmd commit];              /* 一次性提交给 GPU */
    [cmd waitUntilCompleted];  /* 唯一一次 CPU↔GPU 同步 */

    /* Copy combined output back to host */
    memcpy(output, c->cb_b.contents, hidden * sizeof(float));  /* 结果拷回 CPU */
    c->ndispatch++;  /* 统计口径：整批算一次提交 */
}

/* ─── Shared experts (single command buffer) ─── */

/* ds_metal_shared_experts — 共享专家前向（每层恒被调用、无路由）。
 * 与单个路由专家同构（gate_up matvec → SwiGLU → down matvec），区别：
 *   1. 权重是单一矩阵（shared_inter = 2*896 = 1792，等效 2 个专家宽）；
 *   2. 三个 kernel 同样编码进一个 command buffer，一次提交一次同步；
 *   3. 结果是"累加"语义：output += shared_out——与 CPU 路径的共享
 *      专家合并方式一致，调用方先放好路由专家的合并结果再调本函数。
 * 同样走 get_wbuf_offset：共享专家权重若也在连续块内即零拷贝+偏移访问。 */
void ds_metal_shared_experts(ds_metal_ctx_t *c, const float *x,
                              const uint16_t *gu_bf16, const uint16_t *dn_bf16,
                              int hidden, int shared_inter, float *output) {
    if (!c||!x||!output) return;

    id<MTLComputePipelineState> matvec_ps = get_ps(c, "bf16_matvec"); /* 投影 kernel */
    id<MTLComputePipelineState> swiglu_ps = get_ps(c, "swiglu");      /* 无偏移版激活即可 */
    if (!matvec_ps || !swiglu_ps) return;

    int gu_dim = 2 * shared_inter;              /* gate+up 两段融合输出 */
    NSUInteger sw = matvec_ps.threadExecutionWidth;

    c->x_b  = ensure_buf(c->dev, c->x_b,  &c->x_n,  hidden);        /* 输入 */
    c->gu_b = ensure_buf(c->dev, c->gu_b, &c->gu_n, gu_dim);        /* gate_up 输出 */
    c->sw_b = ensure_buf(c->dev, c->sw_b, &c->sw_n, shared_inter);  /* SwiGLU 输出 */
    c->eo_b = ensure_buf(c->dev, c->eo_b, &c->eo_n, hidden);        /* down 输出 */

    memcpy(c->x_b.contents, x, hidden * sizeof(float));  /* CPU→GPU 拷输入 */

    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    /* gate_up matvec */
    {   /* 融合投影：x[hidden] → gate_up[2*shared_inter] */
        NSUInteger w_off = 0;    /* 块内字节偏移（专家块命中时非 0） */
        id<MTLBuffer> Wgu = get_wbuf_offset(c, gu_bf16, (size_t)gu_dim*hidden*2, &w_off);
        [enc setComputePipelineState:matvec_ps];
        [enc setBuffer:c->gu_b offset:0 atIndex:0];    /* 输出整段（无段偏移） */
        [enc setBuffer:c->x_b  offset:0 atIndex:1];
        [enc setBuffer:Wgu    offset:w_off atIndex:2]; /* 权重（可能带块内偏移） */
        int z=0;                                       /* y_offset = 0 */
        [enc setBytes:&hidden  length:4 atIndex:3];    /* in_dim */
        [enc setBytes:&gu_dim  length:4 atIndex:4];    /* out_dim */
        [enc setBytes:&z       length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(gu_dim,1,1)  /* 一行一个 TG */
              threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    }

    /* SwiGLU */
    {   /* 激活：gate_up → sw[shared_inter]，无段偏移的简单版 kernel */
        [enc setComputePipelineState:swiglu_ps];
        [enc setBuffer:c->sw_b offset:0 atIndex:0];
        [enc setBuffer:c->gu_b offset:0 atIndex:1];
        [enc setBytes:&shared_inter length:4 atIndex:2];  /* gate/up 分界 */
        [enc dispatchThreadgroups:MTLSizeMake((shared_inter+63)/64,1,1)
              threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    }

    /* down matvec — use sw_b as input */
    {   /* down 投影：sw[shared_inter] → eo[hidden] */
        NSUInteger w_off = 0;
        id<MTLBuffer> Wdn = get_wbuf_offset(c, dn_bf16, (size_t)hidden*shared_inter*2, &w_off);
        [enc setComputePipelineState:matvec_ps];
        [enc setBuffer:c->eo_b offset:0 atIndex:0];
        [enc setBuffer:c->sw_b offset:0 atIndex:1];    /* 输入换 sw_b */
        [enc setBuffer:Wdn    offset:w_off atIndex:2];
        int z=0;
        [enc setBytes:&shared_inter length:4 atIndex:3]; /* in_dim */
        [enc setBytes:&hidden       length:4 atIndex:4]; /* out_dim */
        [enc setBytes:&z            length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(hidden,1,1)
              threadsPerThreadgroup:MTLSizeMake(sw,1,1)];
    }

    [enc endEncoding];         /* 三道 kernel 一次提交 */
    [cmd commit];
    [cmd waitUntilCompleted];  /* 一次同步 */

    /* Add to output (matches CPU: output += shared_out) */
    /* 累加进调用方的 output（Shared buffer 直接读，无需先 memcpy 到临时数组） */
    float *r = (float*)c->eo_b.contents;
    for (int i = 0; i < hidden; i++) output[i] += r[i];
    c->ndispatch++;
}

/* ─── LM head argmax ─── */

/* ds_metal_lm_head_argmax — LM head 的 GPU 流式 argmax（贪心解码终站）。
 * 数学: 只需 argmax_v(x·W_v^T)，不必物化 vocab 维 logits（129280 维 F32
 * 约 0.5 MB 的写回完全省掉）——分块归约思路见 docs/05 §4.1。
 * 两级归约:
 *   GPU 侧（kernel argmax_partial）: 每 threadgroup 的 4 个 simdgroup 各
 *     完整算一行的点积，TG 内选优写进 results[TG 号]（value+index 对）；
 *   CPU 侧: 对 nc 个 chunk 局部最优线性扫一遍取全局最大。
 * 派发参数: rpc = 每 chunk 行数（SIMD 宽度），nc = ceil(vocab/rpc) 个
 * chunk；tgs = 4×SIMD 宽（一 TG 四个 simdgroup），ntg = ceil(nc/4)。
 * @return 全局 argmax 的 token id；GPU 路径不可用时返回 -1（调用方回退
 * CPU 的 argmax matvec，见 docs/08-解码策略与后处理篇.md）。 */
int ds_metal_lm_head_argmax(ds_metal_ctx_t *c, const float *x,
                              const uint16_t *W, int hidden, int vocab) {
    if (!c||!x||!W) return -1;
    id<MTLComputePipelineState> ps = get_ps(c, "argmax_partial"); /* 查 argmax kernel 管线 */
    if (!ps) return -1;
    int rpc = (int)ps.threadExecutionWidth;    /* 每 chunk 行数 = SIMD 宽度 */
    int nc = (vocab + rpc - 1) / rpc;          /* chunk 总数（向上取整） */
    c->x_b = ensure_buf(c->dev, c->x_b, &c->x_n, hidden);  /* 输入 x 复用 */
    if (!c->ar_b || c->ar_n < nc) {            /* 结果表按 nc 扩容（argmax_pair 对） */
        c->ar_b = [c->dev newBufferWithLength:(size_t)nc*sizeof(argmax_pair_t)
                                       options:MTLResourceStorageModeShared];
        c->ar_n = nc;
    }
    memcpy(c->x_b.contents, x, hidden*sizeof(float));      /* 拷入输入向量 */
    id<MTLBuffer> Wb = get_wbuf(c, W, (size_t)vocab*hidden*2); /* LM head 权重零拷贝注册 */
    id<MTLCommandBuffer> cmd = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:c->ar_b offset:0 atIndex:0];  /* buffer(0) = chunk 结果表 */
    [enc setBuffer:c->x_b  offset:0 atIndex:1];  /* buffer(1) = 输入 x */
    [enc setBuffer:Wb      offset:0 atIndex:2];  /* buffer(2) = BF16 权重 */
    [enc setBytes:&hidden length:4 atIndex:3];   /* buffer(3) = in_dim */
    [enc setBytes:&vocab  length:4 atIndex:4];   /* buffer(4) = vocab（行总数） */
    [enc setBytes:&rpc    length:4 atIndex:5];   /* buffer(5) = rows_per_chunk */
    NSUInteger sw = ps.threadExecutionWidth;
    NSUInteger tgs = sw * 4;                 /* 一 TG = 4 个 simdgroup = 4 行并行 */
    NSUInteger ntg = (nc + 3) / 4;           /* TG 数 = 每 TG 4 行覆盖 nc */
    [enc dispatchThreadgroups:MTLSizeMake(ntg,1,1)
          threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    /* CPU 侧第二级归约：扫全部 chunk 局部最优，取全局最大 */
    argmax_pair_t *res = (argmax_pair_t*)c->ar_b.contents;
    float bv = -1e30f; int bi = 0;           /* bv/bi: 当前全局最优值/下标 */
    for (int i = 0; i < nc; i++)
        if (res[i].index >= 0 && res[i].value > bv) { bv = res[i].value; bi = res[i].index; }
    c->ndispatch++;
    return bi;                               /* argmax token id */
}
