/*
 * ds_safetensors.h - Safetensors file format reader (multi-shard support)
 * Adapted from antirez/qwen-asr project.
 *
 * Safetensors文件格式读取器（支持多分片）
 *
 * 【Safetensors格式概览】
 * Safetensors 是 HuggingFace 设计的模型权重存储格式，相比 PyTorch 的 .pt 格式，
 * 它更安全（不含可执行代码）、加载更快（支持 mmap 零拷贝）。
 *
 * 文件二进制布局：
 *   [8字节] header长度 (little-endian uint64)  ── 记录JSON header的字节数
 *   [N字节] JSON header                         ── 描述所有张量的名称、数据类型、形状、偏移
 *   [剩余]  二进制数据区                         ── 所有张量的实际权重数据连续存放
 *
 * JSON header 示例：
 *   {
 *     "layer.weight": { "dtype": "BF16", "shape": [4096, 4096], "data_offsets": [0, 33554432] },
 *     "layer.bias":   { "dtype": "F32",  "shape": [4096],       "data_offsets": [33554432, 33570848] }
 *   }
 * 其中 data_offsets 是相对于数据区起始位置的偏移（不是文件偏移！）。
 *
 * 【核心设计理念】
 * 1. mmap零拷贝：将整个文件映射到虚拟地址空间，无需 read() 系统调用，无需用户态缓冲区复制
 * 2. 惰性加载：操作系统按需将文件页读入物理内存，未被访问的页不会占用 RAM
 * 3. BF16直接指针：权重以 BF16 格式存储在文件中，直接返回 mmap 区域内的指针，
 *    推理时由数学内核在 matvec 过程中完成 BF16→F32 转换，避免预转换的内存开销
 * 4. 多分片支持：大模型（如 DeepSeek-V2）权重可达数十 GB，需拆分为多个文件存储，
 *    本模块提供统一的跨分片查找接口
 */

#ifndef DS_SAFETENSORS_H
#define DS_SAFETENSORS_H

#include <stddef.h>   /* size_t 类型定义 */
#include <stdint.h>   /* uint16_t, int64_t 等定宽整数类型 */

/* 单个safetensors文件中最多允许的张量数量上限。
 * 为什么需要上限：避免动态分配，简化内存管理，4096足以覆盖绝大多数模型。 */
#define SAFETENSORS_MAX_TENSORS 4096

/* 多分片模型的最大分片文件数量。
 * 为什么是8：DeepSeek-V2等大模型通常拆分为5-8个分片文件，8足够且不过度浪费栈空间。 */
#define SAFETENSORS_MAX_SHARDS 8

/* Safetensors支持的数据类型枚举。
 * 为什么需要枚举：JSON header中的 "dtype" 字段是字符串（如 "BF16"），
 * 解析后转换为枚举值，便于在C代码中用 switch 判断，避免字符串比较的开销。 */
typedef enum {
    DTYPE_F32 = 0,     /* 32位浮点（IEEE 754 single），4字节/元素 */
    DTYPE_F16 = 1,     /* 16位浮点（IEEE 754 half），2字节/元素，GPU常用 */
    DTYPE_BF16 = 2,    /* 16位浮点（Brain Float），2字节/元素，DeepSeek/LLaMA等现代LLM的标准格式
                        * BF16 vs F16：BF16的指数位与F32相同（8位），尾数7位；
                        * F16指数5位尾数10位。BF16范围更大不易溢出，精度略低但对深度学习足够。 */
    DTYPE_I32 = 3,     /* 32位有符号整数，用于token embedding的查找表索引等 */
    DTYPE_I64 = 4,     /* 64位有符号整数，用于大型shape参数或rope频率表 */
    DTYPE_BOOL = 5,    /* 布尔类型，1字节/元素，用于注意力掩码等 */
    DTYPE_UNKNOWN = -1 /* 未知类型：遇到不支持的dtype字符串时回退到此值，避免解析崩溃 */
} safetensor_dtype_t;

/* 单个张量的元数据描述。
 * 这是从JSON header中解析出来的结构化信息，配合safetensors_file_t使用，
 * 不存储实际数据——数据仍在mmap区域中，通过 data_offset + 基地址 计算指针访问。 */
typedef struct {
    char name[256];                /* 张量名称，如 "model.layers.0.self_attn.q_a_proj.weight"
                                    * 为什么是固定256字节：简化内存管理，避免动态分配；
                                    * 256足以容纳任何合理的张量命名层级。 */
    safetensor_dtype_t dtype;      /* 数据类型枚举值，决定元素大小和解析方式 */
    int ndim;                      /* 维度数（张量秩），0=标量, 1=向量, 2=矩阵, ... */
    int64_t shape[8];              /* 每个维度的大小，如 [4096, 4096] 表示4096×4096矩阵
                                    * 最多支持8维，覆盖所有常见张量（权重矩阵最多4维） */
    size_t data_offset;            /* 张量数据在mmap区域【数据区】内的偏移（字节）
                                    * 注意：这是相对于数据区起始位置的偏移，不是文件偏移！
                                    * 实际指针 = (char*)mmap_base + header_size + data_offset */
    size_t data_size;              /* 张量数据的总字节数 = 产品(shape) × sizeof(dtype)
                                    * 可用于校验：确认shape和dtype是否与数据大小一致 */
} safetensor_t;

/* 单个Safetensors文件的完整映射。
 * 打开文件时执行：mmap映射 → 解析8字节header长度 → 解析JSON header →
 * 填充tensors数组。文件内容始终保留在mmap区域中，不会复制到用户态缓冲区。 */
typedef struct {
    char *path;                    /* 文件路径，用于调试和错误报告时追溯来源
                                    * 为什么保存路径：多分片场景下需要知道某个张量来自哪个文件 */
    void *data;                    /* mmap映射的基地址（指向文件的第一个字节）
                                    * 为什么用void*：mmap返回void*，保持类型中立；
                                    * 实际访问时需要转换为char*做指针算术 */
    size_t file_size;              /* 文件总大小（字节），= 8 + header_size + 数据区大小
                                    * 为什么记录：mmap映射需要指定长度；关闭时munmap也需要 */
    size_t header_size;            /* JSON header的字节数（即前8字节记录的值）
                                    * 为什么单独记录：数据区起始位置 = data + 8 + header_size，
                                    * 这是计算任意张量数据指针的基础 */
    char *header_json;             /* 指向JSON header字符串的指针（位于mmap区域内，零拷贝）
                                    * 为什么保留：调试时可以打印原始header；也避免了字符串复制。
                                    * 注意：此指针指向mmap区域内部，不需要单独free */
    int num_tensors;               /* 本文件中包含的张量总数 */
    safetensor_t tensors[SAFETENSORS_MAX_TENSORS]; /* 张量元数据数组，从JSON header解析填充 */
} safetensors_file_t;

/* 多分片包装器：打开所有分片文件，提供统一的张量查找接口。
 *
 * 为什么需要多分片：
 * - 大模型权重可达数十GB（如DeepSeek-V2 MoE约20GB+）
 * - 单个文件在传输、存储、mmap时都有不便
 * - HuggingFace会将大模型拆分为 model-00001-of-00005.safetensors 等多个文件
 * - 本结构将这些分片统一管理，对上层透明——调用者只需按张量名查找即可
 *
 * 使用方式：
 *   multi_safetensors_t *ms = multi_safetensors_open("model_dir/");
 *   const safetensor_t *t = multi_safetensors_find(ms, "layer.weight", &sf);
 *   uint16_t *bf16_ptr = safetensors_get_bf16_direct(sf, t);  // 零拷贝获取BF16数据
 */
typedef struct {
    safetensors_file_t *shards[SAFETENSORS_MAX_SHARDS]; /* 分片文件数组，每个元素是一个已mmap的文件 */
    int num_shards;              /* 实际打开的分片文件数量 */
} multi_safetensors_t;

/* ======================== 单文件操作 ======================== */

/* safetensors_open - 打开单个safetensors文件（使用mmap映射）
 *
 * 流程：
 *   1. open() 打开文件 → 获取文件描述符
 *   2. fstat() 获取文件大小
 *   3. mmap() 将整个文件映射到内存 → 返回基地址 data
 *   4. 读取前8字节 → 得到 header_size（little-endian uint64）
 *   5. 从 data+8 开始提取 header_size 字节的JSON字符串
 *   6. 解析JSON → 填充 tensors[] 数组（名称、dtype、shape、偏移、大小）
 *
 * 为什么用mmap而不是read：
 * - 零拷贝：数据直接在内核页缓存中访问，无需复制到用户态缓冲区
 * - 惰性加载：只有被访问的页才从磁盘读入，节省启动时间和内存
 * - 简化代码：不需要手动管理读取缓冲区和偏移计算
 *
 * @param path  safetensors文件路径
 * @return      成功返回safetensors_file_t指针，失败返回NULL
 */
safetensors_file_t *safetensors_open(const char *path);

/* safetensors_close - 关闭safetensors文件，释放资源
 *
 * 流程：munmap(data, file_size) → 释放mmap映射 → free(sf) 释放结构体
 * 注意：header_json指向mmap区域内部，munmap后自动失效，不需要单独释放。
 *
 * @param sf  之前通过safetensors_open返回的句柄
 */
void safetensors_close(safetensors_file_t *sf);

/* ======================== 多分片操作 ======================== */

/* multi_safetensors_open - 从模型目录打开权重（自动检测单文件或多分片）
 *
 * 自动检测逻辑：
 * - 如果目录中只有 model.safetensors → 单文件模式，shards[0] 指向它
 * - 如果存在 model-00001-of-0000X.safetensors → 多分片模式，按序号依次打开
 *
 * 为什么自动检测：上层代码不需要关心模型是单文件还是多分片，
 * 统一调用 multi_safetensors_find() 即可，降低了使用复杂度。
 *
 * @param model_dir  模型目录路径（如 "/data/deepseek-ocr/"）
 * @return           成功返回multi_safetensors_t指针，失败返回NULL
 */
multi_safetensors_t *multi_safetensors_open(const char *model_dir);

/* multi_safetensors_close - 关闭所有分片文件，释放全部资源
 *
 * 依次对每个shard调用safetensors_close()，然后释放multi_safetensors_t本身。
 *
 * @param ms  之前通过multi_safetensors_open返回的句柄
 */
void multi_safetensors_close(multi_safetensors_t *ms);

/* multi_safetensors_find - 跨所有分片查找指定名称的张量
 *
 * 查找策略：线性遍历所有分片，在每个分片的tensors[]中按名称匹配。
 * 为什么线性遍历够用：张量查找只在模型加载阶段进行（不是推理热路径），
 * 且张量总数通常在数百级别，线性搜索开销可忽略。
 *
 * @param ms      多分片句柄
 * @param name    要查找的张量名称（如 "vision_encoder.blocks.0.attn.q.weight"）
 * @param out_sf  [输出] 找到时，返回该张量所属的分片文件指针
 *                为什么需要out_sf：后续获取数据时需要知道张量在哪个分片的mmap区域中，
 *                才能计算正确的数据指针。
 * @return        找到返回safetensor_t指针（指向shard内部的tensors数组元素，不要free），
 *                未找到返回NULL
 */
const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            safetensors_file_t **out_sf);

/* ======================== 数据访问 ======================== */

/* safetensors_data - 获取张量的原始数据指针（位于mmap区域内）
 *
 * 计算方式：ptr = (char*)sf->data + 8 + sf->header_size + t->data_offset
 *                    \_mmap基地址_/  \____跳过header____/  \_张量偏移_/
 *
 * 为什么返回const void*：
 * - 数据在mmap区域内，不应该被修改（模型权重是只读的）
 * - void*是类型中立的，调用者需要根据dtype自行转换为具体类型指针
 * - 返回的是mmap区域内部指针，零拷贝，无需调用者释放
 *
 * @param sf  张量所属的分片文件（由multi_safetensors_find的out_sf提供）
 * @param t   目标张量的元数据
 * @return    指向张量数据起始位置的只读指针
 */
const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t);

/* safetensors_get_f32 - 获取张量数据并转换为float32格式（需要分配新内存）
 *
 * 为什么需要此函数：某些场景（如后处理、调试输出）需要F32格式数据，
 * 而权重文件中存储的可能是BF16/F16。
 *
 * 为什么会分配内存：BF16→F32转换需要2倍空间，无法原地完成，
 * 必须分配新缓冲区并逐元素转换。调用者负责free()释放返回的指针。
 *
 * 注意：如果原始数据已经是F32，此函数仍会复制一份（保持接口一致性）。
 * 对于BF16权重，推荐使用 safetensors_get_bf16_direct() 零拷贝接口，
 * 在matvec内核中完成转换，避免额外的内存分配和复制开销。
 *
 * @param sf  张量所属的分片文件
 * @param t   目标张量的元数据
 * @return    新分配的float数组指针（caller必须free），失败返回NULL
 */
float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t);

/* safetensors_get_bf16_direct - 直接获取BF16数据指针（零拷贝，不分配内存）
 *
 * 这是最核心的性能优化接口！
 *
 * 为什么BF16可以零拷贝：
 * - 模型权重在文件中就是以BF16格式存储的（2字节/元素）
 * - mmap将文件映射到内存后，直接返回mmap区域内的指针
 * - 推理时，数学内核（NEON/AVX/generic）在matvec过程中"即时"将BF16转为F32运算
 * - 这样无需预先将全部权重转换为F32（否则内存占用翻倍！）
 *
 * 使用模式：
 *   uint16_t *w = safetensors_get_bf16_direct(sf, tensor);
 *   // 将 w 指针传给 ds_matvec_bf16() 等内核函数
 *   // 内核在运算时逐元素做 bf16_to_f32 转换
 *
 * @param sf  张量所属的分片文件
 * @param t   目标张量的元数据（必须确保dtype == DTYPE_BF16，否则行为未定义）
 * @return    指向mmap区域内BF16数据的指针（不需要free！），失败返回NULL
 */
uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t);

/* ======================== 辅助工具 ======================== */

/* safetensor_is_bf16 - 检查张量是否为BF16类型
 *
 * 为什么需要单独检查：在加载权重时，需要根据数据类型选择不同的处理路径：
 * - BF16 → 使用 safetensors_get_bf16_direct() 零拷贝获取，传给融合内核
 * - F32  → 使用 safetensors_get_f32() 获取（或直接 safetensors_data + 强制转换）
 * - 其他 → 根据场景特殊处理
 *
 * @param t  目标张量
 * @return   1表示BF16，0表示不是
 */
int safetensor_is_bf16(const safetensor_t *t);

/* safetensor_numel - 计算张量的元素总数
 *
 * 计算方式：shape[0] × shape[1] × ... × shape[ndim-1]
 * 等价于 t->data_size / sizeof(dtype)，但此函数不需要知道dtype。
 *
 * 为什么需要：分配输出缓冲区、验证shape合法性、计算RoPE位置索引等场景都需要元素总数。
 *
 * @param t  目标张量
 * @return   元素总数（所有维度的乘积），0维标量返回1
 */
int64_t safetensor_numel(const safetensor_t *t);

/* safetensor_print - 打印单个张量的元数据（用于调试）
 *
 * 输出格式示例：layer.weight [BF16] shape=[4096,4096] offset=0 size=33554432
 *
 * @param t  目标张量
 */
void safetensor_print(const safetensor_t *t);

/* safetensors_print_all - 打印文件中所有张量的元数据（用于调试）
 *
 * 为什么需要：模型加载后打印所有张量信息，可验证：
 * - 文件是否正确解析（张量数量、名称是否与预期一致）
 * - 权重是否完整（没有遗漏的层）
 * - dtype是否正确（全部BF16还是混合精度）
 *
 * @param sf  目标safetensors文件
 */
void safetensors_print_all(const safetensors_file_t *sf);

#endif /* DS_SAFETENSORS_H */