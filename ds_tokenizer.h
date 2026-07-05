/*
 * ds_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 * ds_tokenizer.h — Qwen2/Qwen3 BPE 分词器（GPT-2 字节级）
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 【模块角色】分词器是文本和数字之间的"翻译官"
 * ─────────────────────────────────────────────────────────────────────
 *
 * 【编码 (Encode)】文本 → token IDs
 *   "Hello world" → [15496, 993]
 *   用途: 将用户输入的prompt文本编码为token ID序列，构造解码器输入
 *
 * 【解码 (Decode)】token ID → 文本
 *   [15496, 993] → "Hello world"
 *   用途: 将解码器生成的token ID解码为可读文本，输出OCR结果
 *
 * 【BPE (Byte-Pair Encoding) 算法】
 *   1. 初始词表: 256个字节(0x00-0xFF) + 特殊token(BOS/EOS/PAD等)
 *   2. 迭代合并: 统计相邻token对频率，合并最高频对为新token
 *   3. 重复直到词表达到目标大小(129280)
 *   4. 编码时: 贪心匹配最长token（从左到右扫描）
 *
 * 【GPT-2 字节级 BPE 的特点】
 *   - 基础词表是字节而非字符 → 天然支持所有语言/Unicode
 *   - 每个 UTF-8 字节都有对应的初始token → 不会出现 <unk>
 *   - 合并规则按字节频率学习 → 常见字节组合获得更短编码
 *
 * 【词表结构】(vocab_size = 129280)
 *   0-255:     字节token (原始UTF-8字节)
 *   256-N:     合并token (BPE学习到的常见组合)
 *   特殊token: BOS(0), EOS(1), PAD(2), IMAGE_START(151655), IMAGE_END(151656), etc.
 *
 * 【与 Python 的对应】
 *   Python:  transformers.AutoTokenizer.from_pretrained(model_dir)
 *   C:       ds_tokenizer_load(model_dir/vocab.json)
 *   两者使用相同的 vocab.json 和 merges.txt，产生完全相同的编码/解码结果
 *
 * 【文件格式】
 *   vocab.json: {"token_text": token_id, ...}  — 主要格式（V1使用）
 *   tokenizer.json (HuggingFace): 嵌套JSON格式 — 备选格式（V2使用）
 *     {"model": {"vocab": {...}, "merges": ["a b", ...]}, ...}
 *   merges.txt / tokenizer.json.merges: BPE合并规则列表
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef DS_TOKENIZER_H
#define DS_TOKENIZER_H

/* ds_tokenizer_t — 分词器数据结构
 *
 * id_to_text: [vocab_size] — token ID → 解码文本的查找表
 *   用途: decode(token_id) → 直接索引取出对应文本
 *   例如: id_to_text[15496] = "Hello", id_to_text[0] = "<|endoftext|>"
 *
 * id_to_bpe: [vocab_size] — token ID → 原始BPE token字符串
 *   用途: 保留vocab.json中的原始token表示（可能包含特殊编码如Ġ=空格前缀）
 *   与 id_to_text 的区别: id_to_bpe保留GPT-2的特殊字节表示(如Ġ=0x20空格)
 *                         id_to_text是解码后的可读文本
 *
 * vocab_size: 词表大小 = 129280 (DeepSeek-OCR/V3的词表大小)
 *
 * vocab_map / merge_map: 内部哈希表（对调用者透明）
 *   vocab_map: BPE token字符串 → token ID（编码时查找）
 *   merge_map: 合并规则对的优先级（BPE编码时决定合并顺序）
 *
 * 【为什么需要两个映射?】
 *   encode 需要: 文本 → ID (vocab_map)
 *   decode 需要: ID → 文本 (id_to_text)
 *   BPE合并 需要: (token_a, token_b) → 优先级 (merge_map)
 */
typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings — ID→文本解码表 */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json — ID→BPE原始表示 */
    int vocab_size;

    /* Internal hash maps (opaque to callers) — 内部哈希表（调用者无需关心） */
    void *vocab_map;     /* BPE token string → token ID 映射（编码用） */
    int vocab_map_cap;
    void *merge_map;     /* (token_a, token_b) → merge priority 映射（BPE合并用） */
    int merge_map_cap;
} ds_tokenizer_t;

/* ds_tokenizer_load — 从 vocab.json 加载分词器
 *
 * 解析 vocab.json 格式: {"ĠHello": 15496, "Ġworld": 993, ...}
 * 每行一个键值对，键是BPE token字符串，值是token ID
 *
 * 同时加载同目录下的 merges.txt（BPE合并规则）
 *
 * 返回: 新分配的 ds_tokenizer_t（调用者负责释放），失败返回NULL
 */
ds_tokenizer_t *ds_tokenizer_load(const char *vocab_json_path);

/* ds_tokenizer_load_from_tokenizer_json — 从 HuggingFace tokenizer.json 加载
 *
 * 解析 HuggingFace 格式的 tokenizer.json:
 * {
 *   "model": {
 *     "vocab": {"ĠHello": 15496, ...},  ← 提取为 id_to_text / id_to_bpe
 *     "merges": ["Ġ H", "ĠHe", ...]     ← 提取为 merge_map
 *   },
 *   ...
 * }
 *
 * 这是 vocab.json 不可用时的备选方案（如 DeepSeek-OCR V2）
 * 提取逻辑: 解析嵌套JSON，从 model.vocab 和 model.merges 中提取数据
 */
ds_tokenizer_t *ds_tokenizer_load_from_tokenizer_json(const char *tokenizer_json_path);

/* ds_tokenizer_decode — 将单个 token ID 解码为文本
 *
 * 实现: 直接查表 id_to_text[token_id]
 * 返回: 指向内部字符串的指针（不需要调用者释放，但不要跨调用保存）
 *
 * 特殊token解码:
 *   EOS(1) → "" (空字符串，不输出)
 *   BOS(0) → "" (空字符串)
 *   IMAGE_START(151655) → "<|image_start|>" (特殊标记)
 *
 * 【UTF-8 字节级解码】
 * GPT-2 BPE 的 token 可能只包含 UTF-8 字节的一部分，
 * 多个 token 组合后才形成完整的 UTF-8 字符。
 * 例如: 中文字符"你" = UTF-8 [0xE4, 0xBD, 0xA0]
 * 可能被编码为3个独立的字节token，解码时需要拼接后才能还原
 */
const char *ds_tokenizer_decode(const ds_tokenizer_t *tok, int token_id);

/* ds_tokenizer_encode — 将 UTF-8 文本编码为 token ID 序列
 *
 * BPE 编码算法:
 *   1. 将输入文本转为 UTF-8 字节序列
 *   2. 每个字节映射为初始 token ID (字节级BPE的基础词表)
 *   3. 迭代合并: 按优先级从高到低，检查相邻token对是否在合并规则中
 *      如果在，则合并为新token；重复直到无法合并
 *   4. 返回最终的 token ID 数组
 *
 * 返回: malloc'd 的 token ID 数组，*out_n_tokens = token数量
 *       调用者需 free() 返回的数组
 *       失败返回 NULL (*out_n_tokens = 0)
 *
 * 【实际使用】
 * 在 ds-ocr 中，encode 主要用于编码 prompt 文本:
 *   V1/V2: "\nFree OCR." → [201, 21431, 126041, 16]
 *   V3: "\ndocument parsing." → [201, 34030, 76466, 16]
 * 编码的 token 数很少(4个)，所以性能不是瓶颈
 */
int *ds_tokenizer_encode(const ds_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* ds_tokenizer_free — 释放分词器资源
 *
 * 释放所有动态分配的内存:
 *   - id_to_text / id_to_bpe 字符串数组
 *   - vocab_map / merge_map 哈希表
 *   - 结构体本身
 */
void ds_tokenizer_free(ds_tokenizer_t *tok);

#endif /* DS_TOKENIZER_H */
