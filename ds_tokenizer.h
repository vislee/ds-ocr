/*
 * ds_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt)
 */

#ifndef DS_TOKENIZER_H
#define DS_TOKENIZER_H

typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} ds_tokenizer_t;

/* Load tokenizer from vocab.json in model directory */
ds_tokenizer_t *ds_tokenizer_load(const char *vocab_json_path);

/* Decode a single token ID to text. Returns pointer to internal string. */
const char *ds_tokenizer_decode(const ds_tokenizer_t *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
int *ds_tokenizer_encode(const ds_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Free tokenizer */
void ds_tokenizer_free(ds_tokenizer_t *tok);

#endif /* DS_TOKENIZER_H */
