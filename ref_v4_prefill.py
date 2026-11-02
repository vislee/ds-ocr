#!/usr/bin/env python3
"""对照测试: 官方 V4 模型 prefill 同样输入 → 比较 logits 与 C 引擎。
用法: /tmp/v452/bin/python ref_v4_prefill.py [embeds_bin]"""
import sys, os
import numpy as np
import torch

MODEL_DIR = 'models/Unlimited-OCR-2'
embeds_bin = sys.argv[1] if len(sys.argv) > 1 else 'dump/c_inputs_embeds.bin'

import importlib.util, sys, types
spec = importlib.util.spec_from_file_location(
    "m_uocr", "models/Unlimited-OCR-2/modeling_unlimitedocr.py")
m = importlib.util.module_from_spec(spec)
sys.modules['m_uocr'] = m
# 相对导入失败前的处理: 官方文件用 "from .modeling_deepseekv2 import ..."
# 在无包上下文中执行——先注册成包
import importlib
pkg = types.ModuleType("hf_models_pkg")
pkg.__path__ = ["models/Unlimited-OCR-2"]
sys.modules["hf_models_pkg"] = pkg
spec2 = importlib.util.spec_from_file_location(
    "hf_models_pkg.modeling_unlimitedocr", "models/Unlimited-OCR-2/modeling_unlimitedocr.py")
mod = importlib.util.module_from_spec(spec2)
sys.modules["hf_models_pkg.modeling_unlimitedocr"] = mod
spec2.loader.exec_module(mod)
UnlimitedOCRForCausalLM = mod.UnlimitedOCRForCausalLM

from transformers import AutoTokenizer
print('loading model (bf16)...', flush=True)
cfg = mod.UnlimitedOCRConfig.from_pretrained(MODEL_DIR)
model = UnlimitedOCRForCausalLM.from_pretrained(
    MODEL_DIR, config=cfg, torch_dtype=torch.bfloat16)
model = model.to('mps' if torch.backends.mps.is_available() else 'cpu')
model.eval()
print('loaded.', flush=True)

emb = np.fromfile(embeds_bin, dtype=np.float32).reshape(-1, 1280)
print('embeds:', emb.shape)
inputs_embeds = torch.from_numpy(emb).unsqueeze(0).to(model.dtype).to(next(model.parameters()).device)

with torch.no_grad():
    out = model(inputs_embeds=inputs_embeds, use_cache=False)
logits = out.logits[0, -1].float()
top = torch.topk(logits, 8)
print('Python ref prefill top8:')
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f'  {i}: {v:.3f}')
np.savetxt('dump/py_prefill_logits.txt', logits.cpu().numpy())
print('saved dump/py_prefill_logits.txt')
