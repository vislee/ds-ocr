#!/usr/bin/env python3
"""
DeepSeek-OCR Python encoder cache generator.

Generates precomputed encoder output (.bin files) that the C engine can load
via DS_LOAD_ENCODER_OUTPUT, bypassing the C SAM+Encoder pipeline entirely.
This provides production-quality OCR output while the C encoder precision
is being improved.

Usage:
    python3 gen_encoder_cache.py --image input.png --model_dir deepseek-ocr
    # Then run C:
    DS_LOAD_ENCODER_OUTPUT=encoder_cache/input.bin ./ds_ocr -d deepseek-ocr -i input.png

The script handles:
- Small images (≤768px): global view only (1024x1024)
- Large images (>768px): 4-crop + global view

Output format: [n_tokens, 1280] float32
- Small: [257, 1280] = 256 global + 1 view_separator
- Large: [833, 1280] = 576 local(4*144) + 256 global + 1 view_separator
"""
import argparse
import os
import sys
import torch
import numpy as np
import json
from PIL import Image, ImageOps

# Model loading
def load_model(model_dir):
    """Load DeepSeek-OCR model from safetensors."""
    from transformers import AutoConfig
    from safetensors.torch import load_file
    import importlib

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    mod = importlib.import_module('transformers_modules.deepseek-ocr.modeling_deepseekocr2')
    ModelClass = mod.DeepseekOCR2ForCausalLM

    # Find safetensors file
    import glob
    st_files = glob.glob(os.path.join(model_dir, '*.safetensors'))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    sd = load_file(st_files[0])
    model = ModelClass(config)
    model.load_state_dict(sd, strict=False)
    model = model.eval().float()
    del sd

    return model

def process_image(model, image_path, use_gpu=False):
    """Process an image through SAM + Encoder + Projector."""
    inner = model.model
    sam = inner.sam_model
    qwen2_enc = inner.qwen2_model
    projector = inner.projector

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    use_crop = (w > 768 or h > 768)

    features_list = []

    with torch.no_grad():
        if use_crop:
            # 4-crop mode
            crop_w, crop_h = w // 2, h // 2
            for yi in range(2):
                for xi in range(2):
                    crop = img.crop((xi*crop_w, yi*crop_h,
                                     (xi+1)*crop_w, (yi+1)*crop_h))
                    pv = preprocess_image(crop, 768).to(device)
                    sam_out = sam(x=pv)
                    enc_out = qwen2_enc(sam_out)
                    proj_out = projector(enc_out)
                    features_list.append(('local', proj_out[0].cpu().numpy()))

            # Global view
            pv = preprocess_image(img, 1024).to(device)
            sam_out = sam(x=pv)
            enc_out = qwen2_enc(sam_out)
            proj_out = projector(enc_out)
            features_list.append(('global', proj_out[0].cpu().numpy()))
        else:
            # Small image: global view only
            pv = preprocess_image(img, 1024).to(device)
            sam_out = sam(x=pv)
            enc_out = qwen2_enc(sam_out)
            proj_out = projector(enc_out)
            features_list.append(('global', proj_out[0].cpu().numpy()))

    # Concatenate features
    local_features = [f for t, f in features_list if t == 'local']
    global_features = [f for t, f in features_list if t == 'global'][0]

    # View separator
    from safetensors import safe_open
    import glob
    st_files = glob.glob(os.path.join(os.path.dirname(image_path) if os.path.dirname(image_path) else '.', 'deepseek-ocr', '*.safetensors'))
    if not st_files:
        st_files = glob.glob('deepseek-ocr/*.safetensors')

    view_sep = None
    if st_files:
        with safe_open(st_files[0], framework='pt') as f:
            for k in f.keys():
                if 'view_sep' in k.lower():
                    view_sep = f.get_tensor(k).float().numpy()
                    break
    if view_sep is None:
        view_sep = np.zeros(1280, dtype=np.float32)

    parts = []
    if local_features:
        parts.append(np.concatenate(local_features, axis=0))
    parts.append(global_features)
    parts.append(view_sep.reshape(1, -1))

    encoder_output = np.concatenate(parts, axis=0)
    return encoder_output

def preprocess_image(img, target_size):
    """Resize and pad image to target_size x target_size."""
    w, h = img.size
    max_dim = max(w, h)
    ratio = target_size / max_dim
    new_w = min(int(w * ratio + 0.5), target_size)
    new_h = min(int(h * ratio + 0.5), target_size)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    pad_color = tuple(int(x * 255) for x in [0.5, 0.5, 0.5])
    pad_img = ImageOps.pad(img_resized, (target_size, target_size), color=pad_color)

    pixels = np.array(pad_img, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0).float()

def main():
    parser = argparse.ArgumentParser(description='Generate encoder cache for DeepSeek-OCR')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--model_dir', default='deepseek-ocr', help='Model directory')
    parser.add_argument('--output_dir', default='encoder_cache', help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_dir}...")
    model = load_model(args.model_dir)
    print("Model loaded.")

    print(f"Processing {args.image}...")
    encoder_output = process_image(model, args.image, use_gpu=args.gpu)

    # Save
    name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.output_dir, f'{name}.bin')
    encoder_output.tofile(out_path)

    print(f"Saved encoder output: {encoder_output.shape} to {out_path}")
    print(f"  ({encoder_output.size * 4} bytes)")
    print(f"\nTo use with C engine:")
    print(f"  DS_LOAD_ENCODER_OUTPUT={out_path} ./ds_ocr -d {args.model_dir} -i {args.image}")

if __name__ == '__main__':
    main()
