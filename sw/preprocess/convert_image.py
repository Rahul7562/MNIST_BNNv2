import os
import argparse
import json
import numpy as np
from PIL import Image

def load_meta(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            return meta.get('mean', 0.5), meta.get('std', 0.5)
    return 0.5, 0.5

def convert_image(image_path, out_mem_path, mean, std):
    img = Image.open(image_path).convert('L')

    arr = np.array(img, dtype=np.float32) / 255.0
    corners = np.concatenate([arr[:5, :5].flatten(), arr[:5, -5:].flatten(), arr[-5:, :5].flatten(), arr[-5:, -5:].flatten()])
    bg_intensity = np.mean(corners)

    if bg_intensity > 0.5:
        img = Image.eval(img, lambda x: 255 - x)

    arr = np.array(img)
    non_zero_rows = np.any(arr > 20, axis=1)
    non_zero_cols = np.any(arr > 20, axis=0)

    if not np.any(non_zero_rows):
        bbox = (0, 0, 28, 28)
    else:
        rmin, rmax = np.where(non_zero_rows)[0][[0, -1]]
        cmin, cmax = np.where(non_zero_cols)[0][[0, -1]]
        bbox = (cmin, rmin, cmax+1, rmax+1)

    img = img.crop(bbox)

    max_dim = max(img.width, img.height)
    new_w = max(1, int((img.width / max_dim) * 20))
    new_h = max(1, int((img.height / max_dim) * 20))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))

    arr = np.array(new_img, dtype=np.float32) / 255.0
    arr = arr.flatten()

    arr = (arr - mean) / (std + 1e-7)

    b = np.where(arr >= 0, 1.0, -1.0)
    p = ((b + 1) / 2).astype(np.int32)

    out_npy_path = out_mem_path.replace('.mem', '.npy')
    np.save(out_npy_path, b)

    with open(out_mem_path, 'w') as f:
        binary_str = "".join(str(bit) for bit in p)
        f.write(binary_str + "\n")

    print(f"Exported image to {out_mem_path} and {out_npy_path}")
    return b, p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--meta', type=str, default='mem_files/export_meta.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mean, std = load_meta(args.meta)
    convert_image(args.image_path, args.out, mean, std)
