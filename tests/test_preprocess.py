import os
import json
import pytest
import numpy as np
from PIL import Image

from sw.preprocess.convert_image import convert_image

def test_convert_image_roundtrip(tmp_path):
    # Create a dummy meta file
    meta_path = tmp_path / "export_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({'mean': 0.5, 'std': 0.5}, f)

    # Create a dummy image (black background, white square)
    img_arr = np.zeros((28, 28), dtype=np.uint8)
    img_arr[10:18, 10:18] = 255
    img = Image.fromarray(img_arr)
    img_path = tmp_path / "dummy.png"
    img.save(img_path)

    out_mem_path = str(tmp_path / "dummy.mem")
    b, p = convert_image(str(img_path), out_mem_path, 0.5, 0.5)

    assert os.path.exists(out_mem_path)
    assert os.path.exists(out_mem_path.replace('.mem', '.npy'))
    assert b.shape == (784,)
    assert p.shape == (784,)

    # Values should be in {-1, 1} and {0, 1}
    assert np.all(np.isin(b, [-1.0, 1.0]))
    assert np.all(np.isin(p, [0, 1]))

    # Check mem file
    with open(out_mem_path, 'r') as f:
        mem_str = f.read().strip()
    assert len(mem_str) == 784

def test_convert_image_invert(tmp_path):
    # Dummy image with white background, black square
    img_arr = np.ones((28, 28), dtype=np.uint8) * 255
    img_arr[10:18, 10:18] = 0
    img = Image.fromarray(img_arr)
    img_path = tmp_path / "dummy_inv.png"
    img.save(img_path)

    out_mem_path = str(tmp_path / "dummy_inv.mem")
    b, p = convert_image(str(img_path), out_mem_path, 0.5, 0.5)

    # Auto-invert should trigger and the result should roughly match the non-inverted case's center
    center = p.reshape(28, 28)[10:18, 10:18]
    assert np.mean(center) > 0.5 # Should mostly be 1s because it's the "content"
