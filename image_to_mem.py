#!/usr/bin/env python3
"""Simple image-to-.mem converter for BNN test inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_SIZE = 28
THRESHOLD = 127


def resolve_mem_dir() -> Path:
    root = Path(__file__).resolve().parent
    candidates = [root / "project_1" / "mem_files", root / "mem_files"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find mem_files folder")


def convert_image_to_bits(image_path: Path) -> str:
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    binary = (arr < THRESHOLD).astype(np.uint8)
    return "".join("1" if v else "0" for v in binary.reshape(-1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert image to .mem in mem_files")
    parser.add_argument("image", type=Path, help="Input image path (png/jpg)")
    parser.add_argument(
        "output_name",
        nargs="?",
        default="input.mem",
        help="Optional output file name (default: input.mem)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_name = args.output_name
    if not output_name.lower().endswith(".mem"):
        output_name = f"{output_name}.mem"

    mem_dir = resolve_mem_dir()
    output_path = mem_dir / output_name

    bits = convert_image_to_bits(image_path)
    if len(bits) != 784:
        raise ValueError("Generated bitstream is not 784 bits")

    output_path.write_text(bits + "\n", encoding="ascii")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
