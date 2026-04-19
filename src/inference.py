"""
Batch text-to-image inference pipeline.

Wraps AttnGANWrapper to handle a list of prompts, saves each generated
image to disk, and returns the paths.
"""

import os
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

# Project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.kaggle_config import cfg
from src.utils import save_image, mkdir_p


def run_inference(
    wrapper,
    texts: Sequence[str],
    output_dir: str,
    copies: int = 2,
    filename_prefix: str = "generated",
) -> list[str]:
    """
    Generate one image per text prompt and save results to *output_dir*.

    Args:
        wrapper:         Loaded AttnGANWrapper instance.
        texts:           List of text prompts.
        output_dir:      Directory to write PNG files into.
        copies:          Internal batch copies for BatchNorm stability (≥2).
        filename_prefix: Prefix for saved file names.

    Returns:
        List of absolute paths to the saved images.
    """
    mkdir_p(output_dir)
    saved_paths: list[str] = []

    print(f"\n[Inference] Generating {len(texts)} image(s) → {output_dir}\n")

    for idx, text in enumerate(texts, start=1):
        print(f"  [{idx}/{len(texts)}] Prompt: \"{text}\"")

        try:
            image = wrapper.generate(text, copies=copies)
        except ValueError as exc:
            print(f"    WARNING: Skipping — {exc}")
            continue

        filename = f"{filename_prefix}_{idx:03d}.png"
        out_path = os.path.join(output_dir, filename)
        save_image(image, out_path)
        saved_paths.append(out_path)

    print(f"\n[Inference] Done. {len(saved_paths)} image(s) saved.\n")
    return saved_paths


def display_grid(image_paths: list[str], cols: int = 4) -> None:
    """
    Compose a contact-sheet grid from saved images and display it inline
    (works in Kaggle / Jupyter notebooks).

    Falls back to printing paths if the display environment is unavailable.
    """
    try:
        from IPython.display import display as ipy_display
        import math

        images = [Image.open(p) for p in image_paths]
        if not images:
            return

        w, h = images[0].size
        rows = math.ceil(len(images) / cols)
        grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))

        for i, img in enumerate(images):
            col = i % cols
            row = i // cols
            grid.paste(img, (col * w, row * h))

        ipy_display(grid)

    except ImportError:
        print("IPython not available — paths to saved images:")
        for p in image_paths:
            print(f"  {p}")
