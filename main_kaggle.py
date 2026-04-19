"""
main_kaggle.py — AttnGAN Kaggle inference entry point.

Usage (from inside the kaggle_attngan/ directory):
    python main_kaggle.py

Or from the parent directory:
    python kaggle_attngan/main_kaggle.py

Before running, update the PATH CONFIGURATION block below to match
the paths inside your Kaggle notebook.

Expected dataset layout (upload as a Kaggle dataset):
    attngan-pretrained/
    ├── captions.pickle        ← vocabulary (from CUB-200 preprocessing)
    ├── text_encoder200.pth    ← DAMSM text encoder checkpoint
    └── bird_AttnGAN2.pth     ← AttnGAN generator checkpoint

Pretrained files can be obtained from the original repository:
    https://github.com/taoxugit/AttnGAN
    (Google Drive links in the README)
"""

import os
import sys

# ── Ensure the project root is always on sys.path ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── PATH CONFIGURATION — edit these for your Kaggle environment ──────────────

# Directory containing model checkpoints + captions.pickle.
# In Kaggle: /kaggle/input/<your-dataset-name>/
MODEL_DIR = "/kaggle/input/attngan-pretrained"

# File names inside MODEL_DIR (defaults match the released bird checkpoint).
TEXT_ENCODER_FILE = "text_encoder200.pth"
GENERATOR_FILE    = "bird_AttnGAN2.pth"
VOCAB_FILE        = "captions.pickle"

# Where to write generated PNG images.
OUTPUT_DIR = "/kaggle/working/outputs"

# ── TEXT PROMPTS — add or modify freely ──────────────────────────────────────

TEXTS = [
    "a small red bird with blue wings and a yellow beak",
    "a yellow bird sitting on a branch with green leaves",
    "this bird has a white belly and black crown with orange breast",
    "a large bird with a long orange beak and blue feathers",
    "a tiny bird with a red head and grey body perched on a twig",
]

# ── INFERENCE SETTINGS ───────────────────────────────────────────────────────

COPIES = 2          # internal batch size (keep ≥ 2 for BatchNorm stability)
FILENAME_PREFIX = "attngan"

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Apply path config to global cfg ──────────────────────────────────────
    from config.kaggle_config import cfg

    cfg.MODEL_DIR       = MODEL_DIR
    cfg.CAPTIONS_PICKLE = VOCAB_FILE
    cfg.TRAIN.NET_E     = TEXT_ENCODER_FILE
    cfg.TRAIN.NET_G     = GENERATOR_FILE
    cfg.OUTPUT_DIR      = OUTPUT_DIR

    # ── Verify model files exist before loading ───────────────────────────────
    required_files = {
        "Vocabulary":     os.path.join(MODEL_DIR, VOCAB_FILE),
        "Text encoder":   os.path.join(MODEL_DIR, TEXT_ENCODER_FILE),
        "Generator":      os.path.join(MODEL_DIR, GENERATOR_FILE),
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not os.path.isfile(path)
    ]
    if missing:
        print("\n[ERROR] The following required files were not found:")
        for m in missing:
            print(f"  {m}")
        print(
            "\nUpdate the PATH CONFIGURATION block at the top of main_kaggle.py "
            "to point to your dataset, then re-run.\n"
        )
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    from src.model_wrapper import AttnGANWrapper
    from src.inference import run_inference, display_grid

    print("=" * 60)
    print("  AttnGAN — Kaggle Text-to-Image Inference")
    print("=" * 60)

    wrapper = AttnGANWrapper(MODEL_DIR)

    # ── Run inference ─────────────────────────────────────────────────────────
    saved_paths = run_inference(
        wrapper=wrapper,
        texts=TEXTS,
        output_dir=OUTPUT_DIR,
        copies=COPIES,
        filename_prefix=FILENAME_PREFIX,
    )

    # ── Display results (Jupyter / Kaggle notebook) ───────────────────────────
    if saved_paths:
        display_grid(saved_paths)

    print("All done!")


if __name__ == "__main__":
    main()
