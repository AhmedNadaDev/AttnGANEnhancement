"""
main_kaggle.py — AttnGAN inference entry point (local or Kaggle).

Usage (from this project directory):
    python main_kaggle.py

By default, pretrained weights are loaded from ./data (same three files as
AttnGAN/eval/data/). On Kaggle, set MODEL_DIR (and optionally OUTPUT_DIR)
in the PATH CONFIGURATION block below.

Expected layout for MODEL_DIR:
    captions.pickle        ← vocabulary (from CUB-200 preprocessing)
    text_encoder200.pth    ← DAMSM text encoder checkpoint
    bird_AttnGAN2.pth      ← AttnGAN generator checkpoint

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

# ── PATH CONFIGURATION — override on Kaggle if needed ────────────────────────

# Directory containing model checkpoints + captions.pickle.
# Default: bundled <project>/data (copy of AttnGAN/eval/data).
# Kaggle example: MODEL_DIR = "/kaggle/input/attngan-pretrained"
MODEL_DIR = os.path.join(_HERE, "data")

# File names inside MODEL_DIR (defaults match the released bird checkpoint).
TEXT_ENCODER_FILE = "text_encoder200.pth"
GENERATOR_FILE    = "bird_AttnGAN2.pth"
VOCAB_FILE        = "captions.pickle"
TEXT_ENCODER_TYPE = "rnn"  # set to "bert" for bert_text_encoder.pth

# BERT-compatible Kaggle example:
# TEXT_ENCODER_FILE = "bert_text_encoder.pth"
# GENERATOR_FILE    = "bert_AttnGAN_generator.pth"
# TEXT_ENCODER_TYPE = "bert"

# Where to write generated PNG images (default: <project>/outputs).
# Kaggle example: OUTPUT_DIR = "/kaggle/working/outputs"
OUTPUT_DIR = os.path.join(_HERE, "outputs")

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
    cfg.TEXT.ENCODER_TYPE = TEXT_ENCODER_TYPE
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
    print("  AttnGAN - Text-to-Image Inference")
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
