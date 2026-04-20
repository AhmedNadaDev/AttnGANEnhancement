"""
main_enhanced.py — Enhanced AttnGAN pipeline entry point.

Runs the three-stage enhanced pipeline:
  Stage 1: RefinedAttnGAN  — temperature-scaled attention (τ = 0.7), top-k word masking
  Stage 2: CLIP reranking  — best of N candidates selected by CLIP ViT-B/32
  Stage 3: Post-processing — Lanczos upscale to 512×512 + sharpen + colour

Usage
-----
    python main_enhanced.py

For Kaggle:
    %run AttnGANEnhancement/main_enhanced.py

Adjust the PATH CONFIGURATION and ENHANCED SETTINGS blocks below
to match your environment and preferences.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── PATH CONFIGURATION ────────────────────────────────────────────────────────

MODEL_DIR  = os.path.join(_HERE, "data")
OUTPUT_DIR = os.path.join(_HERE, "outputs", "enhanced")

# Kaggle overrides:
# MODEL_DIR  = "/kaggle/input/attngan-pretrained"
# OUTPUT_DIR = "/kaggle/working/outputs/enhanced"

TEXT_ENCODER_FILE = "text_encoder200.pth"
GENERATOR_FILE    = "bird_AttnGAN2.pth"
VOCAB_FILE        = "captions.pickle"
TEXT_ENCODER_TYPE = "rnn"  # set to "bert" for bert_text_encoder.pth

# ── ENHANCED PIPELINE SETTINGS ────────────────────────────────────────────────

NUM_CANDIDATES   = 6      # CLIP-reranked candidates per prompt  (≥2, default 6)
ATTENTION_TEMP   = 0.7    # attention temperature τ ∈ (0, 1]     (default 0.7)
TOP_K_RATIO      = 0.75   # fraction of words kept in top-k mask (default 0.75)
BASE_SEED        = 100    # base noise seed for candidate generation

FILENAME_PREFIX  = "enhanced"

# ── TEXT PROMPTS ──────────────────────────────────────────────────────────────

TEXTS = [
    "a small red bird with blue wings and a yellow beak",
    "a yellow bird sitting on a branch with green leaves",
    "this bird has a white belly and black crown with orange breast",
    "a large bird with a long orange beak and blue feathers",
    "a tiny bird with a red head and grey body perched on a twig",
]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from config.kaggle_config import cfg

    cfg.MODEL_DIR       = MODEL_DIR
    cfg.CAPTIONS_PICKLE = VOCAB_FILE
    cfg.TRAIN.NET_E     = TEXT_ENCODER_FILE
    cfg.TRAIN.NET_G     = GENERATOR_FILE
    cfg.TEXT.ENCODER_TYPE = TEXT_ENCODER_TYPE
    cfg.OUTPUT_DIR      = OUTPUT_DIR

    # ── verify model files ────────────────────────────────────────────────────
    required = {
        "Vocabulary":   os.path.join(MODEL_DIR, VOCAB_FILE),
        "Text encoder": os.path.join(MODEL_DIR, TEXT_ENCODER_FILE),
        "Generator":    os.path.join(MODEL_DIR, GENERATOR_FILE),
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not os.path.isfile(v)]
    if missing:
        print("\n[ERROR] Missing required files:")
        for m in missing:
            print(f"  {m}")
        print(
            "\nUpdate MODEL_DIR at the top of main_enhanced.py "
            "to point to your pretrained weights.\n"
        )
        sys.exit(1)

    # ── initialise enhanced pipeline ──────────────────────────────────────────
    from src.enhanced_pipeline import EnhancedAttnGANWrapper
    from src.inference import display_grid
    from src.utils import mkdir_p, save_image

    wrapper = EnhancedAttnGANWrapper(
        MODEL_DIR,
        num_candidates=NUM_CANDIDATES,
        temperature=ATTENTION_TEMP,
        top_k_ratio=TOP_K_RATIO,
    )

    mkdir_p(OUTPUT_DIR)
    saved_paths: list[str] = []

    print(f"\n[Enhanced] Generating {len(TEXTS)} image(s) → {OUTPUT_DIR}\n")

    for idx, text in enumerate(TEXTS, start=1):
        print(f"  [{idx}/{len(TEXTS)}] {text!r}")

        enhanced, best_score, all_scores = wrapper.generate_with_scores(
            text, base_seed=BASE_SEED
        )

        filename = f"{FILENAME_PREFIX}_{idx:03d}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        save_image(enhanced, out_path)
        saved_paths.append(out_path)

        score_strs = ", ".join(f"{s:.2f}" for s in all_scores)
        print(f"    best CLIP: {best_score:.3f}  |  all: [{score_strs}]")

    # ── display grid in notebook environments ─────────────────────────────────
    if saved_paths:
        display_grid(saved_paths)

    print("\nDone! Enhanced images saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
