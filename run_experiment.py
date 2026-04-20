"""
run_experiment.py — Controlled baseline vs. enhanced comparison experiment.

Experiment design
-----------------
  Step 1  Baseline  — run original AttnGAN on all prompts (seed=42, 1 sample)
                       save to outputs/baseline/
  Step 2  Enhanced  — run three-stage pipeline on same prompts
                       (6 CLIP-ranked candidates, τ=0.7, 512×512 output)
                       save to outputs/enhanced/
  Step 3  Score     — compute CLIP ViT-B/32 scores for all baseline images
                       (enhanced pipeline already stores CLIP scores internally)
  Step 4  Report    — write comparison_grid.png + comparison_report.md

Usage
-----
    python run_experiment.py

For Kaggle:
    %run AttnGANEnhancement/run_experiment.py

Adjust MODEL_DIR and OUTPUT_ROOT to match your environment.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── PATH CONFIGURATION ────────────────────────────────────────────────────────

MODEL_DIR   = os.path.join(_HERE, "data")
OUTPUT_ROOT = os.path.join(_HERE, "outputs")

# Kaggle overrides:
# MODEL_DIR   = "/kaggle/input/attngan-pretrained"
# OUTPUT_ROOT = "/kaggle/working/outputs"

TEXT_ENCODER_FILE = "text_encoder200.pth"
GENERATOR_FILE    = "bird_AttnGAN2.pth"
VOCAB_FILE        = "captions.pickle"

# ── EXPERIMENT SETTINGS ───────────────────────────────────────────────────────

BASELINE_SEED  = 42    # fixed seed for reproducible baseline
ENHANCED_SEED  = 100   # base seed for enhanced candidate diversity
NUM_CANDIDATES = 6     # CLIP-ranked candidates per prompt
ATTN_TEMP      = 0.7   # attention temperature τ
TOP_K_RATIO    = 0.75  # top-k word masking ratio

# ── STANDARDISED TEST PROMPTS (12 prompts, all CUB-200 domain) ───────────────

EXPERIMENT_TEXTS: List[str] = [
    "a small red bird with blue wings and a yellow beak",
    "a yellow bird sitting on a branch with green leaves",
    "this bird has a white belly and black crown with orange breast",
    "a large bird with a long orange beak and blue feathers",
    "a tiny bird with a red head and grey body perched on a twig",
    "a bird with bright purple feathers and a white pointed beak",
    "a small brown bird with a spotted chest and a long tail",
    "this bird has green feathers on its back and a yellow belly",
    "a blue bird with white chest and black markings on its wings",
    "a hummingbird with iridescent red and green plumage",
    "a black and white bird with a bright red cap on its head",
    "a large orange bird with long legs standing near the water",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_cfg(model_dir: str) -> None:
    from config.kaggle_config import cfg
    cfg.MODEL_DIR       = model_dir
    cfg.CAPTIONS_PICKLE = VOCAB_FILE
    cfg.TRAIN.NET_E     = TEXT_ENCODER_FILE
    cfg.TRAIN.NET_G     = GENERATOR_FILE


def _check_files(model_dir: str) -> None:
    missing = [
        p
        for p in [
            os.path.join(model_dir, VOCAB_FILE),
            os.path.join(model_dir, TEXT_ENCODER_FILE),
            os.path.join(model_dir, GENERATOR_FILE),
        ]
        if not os.path.isfile(p)
    ]
    if missing:
        print("[ERROR] Missing model files:")
        for p in missing:
            print(f"  {p}")
        print(
            "\nUpdate MODEL_DIR at the top of run_experiment.py "
            "to point to your pretrained weights.\n"
        )
        sys.exit(1)


def _banner(title: str) -> None:
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ── experiment stages ─────────────────────────────────────────────────────────

def run_baseline(output_dir: str) -> List[Dict[str, Any]]:
    """
    Generate one baseline image per prompt with the original AttnGAN.

    Seed is fixed to BASELINE_SEED for full reproducibility.
    Returns a list of result dicts (text, path, image).
    """
    from src.model_wrapper import AttnGANWrapper
    from src.utils import mkdir_p, save_image

    _banner("STAGE 1 - Baseline AttnGAN")

    mkdir_p(output_dir)
    wrapper = AttnGANWrapper(MODEL_DIR)
    results: List[Dict[str, Any]] = []

    for i, text in enumerate(EXPERIMENT_TEXTS, start=1):
        print(f"  [{i:2d}/{len(EXPERIMENT_TEXTS)}] {text!r}")
        torch.manual_seed(BASELINE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(BASELINE_SEED)

        img  = wrapper.generate(text, copies=2)
        path = os.path.join(output_dir, f"baseline_{i:03d}.png")
        save_image(img, path)
        results.append({"text": text, "path": path, "image": img})

    print(f"\n  Baseline done — {len(results)} images in {output_dir}")
    return results


def run_enhanced(output_dir: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Run the three-stage enhanced pipeline for all prompts.

    Returns (clip_scorer_instance, results).
    The clip_scorer is returned so it can be reused to score baseline images
    without loading a second copy of CLIP.
    """
    from src.enhanced_pipeline import EnhancedAttnGANWrapper
    from src.utils import mkdir_p, save_image

    _banner("STAGE 2 - Enhanced AttnGAN Pipeline")

    mkdir_p(output_dir)
    wrapper = EnhancedAttnGANWrapper(
        MODEL_DIR,
        num_candidates=NUM_CANDIDATES,
        temperature=ATTN_TEMP,
        top_k_ratio=TOP_K_RATIO,
    )
    results: List[Dict[str, Any]] = []

    for i, text in enumerate(EXPERIMENT_TEXTS, start=1):
        print(f"  [{i:2d}/{len(EXPERIMENT_TEXTS)}] {text!r}")
        enhanced, best_score, all_scores = wrapper.generate_with_scores(
            text, base_seed=ENHANCED_SEED
        )
        path = os.path.join(output_dir, f"enhanced_{i:03d}.png")
        save_image(enhanced, path)
        results.append(
            {
                "text":       text,
                "path":       path,
                "image":      enhanced,
                "clip_score": best_score,
                "all_scores": all_scores,
            }
        )
        score_strs = ", ".join(f"{s:.2f}" for s in all_scores)
        print(f"    best CLIP: {best_score:.3f}  |  all: [{score_strs}]")

    print(f"\n  Enhanced done — {len(results)} images in {output_dir}")
    return wrapper.clip_scorer, results


def score_baseline(
    baseline_results: List[Dict[str, Any]],
    clip_scorer: Any,
) -> List[Dict[str, Any]]:
    """
    Add CLIP scores to *baseline_results* using the already-loaded scorer.

    Reuses the scorer instance from run_enhanced() to avoid loading CLIP twice.
    """
    _banner("STAGE 3 - Scoring baseline images with CLIP")

    for i, r in enumerate(baseline_results, start=1):
        score = clip_scorer.score(r["image"], r["text"])
        r["clip_score"] = score
        print(f"  [{i:2d}] CLIP = {score:.3f}  |  {r['text'][:55]!r}")

    return baseline_results


def build_comparison(
    baseline_results: List[Dict[str, Any]],
    enhanced_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge per-prompt baseline and enhanced results into a comparison list."""
    return [
        {
            "text":           b["text"],
            "baseline_path":  b["path"],
            "enhanced_path":  e["path"],
            "baseline_image": b["image"],
            "enhanced_image": e["image"],
            "baseline_clip":  b["clip_score"],
            "enhanced_clip":  e["clip_score"],
        }
        for b, e in zip(baseline_results, enhanced_results)
    ]


def print_summary(comparison: List[Dict[str, Any]]) -> None:
    b_scores  = [c["baseline_clip"] for c in comparison]
    e_scores  = [c["enhanced_clip"]  for c in comparison]
    deltas    = [e - b for b, e in zip(b_scores, e_scores)]
    improved  = sum(1 for d in deltas if d > 0)

    _banner("EXPERIMENT COMPLETE")
    print(f"  Baseline mean CLIP  : {np.mean(b_scores):.3f}")
    print(f"  Enhanced mean CLIP  : {np.mean(e_scores):.3f}")
    print(f"  Mean improvement    : {np.mean(deltas):+.3f}")
    print(f"  Std of improvement  : +/-{np.std(deltas):.3f}")
    print(f"  Prompts improved    : {improved}/{len(comparison)}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _apply_cfg(MODEL_DIR)
    _check_files(MODEL_DIR)

    baseline_dir = os.path.join(OUTPUT_ROOT, "baseline")
    enhanced_dir = os.path.join(OUTPUT_ROOT, "enhanced")

    # ── run the two experiment arms ───────────────────────────────────────────
    baseline_results = run_baseline(baseline_dir)
    clip_scorer, enhanced_results = run_enhanced(enhanced_dir)
    baseline_results = score_baseline(baseline_results, clip_scorer)

    # ── merge results ─────────────────────────────────────────────────────────
    comparison = build_comparison(baseline_results, enhanced_results)

    # ── generate outputs ──────────────────────────────────────────────────────
    from src.comparison import create_comparison_grid, generate_markdown_report

    grid_path   = os.path.join(OUTPUT_ROOT, "comparison_grid.png")
    report_path = os.path.join(OUTPUT_ROOT, "comparison_report.md")

    _banner("STAGE 4 - Generating comparison outputs")

    create_comparison_grid(
        baseline_paths  = [c["baseline_path"] for c in comparison],
        enhanced_paths  = [c["enhanced_path"]  for c in comparison],
        texts           = [c["text"]           for c in comparison],
        output_path     = grid_path,
        baseline_scores = [c["baseline_clip"]  for c in comparison],
        enhanced_scores = [c["enhanced_clip"]  for c in comparison],
    )

    generate_markdown_report(
        comparison_results = comparison,
        output_path        = report_path,
        clip_available     = clip_scorer.available,
    )

    # ── display grid in notebook environments ─────────────────────────────────
    try:
        from IPython.display import display as ipy_display  # type: ignore
        from PIL import Image as _PIL_Image
        ipy_display(_PIL_Image.open(grid_path))
    except Exception:
        pass

    print_summary(comparison)
    print(f"  Report   -> {report_path}")
    print(f"  Grid     -> {grid_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
