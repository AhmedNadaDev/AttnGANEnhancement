"""
Full three-stage enhanced AttnGAN pipeline.

Stage 1 — Attention Refinement
    RefinedAttnGANWrapper with temperature-scaled cross-modal attention
    (τ = 0.7) and top-k word masking (top_k = 0.75).

Stage 2 — CLIP-Guided Reranking
    Generate `num_candidates` independent samples from different noise seeds.
    Score each with CLIP ViT-B/32 and select the highest-scoring candidate.

Stage 3 — Post-Processing Enhancement
    Apply the ImageEnhancer pipeline (Lanczos upscaling to 512×512, Gaussian
    denoising, UnsharpMask sharpening, contrast and saturation boost).

Usage
-----
>>> wrapper = EnhancedAttnGANWrapper("/path/to/model_dir")
>>> image   = wrapper.generate("a red bird with blue wings")
>>> image.save("enhanced_bird.png")
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.attention_refinement import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K_RATIO,
    RefinedAttnGANWrapper,
)
from src.clip_scorer import CLIPScorer
from src.enhancer import ImageEnhancer


class EnhancedAttnGANWrapper:
    """
    Three-stage enhanced text-to-image pipeline built on top of AttnGAN.

    Parameters
    ----------
    model_dir      : Directory containing AttnGAN checkpoint files.
    num_candidates : Number of noise samples to generate per prompt.
                     More candidates → higher CLIP-selected quality
                     but proportionally longer inference time.
                     Recommended: 4–8.
    temperature    : Attention softmax temperature τ ∈ (0, 1].
                     Lower = sharper word-region alignment (default 0.7).
    top_k_ratio    : Fraction of words kept in top-k masking (default 0.75).
    device         : Torch device; auto-detected when None.

    Example
    -------
    >>> w = EnhancedAttnGANWrapper("data/", num_candidates=6)
    >>> img, score, all_scores = w.generate_with_scores("a yellow bird on a branch")
    """

    def __init__(
        self,
        model_dir:      str,
        num_candidates: int                  = 6,
        temperature:    float                = DEFAULT_TEMPERATURE,
        top_k_ratio:    float                = DEFAULT_TOP_K_RATIO,
        device:         Optional[torch.device] = None,
    ) -> None:
        self.num_candidates = num_candidates

        print("=" * 60)
        print("  Enhanced AttnGAN — initialising components")
        print("=" * 60)

        # Stage 1: attention-refined generator
        self.attngan = RefinedAttnGANWrapper(
            model_dir,
            temperature=temperature,
            top_k_ratio=top_k_ratio,
            device=device,
        )
        self.device = self.attngan.device

        # Stage 2: CLIP scorer for candidate reranking
        self.clip_scorer = CLIPScorer(device=self.device)

        # Stage 3: post-processing enhancer
        self.enhancer = ImageEnhancer()

        print("=" * 60)
        print(
            f"  Ready — {num_candidates} candidates · "
            f"tau={temperature} · CLIP={self.clip_scorer.backend} · "
            f"enhance={self.enhancer}"
        )
        print("=" * 60)

    # ── candidate generation ──────────────────────────────────────────────────

    def generate_candidates(
        self,
        text:      str,
        base_seed: int = 100,
    ) -> List[Image.Image]:
        """
        Draw `num_candidates` independent 256×256 samples from RefinedAttnGAN.

        Each candidate uses a different random seed so the noise vectors
        are diverse.  All other inputs (text encoding, attention weights)
        are identical across candidates.
        """
        candidates: List[Image.Image] = []
        for i in range(self.num_candidates):
            seed = base_seed + i * 37          # spread seeds to avoid correlation
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            img = self.attngan.generate(text, copies=2)
            candidates.append(img)
        return candidates

    # ── full pipeline ─────────────────────────────────────────────────────────

    def generate(
        self,
        text:       str,
        base_seed:  int  = 100,
        return_all: bool = False,
    ) -> "Image.Image | tuple[Image.Image, List[Image.Image], List[float], int]":
        """
        Run the full three-stage pipeline for one text prompt.

        Parameters
        ----------
        text       : Natural-language description.
        base_seed  : Base seed for candidate diversity.
        return_all : If True, also returns candidates, CLIP scores, best index.

        Returns
        -------
        Enhanced PIL Image (512×512), or a 4-tuple when return_all=True:
            (enhanced_image, candidates, all_scores, best_idx)
        """
        # Stage 1 + 2: generate diverse candidates, CLIP-rerank
        candidates = self.generate_candidates(text, base_seed=base_seed)
        best_raw, scores, best_idx = self.clip_scorer.rerank(candidates, text)

        # Stage 3: post-process the winner
        enhanced = self.enhancer.enhance(best_raw)

        if return_all:
            return enhanced, candidates, scores, best_idx
        return enhanced

    def generate_with_scores(
        self,
        text:      str,
        base_seed: int = 100,
    ) -> Tuple[Image.Image, float, List[float]]:
        """
        Convenience wrapper: returns (enhanced_image, best_clip_score, all_scores).

        The reported CLIP score is measured on the raw 256×256 best candidate
        *before* post-processing, making it directly comparable with the
        baseline CLIP score (which is also computed on a raw 256×256 image).
        """
        enhanced, candidates, scores, best_idx = self.generate(
            text, base_seed=base_seed, return_all=True
        )
        return enhanced, float(scores[best_idx]), scores

    # ── meta ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EnhancedAttnGANWrapper("
            f"candidates={self.num_candidates}, "
            f"clip={self.clip_scorer.backend}, "
            f"device={self.device})"
        )
