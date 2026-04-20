"""
CLIP-based image-text scoring for AttnGAN candidate reranking.

Two backends are tried in order of preference:
  1. openai-clip   — pip install openai-clip  (or git+https://github.com/openai/CLIP)
  2. transformers  — pip install transformers  (HuggingFace CLIP ViT-B/32)

Falls back to a Laplacian-variance sharpness heuristic if neither loads,
so the pipeline degrades gracefully in offline Kaggle kernels.

Typical CLIP cosine-similarity scores (×100) for bird images:
  poor alignment  : 15–20
  decent alignment: 22–27
  good alignment  : 28+
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple


class CLIPScorer:
    """
    Image-text similarity scorer backed by CLIP ViT-B/32.

    Primary purpose: rerank multiple AttnGAN noise samples and select
    the candidate that best matches the conditioning text prompt.

    Usage
    -----
    >>> scorer = CLIPScorer()
    >>> score  = scorer.score(pil_image, "a red bird with blue wings")
    >>> best, scores, idx = scorer.rerank([img1, img2, img3], text)
    """

    _OPENAI_MODEL = "ViT-B/32"
    _HF_MODEL     = "openai/clip-vit-base-patch32"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device: torch.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.backend: str = "none"
        self._init_backend()

    # ── backend initialisation ────────────────────────────────────────────────

    def _init_backend(self) -> None:
        if self._try_openai_clip():
            return
        if self._try_hf_clip():
            return
        self.backend = "heuristic"
        print(
            "[CLIPScorer] WARNING: No CLIP model found. "
            "Falling back to sharpness heuristic. "
            "Install 'openai-clip' or 'transformers' for real CLIP scoring."
        )

    def _try_openai_clip(self) -> bool:
        try:
            import clip  # type: ignore
            model, preprocess = clip.load(self._OPENAI_MODEL, device=self.device)
            model.eval()
            self._clip      = clip
            self._model     = model
            self._preprocess = preprocess
            self.backend    = "openai"
            print(f"[CLIPScorer] openai-clip ({self._OPENAI_MODEL}) on {self.device}")
            return True
        except Exception:
            return False

    def _try_hf_clip(self) -> bool:
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
            hf_model     = CLIPModel.from_pretrained(self._HF_MODEL).to(self.device)
            hf_processor = CLIPProcessor.from_pretrained(self._HF_MODEL)
            hf_model.eval()
            self._hf_model     = hf_model
            self._hf_processor = hf_processor
            self.backend       = "transformers"
            print(f"[CLIPScorer] transformers CLIP ({self._HF_MODEL}) on {self.device}")
            return True
        except Exception as exc:
            print(f"[CLIPScorer] transformers unavailable: {exc}")
            return False

    # ── scoring ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        """
        Compute image-text similarity.

        Returns cosine similarity × 100 for CLIP backends, or a
        sharpness proxy value for the heuristic fallback.
        Higher always means better.
        """
        if self.backend == "openai":
            return self._score_openai(image, text)
        if self.backend == "transformers":
            return self._score_hf(image, text)
        return self._score_heuristic(image)

    def _score_openai(self, image: Image.Image, text: str) -> float:
        img_t = self._preprocess(image).unsqueeze(0).to(self.device)
        txt_t = self._clip.tokenize([text], truncate=True).to(self.device)
        img_f = self._model.encode_image(img_t)
        txt_f = self._model.encode_text(txt_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return float((img_f @ txt_f.T).item() * 100)

    def _score_hf(self, image: Image.Image, text: str) -> float:
        inputs = self._hf_processor(
            text=[text], images=image, return_tensors="pt", padding=True
        ).to(self.device)
        out = self._hf_model(**inputs)
        # logits_per_image already scaled by CLIP's learned temperature (~100)
        return float(out.logits_per_image[0, 0].item())

    @staticmethod
    def _score_heuristic(image: Image.Image) -> float:
        """
        Laplacian gradient variance as a sharpness proxy.
        Higher variance = sharper image = used as quality surrogate
        when no CLIP model is available.
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        gy   = np.diff(gray, axis=0)
        gx   = np.diff(gray, axis=1)
        return float(np.var(gy) + np.var(gx))

    # ── batch helpers ─────────────────────────────────────────────────────────

    def score_batch(
        self,
        images: List[Image.Image],
        text: str,
    ) -> List[float]:
        """Score every image in *images* against the same *text* prompt."""
        return [self.score(img, text) for img in images]

    def rerank(
        self,
        images: List[Image.Image],
        text: str,
    ) -> Tuple[Image.Image, List[float], int]:
        """
        Select the highest-scoring image from *images*.

        Returns
        -------
        best_image  : the selected PIL Image
        all_scores  : list of scores for every candidate
        best_index  : index into *images* of the winner
        """
        if not images:
            raise ValueError("images list must not be empty")
        scores   = self.score_batch(images, text)
        best_idx = int(np.argmax(scores))
        return images[best_idx], scores, best_idx

    # ── meta ──────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when a real CLIP model (not the heuristic) is active."""
        return self.backend in ("openai", "transformers")

    def __repr__(self) -> str:
        return f"CLIPScorer(backend={self.backend!r}, device={self.device})"
