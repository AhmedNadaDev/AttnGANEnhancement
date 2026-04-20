"""
Attention refinement for AttnGAN — inference-time word-region alignment.

Strategy: Temperature-Scaled Cross-Modal Attention
--------------------------------------------------
Standard AttnGAN computes softmax(Q·K^T) with implicit temperature τ = 1.
Reducing τ sharpens each attention distribution, making image regions
attend more decisively to their best-matching words:

  Standard:          attn = softmax(Q·K^T)
  Temperature-scaled: attn = softmax(Q·K^T / τ),   τ ∈ (0, 1]

Lower τ → peakier distribution → reduced cross-modal misalignment.

Additionally, top-k word masking zeros the bottom (1 - top_k_ratio)
fraction of attention logits per query position, further suppressing
distractors such as function words ("a", "the", "with") that carry
little visual content.

Key design constraint
---------------------
These refinements are hyper-parameter changes applied at inference time.
No new trainable parameters are introduced; the pretrained checkpoint
loads into RefinedG_NET with the same state-dict keys as G_NET, requiring
zero checkpoint modification.

Reference: temperature scaling concept from Hinton et al. (2015),
"Distilling the Knowledge in a Neural Network".
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.kaggle_config import cfg
from src.model_wrapper import (
    AttnGANWrapper,
    CA_NET,
    GET_IMAGE_G,
    GlobalAttentionGeneral,
    INIT_STAGE_G,
    ResBlock,
    upBlock,
)

DEFAULT_TEMPERATURE: float = 0.7   # τ < 1 → sharper attention
DEFAULT_TOP_K_RATIO: float = 0.75  # keep top-75% of words, mask the rest


# ── Refined attention ─────────────────────────────────────────────────────────

class RefinedGlobalAttentionGeneral(GlobalAttentionGeneral):
    """
    Temperature-scaled + top-k cross-modal attention module.

    Drop-in replacement for GlobalAttentionGeneral.  Trainable parameters
    (conv_context, sm) are identical, so pretrained weights transfer
    transparently via load_state_dict().

    Refinement steps applied inside forward():
      1. Divide attention logits by temperature τ before softmax.
      2. Apply padding mask (unchanged from base class).
      3. Zero-out bottom (1 - top_k_ratio) words per query position.
      4. Guard NaN rows from all-masked softmax with uniform fallback.
    """

    def __init__(
        self,
        idf: int,
        cdf: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k_ratio: float = DEFAULT_TOP_K_RATIO,
    ) -> None:
        super().__init__(idf, cdf)
        self.temperature  = temperature
        self.top_k_ratio  = top_k_ratio

    def forward(
        self,
        input: torch.Tensor,    # (B, idf, ih, iw)
        context: torch.Tensor,  # (B, cdf, sourceL)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ih, iw     = input.size(2), input.size(3)
        queryL     = ih * iw
        batch_size = context.size(0)
        sourceL    = context.size(2)

        # Flatten spatial dims: (B, idf, queryL)
        target  = input.view(batch_size, -1, queryL)
        targetT = target.transpose(1, 2).contiguous()           # (B, queryL, idf)

        # Project word embeddings to image feature space: (B, idf, sourceL)
        sourceT = self.conv_context(context.unsqueeze(3)).squeeze(3)

        # Raw attention scores: (B, queryL, sourceL) → (B*queryL, sourceL)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)

        # ── 1. Temperature scaling ─────────────────────────────────────────────
        attn = attn / self.temperature

        # ── 2. Padding mask ────────────────────────────────────────────────────
        if self.mask is not None:
            mask_expanded = self.mask.repeat(queryL, 1)
            attn = attn.masked_fill(mask_expanded, -float("inf"))

        # ── 3. Top-k word masking ──────────────────────────────────────────────
        if self.top_k_ratio < 1.0 and sourceL > 1:
            k = max(1, int(sourceL * self.top_k_ratio))
            # find the k-th largest value per query position
            topk_vals, _ = attn.topk(k, dim=1, largest=True, sorted=False)
            threshold    = topk_vals.min(dim=1, keepdim=True).values
            attn         = attn.masked_fill(attn < threshold, -float("inf"))

        attn = self.sm(attn)  # (B*queryL, sourceL)

        # ── 4. NaN guard (all-masked rows become uniform) ──────────────────────
        nan_rows = torch.isnan(attn).any(dim=1, keepdim=True)
        if nan_rows.any():
            uniform = torch.full_like(attn, 1.0 / sourceL)
            attn    = torch.where(nan_rows.expand_as(attn), uniform, attn)

        attn = attn.view(batch_size, queryL, sourceL)
        attn = attn.transpose(1, 2).contiguous()          # (B, sourceL, queryL)

        # Compute weighted context and reshape back to spatial
        weightedContext = torch.bmm(sourceT, attn)        # (B, idf, queryL)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn            = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


# ── Refined generator stages ──────────────────────────────────────────────────

class RefinedNEXT_STAGE_G(nn.Module):
    """
    Subsequent generator stage using RefinedGlobalAttentionGeneral.

    Parameter names (att.conv_context.weight, residual.*, upsample.*)
    are identical to the original NEXT_STAGE_G, ensuring clean weight transfer.
    """

    def __init__(
        self,
        ngf: int,
        nef: int,
        ncf: int,  # ncf unused here but kept for API symmetry with original
        temperature: float = DEFAULT_TEMPERATURE,
        top_k_ratio: float = DEFAULT_TOP_K_RATIO,
    ) -> None:
        super().__init__()
        self.att      = RefinedGlobalAttentionGeneral(ngf, nef, temperature, top_k_ratio)
        self.residual = nn.Sequential(
            *[ResBlock(ngf * 2) for _ in range(cfg.GAN.R_NUM)]
        )
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(
        self,
        h_code:    torch.Tensor,
        c_code:    torch.Tensor,
        word_embs: torch.Tensor,
        mask:      torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.att.applyMask(mask)
        c_code_att, att = self.att(h_code, word_embs)
        h_c_code        = torch.cat((h_code, c_code_att), dim=1)
        out_code        = self.residual(h_c_code)
        out_code        = self.upsample(out_code)
        return out_code, att


class RefinedG_NET(nn.Module):
    """
    Multi-stage AttnGAN generator with temperature-scaled attention.

    Structurally identical to G_NET (same module names, same parameter
    shapes).  The only functional difference is that NEXT_STAGE_G blocks
    use RefinedNEXT_STAGE_G with sharpened attention.

    Pretrained weights from bird_AttnGAN2.pth load without remapping.
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k_ratio: float = DEFAULT_TOP_K_RATIO,
    ) -> None:
        super().__init__()
        ngf = cfg.GAN.GF_DIM          # 32
        nef = cfg.TEXT.EMBEDDING_DIM   # 256
        ncf = cfg.GAN.CONDITION_DIM    # 100

        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1   = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)

        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2   = RefinedNEXT_STAGE_G(ngf, nef, ncf, temperature, top_k_ratio)
            self.img_net2 = GET_IMAGE_G(ngf)

        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3   = RefinedNEXT_STAGE_G(ngf, nef, ncf, temperature, top_k_ratio)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(
        self,
        z_code:    torch.Tensor,
        sent_emb:  torch.Tensor,
        word_embs: torch.Tensor,
        mask:      torch.Tensor,
    ) -> tuple[list, list, torch.Tensor, torch.Tensor]:
        fake_imgs: list = []
        att_maps:  list = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_imgs.append(self.img_net1(h_code1))

        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_imgs.append(self.img_net2(h_code2))
            att_maps.append(att1)

        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_imgs.append(self.img_net3(h_code3))
            att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


# ── High-level wrapper ────────────────────────────────────────────────────────

class RefinedAttnGANWrapper(AttnGANWrapper):
    """
    AttnGANWrapper variant that loads weights into RefinedG_NET.

    All other components (text encoder, vocabulary, tokenisation,
    generate() logic) are inherited from AttnGANWrapper unchanged.

    The only difference visible to callers is that the generator uses
    temperature-scaled attention, producing sharper word-region alignment.

    Parameters
    ----------
    model_dir   : path to directory containing checkpoint files.
    temperature : attention softmax temperature τ (default 0.7).
    top_k_ratio : fraction of words kept per query position (default 0.75).
    device      : torch.device; auto-detected when None.
    """

    def __init__(
        self,
        model_dir:   str,
        temperature: float                  = DEFAULT_TEMPERATURE,
        top_k_ratio: float                  = DEFAULT_TOP_K_RATIO,
        device:      torch.device | None    = None,
    ) -> None:
        # Must be set before super().__init__() calls _load_generator()
        self._temperature = temperature
        self._top_k_ratio = top_k_ratio
        super().__init__(model_dir, device)

    def _load_generator(self, path: str) -> RefinedG_NET:
        model = RefinedG_NET(
            temperature=self._temperature,
            top_k_ratio=self._top_k_ratio,
        )
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device).eval()
        print(
            f"[RefinedAttnGAN] Generator loaded  "
            f"tau={self._temperature}, top_k={self._top_k_ratio}  "
            f"from {path}"
        )
        return model
