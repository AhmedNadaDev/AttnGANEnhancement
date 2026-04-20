"""
BERT-based text encoder compatible with AttnGAN generator inputs.

Outputs:
    words_emb: (B, EMBEDDING_DIM, seq_len)
    sent_emb : (B, EMBEDDING_DIM)
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BERTTextEncoder(nn.Module):
    """Drop-in text encoder replacement for AttnGAN."""

    def __init__(
        self,
        ixtoword: Dict[int, str],
        embedding_dim: int = 256,
        model_name: str = "bert-base-uncased",
        max_words: int = 25,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.ixtoword = ixtoword
        self.embedding_dim = embedding_dim
        self.max_words = max_words

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.last_padding_mask: torch.Tensor | None = None

    def init_hidden(self, bsz: int):
        """Kept for compatibility with RNN encoder interface."""
        return None

    def _indices_to_sentences(self, captions: torch.Tensor) -> List[str]:
        sentences: List[str] = []
        for row in captions.detach().cpu().tolist():
            words: List[str] = []
            for idx in row:
                if idx == 0:
                    break
                words.append(self.ixtoword.get(int(idx), ""))
            text = " ".join(w for w in words if w).strip()
            sentences.append(text if text else "bird")
        return sentences

    def _tokenize_text(self, texts: Iterable[str], device: torch.device):
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_words + 2,  # +2 for [CLS] and [SEP]
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(self, captions, cap_lens=None, hidden=None, mask=None):
        device = captions.device
        texts = self._indices_to_sentences(captions)
        tokenized = self._tokenize_text(texts, device)

        outputs = self.bert(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        )
        last_hidden = outputs.last_hidden_state  # (B, L, 768)

        # Remove [CLS] and [SEP] positions to align with word-level attention.
        token_hidden = last_hidden[:, 1 : self.max_words + 1, :]
        token_mask = tokenized["attention_mask"][:, 1 : self.max_words + 1].bool()

        projected = self.dropout(self.proj(token_hidden))  # (B, seq_len, EMBEDDING_DIM)
        words_emb = projected.transpose(1, 2).contiguous()  # (B, EMBEDDING_DIM, seq_len)

        # Mean pool over valid tokens for sentence embedding.
        valid = token_mask.unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        sent_emb = (projected * valid).sum(dim=1) / denom  # (B, EMBEDDING_DIM)

        # True marks padding positions (same convention as original mask).
        self.last_padding_mask = ~token_mask

        return words_emb, sent_emb
