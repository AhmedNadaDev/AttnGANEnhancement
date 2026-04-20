"""
Utility functions for AttnGAN Kaggle inference.

Covers:
- Text tokenization (NLTK RegexpTokenizer, matching training pre-processing)
- Vocabulary helpers
- Image saving
"""

import os
import sys
import numpy as np
from PIL import Image

# Project root on path for sibling imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─── NLTK bootstrap ──────────────────────────────────────────────────────────

def _ensure_nltk() -> None:
    """Download required NLTK data on first use (silent in Kaggle)."""
    import nltk
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


# ─── Text tokenization ───────────────────────────────────────────────────────

def tokenize_caption(
    caption: str,
    wordtoix: dict,
    words_num: int,
    copies: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenize *caption* with NLTK RegexpTokenizer (matching training pre-processing),
    map tokens to vocabulary indices, and return a batch of identical copies.

    Args:
        caption:  Raw text string.
        wordtoix: Word-to-index dictionary from captions.pickle.
        words_num: Maximum caption length (cfg.TEXT.WORDS_NUM).
        copies:   How many identical copies to pack into the batch.

    Returns:
        captions  : int64 array of shape (copies, cap_len)
        cap_lens  : int64 array of shape (copies,) filled with cap_len

    Raises:
        ValueError if no known words are found in the caption.
    """
    _ensure_nltk()
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(caption.lower())

    # Strip non-ASCII characters, drop empty tokens and OOV words
    cap_v = []
    for tok in tokens:
        tok = tok.encode("ascii", "ignore").decode("ascii")
        if tok and tok in wordtoix:
            cap_v.append(wordtoix[tok])
        if len(cap_v) >= words_num:
            break

    if not cap_v:
        raise ValueError(
            f"No known words found in caption: '{caption}'. "
            "Check that your captions.pickle matches the pretrained model."
        )

    cap_len = len(cap_v)
    captions = np.zeros((copies, cap_len), dtype=np.int64)
    for i in range(copies):
        captions[i] = np.array(cap_v, dtype=np.int64)
    cap_lens = np.full(copies, cap_len, dtype=np.int64)

    return captions, cap_lens


def oov_ratio(caption: str, wordtoix: dict) -> float:
    """Return the fraction of words in *caption* that are out-of-vocabulary."""
    _ensure_nltk()
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = [t.lower() for t in tokenizer.tokenize(caption)]
    if not tokens:
        return 1.0
    oov = sum(1 for t in tokens if t not in wordtoix)
    return oov / len(tokens)


# ─── Image utilities ─────────────────────────────────────────────────────────

def tensor_to_pil(tensor) -> Image.Image:
    """
    Convert a generator output tensor (C, H, W) in range [-1, 1] to a PIL Image.
    """
    im = tensor.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = np.clip(im, 0, 255).astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))  # CHW → HWC
    return Image.fromarray(im)


def save_image(image: Image.Image, path: str) -> None:
    """Save a PIL image to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)
    print(f"  Saved -> {path}")


def mkdir_p(path: str) -> None:
    """Recursively create *path* if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ─── Vocab helpers ───────────────────────────────────────────────────────────

def describe_vocab(wordtoix: dict, ixtoword: dict) -> None:
    """Print a brief summary of the vocabulary."""
    print(f"Vocabulary: {len(wordtoix)} words")
    sample = list(wordtoix.keys())[:10]
    print(f"  Sample words: {sample}")
