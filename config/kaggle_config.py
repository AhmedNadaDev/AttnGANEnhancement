"""
Kaggle AttnGAN — central configuration.

All architecture hyper-parameters match the publicly released bird checkpoint
(bird_AttnGAN2.pth + text_encoder200.pth from the original paper).

Edit the PATH block at the top of main_kaggle.py to point to your dataset;
the rest of this file should not need to change for the default bird model.
"""

import torch
from easydict import EasyDict as edict

# ─── Root config object ──────────────────────────────────────────────────────
cfg = edict()

# ─── Dataset ─────────────────────────────────────────────────────────────────
cfg.DATASET_NAME = "birds"

# ─── Device (overridden at runtime in main_kaggle.py) ────────────────────────
cfg.CUDA = torch.cuda.is_available()

# ─── RNN type ────────────────────────────────────────────────────────────────
cfg.RNN_TYPE = "LSTM"

# ─── Generator tree ──────────────────────────────────────────────────────────
cfg.TREE = edict()
cfg.TREE.BRANCH_NUM = 3   # produces 64 / 128 / 256 px images
cfg.TREE.BASE_SIZE = 64

# ─── Training flags (inference-only; do not change) ──────────────────────────
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = False
cfg.TRAIN.B_NET_D = False
cfg.TRAIN.BATCH_SIZE = 2

# Model file names relative to MODEL_DIR (set in main_kaggle.py).
# Defaults match the bird checkpoint layout.
cfg.TRAIN.NET_E = "text_encoder200.pth"
cfg.TRAIN.NET_G = "bird_AttnGAN2.pth"

# ─── GAN architecture ────────────────────────────────────────────────────────
cfg.GAN = edict()
cfg.GAN.GF_DIM = 32          # generator feature-map base width
cfg.GAN.DF_DIM = 64          # discriminator feature-map base width (unused for inference)
cfg.GAN.Z_DIM = 100          # noise vector dimension
cfg.GAN.CONDITION_DIM = 100  # conditioning augmentation output dim
cfg.GAN.R_NUM = 2            # residual blocks per NEXT_STAGE_G
cfg.GAN.B_ATTENTION = True
cfg.GAN.B_DCGAN = False      # use multi-stage G_NET (not single-stage G_DCGAN)

# ─── Text encoding ───────────────────────────────────────────────────────────
cfg.TEXT = edict()
cfg.TEXT.EMBEDDING_DIM = 256  # RNN hidden size (total, both directions)
cfg.TEXT.WORDS_NUM = 25       # max caption length
cfg.TEXT.CAPTIONS_PER_IMAGE = 10
cfg.TEXT.ENCODER_TYPE = "rnn"  # "rnn" (default) or "bert"
cfg.TEXT.BERT_MODEL = "bert-base-uncased"

# ─── Paths (set / overridden in main_kaggle.py) ──────────────────────────────
# Local default: project ./data and ./outputs (see main_kaggle.py).
# Kaggle example: MODEL_DIR = "/kaggle/input/attngan-pretrained"
cfg.MODEL_DIR = "/kaggle/input/attngan-pretrained"

# Vocabulary pickle produced during AttnGAN dataset preprocessing.
# Layout: [train_captions, test_captions, ixtoword, wordtoix]
cfg.CAPTIONS_PICKLE = "captions.pickle"

# Directory where generated images are written (main_kaggle.py sets this).
cfg.OUTPUT_DIR = "/kaggle/working/outputs"
