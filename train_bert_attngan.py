"""
Train a BERT-based text encoder for AttnGAN compatibility.

This script distills the existing RNN text encoder embedding space into BERT so
the generator can be reused without changing architecture.

Outputs:
    - bert_text_encoder.pth
    - bert_AttnGAN_generator.pth

Run:
    python train_bert_attngan.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config.kaggle_config import cfg
from src.bert_text_encoder import BERTTextEncoder
from src.model_wrapper import G_NET, RNN_ENCODER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_captions_pickle(pickle_path: str):
    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    train_caps, test_caps, ixtoword, wordtoix = data
    all_caps = list(train_caps) + list(test_caps)
    return all_caps, ixtoword, wordtoix


def prepare_caption_batch(captions: Sequence[Sequence[int]], max_words: int, device: torch.device):
    lengths = [min(len(c), max_words) for c in captions]
    max_len = max(max(lengths), 1)

    arr = np.zeros((len(captions), max_len), dtype=np.int64)
    for i, cap in enumerate(captions):
        trimmed = list(cap[:max_words])
        if trimmed:
            arr[i, : len(trimmed)] = np.asarray(trimmed, dtype=np.int64)

    cap_tensor = torch.LongTensor(arr).to(device)
    len_tensor = torch.LongTensor(lengths).to(device)
    return cap_tensor, len_tensor


def iterate_minibatches(items: Sequence[Sequence[int]], batch_size: int) -> Iterable[List[Sequence[int]]]:
    idxs = np.random.permutation(len(items))
    for start in range(0, len(idxs), batch_size):
        part = idxs[start : start + batch_size]
        yield [items[i] for i in part]


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.CUDA = device.type == "cuda"
    cfg.TEXT.ENCODER_TYPE = "bert"

    model_dir = args.model_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    captions_path = os.path.join(model_dir, args.captions_pickle)
    teacher_path = os.path.join(model_dir, args.teacher_text_encoder)
    generator_path = os.path.join(model_dir, args.generator)

    all_captions, ixtoword, wordtoix = load_captions_pickle(captions_path)
    print(f"[Train] Loaded {len(all_captions)} captions")

    teacher = RNN_ENCODER(len(wordtoix), nhidden=cfg.TEXT.EMBEDDING_DIM).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = BERTTextEncoder(
        ixtoword=ixtoword,
        embedding_dim=cfg.TEXT.EMBEDDING_DIM,
        model_name=cfg.TEXT.BERT_MODEL,
        max_words=cfg.TEXT.WORDS_NUM,
        freeze_bert=args.freeze_bert,
    ).to(device)
    student.train()

    netG = G_NET().to(device)
    netG.load_state_dict(torch.load(generator_path, map_location="cpu"))
    netG.eval()  # kept intact for compatibility

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_steps = 0

        for batch_caps in iterate_minibatches(all_captions, args.batch_size):
            captions, cap_lens = prepare_caption_batch(batch_caps, cfg.TEXT.WORDS_NUM, device)

            with torch.no_grad():
                teacher_hidden = teacher.init_hidden(captions.size(0))
                teacher_words, teacher_sent = teacher(captions, cap_lens, teacher_hidden)

            student_words, student_sent = student(captions, cap_lens, None)

            max_t = min(student_words.size(2), teacher_words.size(2))
            sw = student_words[:, :, :max_t]
            tw = teacher_words[:, :, :max_t]

            token_mask = (~student.last_padding_mask[:, :max_t]).float().unsqueeze(1)
            word_loss = ((sw - tw) ** 2 * token_mask).sum() / token_mask.sum().clamp(min=1.0)
            sent_loss = mse(student_sent, teacher_sent)
            loss = sent_loss + args.word_loss_weight * word_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        print(f"[Train] Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.6f}")

    encoder_out = os.path.join(out_dir, "bert_text_encoder.pth")
    generator_out = os.path.join(out_dir, "bert_AttnGAN_generator.pth")

    torch.save(student.state_dict(), encoder_out)
    torch.save(netG.state_dict(), generator_out)

    print("\n[Train] Saved:")
    print(f"  - {encoder_out}")
    print(f"  - {generator_out}")
    print("\n[Kaggle] Load with:")
    print('  MODEL_DIR = "/kaggle/input/your-model-folder"')
    print('  cfg.TRAIN.NET_E = "bert_text_encoder.pth"')
    print('  cfg.TRAIN.NET_G = "bert_AttnGAN_generator.pth"')
    print('  cfg.TEXT.ENCODER_TYPE = "bert"')


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT text encoder for AttnGAN")
    parser.add_argument("--model_dir", default=os.path.join(_HERE, "data"))
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "data"))
    parser.add_argument("--captions_pickle", default="captions.pickle")
    parser.add_argument("--teacher_text_encoder", default="text_encoder200.pth")
    parser.add_argument("--generator", default="bird_AttnGAN2.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--word_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--freeze_bert", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
