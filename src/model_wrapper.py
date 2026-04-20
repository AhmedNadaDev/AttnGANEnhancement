"""
AttnGAN model architecture + high-level inference wrapper.

Architecture classes are ported from the original eval/model.py and
eval/GlobalAttention.py to be fully Python 3 compatible:
  - F.sigmoid → torch.sigmoid
  - nn.Softmax() without dim → nn.Softmax(dim=1)
  - Variable / volatile removed (use torch.no_grad())
  - pack_padded_sequence uses enforce_sorted=False

Only the generator-side classes are included (no discriminators, no CNN encoder)
since they are not needed for inference.
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from PIL import Image

# Ensure project root is on sys.path so sibling packages resolve correctly
# regardless of how this module is imported.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.kaggle_config import cfg
from src.bert_text_encoder import BERTTextEncoder

# ─── Primitive building blocks ───────────────────────────────────────────────

class GLU(nn.Module):
    """Gated Linear Unit — halves the channel dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nc = x.size(1)
        assert nc % 2 == 0, "Channel count must be even for GLU"
        half = nc // 2
        return x[:, :half] * torch.sigmoid(x[:, half:])


def conv1x1(in_planes: int, out_planes: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def upBlock(in_planes: int, out_planes: int) -> nn.Sequential:
    """2× spatial upsample followed by a conv + BN + GLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )


def Block3x3_relu(in_planes: int, out_planes: int) -> nn.Sequential:
    return nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, channel_num: int):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


# ─── Global Attention ────────────────────────────────────────────────────────

class GlobalAttentionGeneral(nn.Module):
    """
    Cross-modal attention: image spatial features (query) attend over
    word embedding sequence (context).

    input  : (B, idf, ih, iw)
    context: (B, cdf, sourceL)
    returns: weighted context (B, idf, ih, iw), attention map (B, sourceL, ih, iw)
    """

    def __init__(self, idf: int, cdf: int):
        super().__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask: torch.Tensor | None = None

    def applyMask(self, mask: torch.Tensor) -> None:
        self.mask = mask  # (B, sourceL) — True marks padding positions

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # (B, idf, queryL)
        target = input.view(batch_size, -1, queryL)
        # (B, queryL, idf)
        targetT = torch.transpose(target, 1, 2).contiguous()

        # Project word embeddings to image feature dim: (B, idf, sourceL)
        sourceT = self.conv_context(context.unsqueeze(3)).squeeze(3)

        # Attention scores: (B, queryL, sourceL) → (B*queryL, sourceL)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)

        if self.mask is not None:
            # broadcast batch mask across all query positions
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float("inf"))

        attn = self.sm(attn)  # (B*queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # (B, sourceL, queryL)
        attn = torch.transpose(attn, 1, 2).contiguous()

        # Weighted context: (B, idf, queryL) → (B, idf, ih, iw)
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


# ─── Text Encoder ────────────────────────────────────────────────────────────

class RNN_ENCODER(nn.Module):
    """
    Bidirectional LSTM that converts token indices to word-level and
    sentence-level embeddings.

    words_emb : (B, EMBEDDING_DIM, seq_len)
    sent_emb  : (B, EMBEDDING_DIM)
    """

    def __init__(
        self,
        ntoken: int,
        ninput: int = 300,
        drop_prob: float = 0.5,
        nhidden: int = 128,
        nlayers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.rnn_type = cfg.RNN_TYPE
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # nhidden is the *total* embedding dim; split across directions
        self.nhidden = nhidden // self.num_directions

        self._build_module()
        self._init_weights()

    def _build_module(self) -> None:
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            self.ninput,
            self.nhidden,
            self.nlayers,
            batch_first=True,
            dropout=self.drop_prob,
            bidirectional=self.bidirectional,
        )

    def _init_weights(self) -> None:
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz: int):
        weight = next(self.parameters()).data
        shape = (self.nlayers * self.num_directions, bsz, self.nhidden)
        if self.rnn_type == "LSTM":
            return (weight.new(*shape).zero_(), weight.new(*shape).zero_())
        return weight.new(*shape).zero_()

    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        lengths = cap_lens.cpu().tolist()
        emb = pack_padded_sequence(emb, lengths, batch_first=True,
                                   enforce_sorted=False)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]

        # (B, nef, seq_len)
        words_emb = output.transpose(1, 2)

        if self.rnn_type == "LSTM":
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        return words_emb, sent_emb


# ─── Generator components ────────────────────────────────────────────────────

class CA_NET(nn.Module):
    """Conditioning Augmentation — maps sentence embedding to a Gaussian sample."""

    def __init__(self):
        super().__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, : self.c_dim]
        logvar = x[:, self.c_dim :]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    """
    Initial generator stage: noise + conditioning code → (B, GF_DIM, 64, 64).

    Linear projection → reshape (4×4) → 4× upBlock → 64×64 feature map.
    Includes BatchNorm1d after the linear layer (matching the eval checkpoint).
    """

    def __init__(self, ngf: int, ncf: int):
        super().__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf

        nz = self.in_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU(),
        )
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        # (B, Z_DIM + ncf)
        c_z_code = torch.cat((c_code, z_code), dim=1)
        # (B, ngf, 4, 4) after GLU halving
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)   # (B, ngf//2,  8,  8)
        out_code = self.upsample2(out_code)   # (B, ngf//4, 16, 16)
        out_code = self.upsample3(out_code)   # (B, ngf//8, 32, 32)
        out_code = self.upsample4(out_code)   # (B, ngf//16,64, 64)
        return out_code


class NEXT_STAGE_G(nn.Module):
    """
    Subsequent generator stages: cross-modal attention + residual blocks + upsample.
    h_code (B, GF_DIM, H, W) → (B, GF_DIM, 2H, 2W)
    """

    def __init__(self, ngf: int, nef: int, ncf: int):
        super().__init__()
        self.att = GlobalAttentionGeneral(ngf, nef)
        self.residual = nn.Sequential(
            *[ResBlock(ngf * 2) for _ in range(cfg.GAN.R_NUM)]
        )
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code_att, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code_att), dim=1)
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)
        return out_code, att


class GET_IMAGE_G(nn.Module):
    """Converts a feature map to a 3-channel RGB image in [-1, 1]."""

    def __init__(self, ngf: int):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class G_NET(nn.Module):
    """
    Multi-stage AttnGAN generator.

    Produces a list of images [64×64, 128×128, 256×256] when BRANCH_NUM=3.
    The last element is always the highest-resolution output.
    """

    def __init__(self):
        super().__init__()
        ngf = cfg.GAN.GF_DIM         # 32
        nef = cfg.TEXT.EMBEDDING_DIM  # 256
        ncf = cfg.GAN.CONDITION_DIM   # 100

        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        fake_imgs, att_maps = [], []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_imgs.append(self.img_net1(h_code1))

        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_imgs.append(self.img_net2(h_code2))
            if att1 is not None:
                att_maps.append(att1)

        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_imgs.append(self.img_net3(h_code3))
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


# ─── High-level wrapper ──────────────────────────────────────────────────────

class AttnGANWrapper:
    """
    One-stop interface for AttnGAN text-to-image inference.

    Example
    -------
    >>> wrapper = AttnGANWrapper("/kaggle/input/attngan-pretrained")
    >>> img = wrapper.generate("a small red bird with blue wings")
    >>> img.save("bird.png")
    """

    def __init__(self, model_dir: str, device: torch.device | None = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        cfg.CUDA = self.device.type == "cuda"

        print(f"[AttnGAN] Using device: {self.device}")

        vocab_path = os.path.join(model_dir, cfg.CAPTIONS_PICKLE)
        self.wordtoix, self.ixtoword = self._load_vocab(vocab_path)
        n_words = len(self.wordtoix)
        print(f"[AttnGAN] Vocabulary size: {n_words}")

        text_enc_path = os.path.join(model_dir, cfg.TRAIN.NET_E)
        self.text_encoder = self._load_text_encoder(text_enc_path, n_words)

        gen_path = os.path.join(model_dir, cfg.TRAIN.NET_G)
        self.netG = self._load_generator(gen_path)

        print("[AttnGAN] Models loaded and ready.")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_vocab(vocab_path: str):
        """Load ixtoword / wordtoix from captions.pickle."""
        with open(vocab_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        ixtoword = data[2]
        wordtoix = data[3]
        return wordtoix, ixtoword

    def _load_text_encoder(self, path: str, n_words: int):
        encoder_type = str(getattr(cfg.TEXT, "ENCODER_TYPE", "rnn")).lower()
        if encoder_type == "bert":
            model = BERTTextEncoder(
                ixtoword=self.ixtoword,
                embedding_dim=cfg.TEXT.EMBEDDING_DIM,
                max_words=cfg.TEXT.WORDS_NUM,
            )
        else:
            model = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device).eval()
        print(f"[AttnGAN] {encoder_type.upper()} text encoder loaded from {path}")
        return model

    def _load_generator(self, path: str) -> G_NET:
        model = G_NET()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device).eval()
        print(f"[AttnGAN] Generator loaded from {path}")
        return model

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, text: str, copies: int = 2) -> Image.Image:
        """
        Generate a 256×256 image from a natural-language description.

        Args:
            text:   Text prompt, e.g. "a small red bird with blue wings".
            copies: Internal batch size (keep ≥2 for BatchNorm stability).

        Returns:
            PIL.Image (RGB, 256×256).
        """
        from src.utils import tokenize_caption

        captions_np, cap_lens_np = tokenize_caption(
            text, self.wordtoix, cfg.TEXT.WORDS_NUM, copies
        )

        captions = torch.LongTensor(captions_np).to(self.device)
        cap_lens = torch.LongTensor(cap_lens_np).to(self.device)
        noise = torch.FloatTensor(copies, cfg.GAN.Z_DIM).to(self.device)

        with torch.no_grad():
            encoder_type = str(getattr(cfg.TEXT, "ENCODER_TYPE", "rnn")).lower()
            hidden = self.text_encoder.init_hidden(copies)
            words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
            if encoder_type == "bert" and getattr(self.text_encoder, "last_padding_mask", None) is not None:
                mask = self.text_encoder.last_padding_mask.to(self.device)
            else:
                mask = (captions == 0)
            noise.normal_(0, 1)
            fake_imgs, _, _, _ = self.netG(noise, sent_emb, words_embs, mask)

        # Take the highest-resolution image from the first sample in the batch
        im = fake_imgs[-1][0].cpu().numpy()
        im = (im + 1.0) * 127.5
        im = np.clip(im, 0, 255).astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))  # CHW → HWC
        return Image.fromarray(im)
