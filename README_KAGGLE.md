# AttnGAN — Kaggle Inference Package

Generate photorealistic bird images from text descriptions using the pretrained
**AttnGAN** model (Xu et al., 2018).

---

## Project Structure

```
kaggle_attngan/
├── src/
│   ├── model_wrapper.py   # PyTorch model architecture + AttnGANWrapper class
│   ├── inference.py       # Batch generation pipeline
│   └── utils.py           # Tokenisation, image saving helpers
├── config/
│   └── kaggle_config.py   # All hyper-parameters and path defaults
├── outputs/               # Generated images are written here
├── main_kaggle.py         # ← MAIN ENTRY POINT
├── requirements.txt
└── README_KAGGLE.md
```

---

## Quick Start on Kaggle

### Step 1 — Upload the pretrained model as a Kaggle Dataset

Download these three files from the
[original AttnGAN repository](https://github.com/taoxugit/AttnGAN)
(Google Drive links in its README):

| File | Description |
|------|-------------|
| `captions.pickle` | CUB-200 vocabulary (`[train_caps, test_caps, ixtoword, wordtoix]`) |
| `text_encoder200.pth` | DAMSM text encoder weights |
| `bird_AttnGAN2.pth` | AttnGAN generator weights |

Create a new Kaggle Dataset called `attngan-pretrained` and upload the three
files into its root.

### Step 2 — Add this notebook to Kaggle

Upload the entire `kaggle_attngan/` folder to a Kaggle Notebook and attach
the `attngan-pretrained` dataset.

### Step 3 — Run

```bash
python main_kaggle.py
```

Or, from a Kaggle Notebook cell:

```python
%run kaggle_attngan/main_kaggle.py
```

Generated images appear in `/kaggle/working/outputs/`.

---

## Configuration

All settings live in two places:

### `main_kaggle.py` — top-level paths and prompts

```python
MODEL_DIR  = "/kaggle/input/attngan-pretrained"   # dataset path
OUTPUT_DIR = "/kaggle/working/outputs"            # output path

TEXTS = [
    "a small red bird with blue wings and a yellow beak",
    "a yellow bird sitting on a branch with green leaves",
    # add more prompts here
]
```

### `config/kaggle_config.py` — model architecture

Change only if you use a checkpoint with different settings (e.g. COCO model):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cfg.GAN.GF_DIM` | 32 | Generator feature-map base width |
| `cfg.GAN.Z_DIM` | 100 | Noise vector dimension |
| `cfg.TEXT.EMBEDDING_DIM` | 256 | Text encoder output size |
| `cfg.TREE.BRANCH_NUM` | 3 | Number of generator stages (64/128/256 px) |

---

## Using the Python API

```python
import sys
sys.path.insert(0, "/kaggle/working/kaggle_attngan")

from config.kaggle_config import cfg
from src.model_wrapper import AttnGANWrapper

cfg.TRAIN.NET_E = "text_encoder200.pth"
cfg.TRAIN.NET_G = "bird_AttnGAN2.pth"

wrapper = AttnGANWrapper("/kaggle/input/attngan-pretrained")

image = wrapper.generate("a tiny bird with a red head and grey body")
image.save("/kaggle/working/outputs/my_bird.png")
image   # display inline in notebook
```

---

## How It Works

```
Text prompt
    │
    ▼
NLTK tokenisation → vocabulary index lookup
    │
    ▼
RNN_ENCODER  (bidirectional LSTM)
    │  words_emb  (B, 256, T)
    │  sent_emb   (B, 256)
    ▼
CA_NET  (Conditioning Augmentation — VAE reparameterisation)
    │  c_code  (B, 100)
    ▼
INIT_STAGE_G  →  64×64 feature map  →  GET_IMAGE_G  →  img₁ (64×64)
    │
    ▼
NEXT_STAGE_G  (GlobalAttention + ResBlocks + upsample)
    →  128×128 feature map  →  GET_IMAGE_G  →  img₂ (128×128)
    │
    ▼
NEXT_STAGE_G  (same)
    →  256×256 feature map  →  GET_IMAGE_G  →  img₃ (256×256)  ← final output
```

---

## Requirements

All packages are pre-installed in the Kaggle GPU runtime:

```
torch >= 1.13
torchvision >= 0.14
numpy >= 1.21
Pillow >= 9.0
nltk >= 3.7
easydict >= 1.10
```

For local development:

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{Xu2018AttnGAN,
  title   = {AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks},
  author  = {Tao Xu and Pengchuan Zhang and Qiuyuan Huang and Han Zhang and Zhe Gan and Xiaolei Huang and Xiaodong He},
  journal = {CVPR},
  year    = {2018}
}
```
