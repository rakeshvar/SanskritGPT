import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from tensorflow.python.training.checkpoint_utils import init_from_checkpoint
from tqdm import tqdm

import torch
import torch.optim as optim

import data_loader as dl
from data.lookup import *
from utils import random_init

# ------------------------
# Available Models
# ------------------------
from model_bigram import Bigram
from model_bagofwords import BagOfWords
from model_posbow import PosEmbedBagOfWords
from model_single_head_attn import CausalSelfAttention
from model_multihead_attn import OneLayerGPT
from model_multihead_attn import GPT


# ------------------------
# Hyper Parameters
# ------------------------

@dataclass
class GPTConfig:
    context_size: int = 64
    vocab_size: int = dl.vocab_size
    num_blocks: int = 4
    num_heads: int = 8
    n_head: int = 64
    n_embed: int = 8*64
    n_hidden = 4*8*64
    dropout: float = .2
    batch_size = 32

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

# ------------------------
# Parse Args
# ------------------------
parser = argparse.ArgumentParser(description="Train Sanskrit GPT")
parser.add_argument("-I", "--init_from", type=str, help="Which checkpoint to init from.", default="")
parser.add_argument("-E", "--num_epochs", type=int, default=100)
parser.add_argument("-S", "--steps_per_epoch", type=int, default=100)
parser.add_argument("-O", "--output_dir", type=str, default="checkpoints/")
args = parser.parse_args()

if args.init_from == "":
    init_checkpoint = None
    config = GPTConfig()
    epoch0 = 0
else:
    init_checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
    config = init_checkpoint["config"]
    epoch0 = init_checkpoint["epoch"]

# ------------------------
# Data
# ------------------------
def get_xy(ds, bsz=config.batch_size):
    if ds == "train":
        ds = dl.train_data
    elif ds == "test":
        ds = dl.test_data
    return dl.get_batch(ds, bsz, config.context_size, device)

# ------------------------
# Model
# ------------------------
#m = Bigram(vocab_size)
#m = BagOfWords(vocab_size, n_embed)
#m = PosEmbedBagOfWords(vocab_size, n_embed, context_size)
#m = CausalSelfAttention(vocab_size, n_embed, context_size)
#m = OneLayerGPT(vocab_size, context_size, n_embed, num_heads, n_hidden, dropout)
m = GPT(config)
m = m.to(device)
optimizer = optim.AdamW(m.parameters(), lr=learning_rate)
if init_checkpoint is not None:
    m.load_state_dict(init_checkpoint["model_state_dict"])
    optimizer.load_state_dict(init_checkpoint["optimizer_state_dict"])

# ------------------------
# Eval
# ------------------------
@torch.no_grad
def estimate_losses(eval_bsz):
    m.eval()             # Put in eval mode (no gradient-descent)
    tr_loss = m(*get_xy("train", eval_bsz))[1].item()
    te_loss = m(*get_xy("test",  eval_bsz))[1].item()
    m.train()
    return tr_loss, te_loss

# ------------------------
# Checkpoint
# ------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%m%d-%H%M")
ckpt_head = f"sans_{timestamp}"

def save_model(epoch):
    ckpt_path = output_dir / (ckpt_head + f"_ep{epoch:02d}.pt")
    torch.save({
        "epoch": epoch,
        "config": config,
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print("Saved ", ckpt_path)

# ------------------------
# Train Loop
# ------------------------
for iepoch in range(args.num_epochs):
    epoch = epoch0 + iepoch
    print("Estimating Losses...")
    train_loss, test_loss = estimate_losses(20*config.batch_size)
    print(f"Epoch: {epoch:3d} Losses Train:{train_loss:5.2f} Test:{test_loss:5.2f}")

    context = torch.tensor([encode(random_init())], dtype=torch.long, device=device)
    gen = decode(m.generate(context, 100)[0].tolist())
    print(gen)

    save_model(epoch)

    for step in tqdm(range(args.steps_per_epoch)):
        x, y = get_xy("train")
        logits, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
