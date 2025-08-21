import torch
import torch.optim as optim
from tqdm import tqdm

from model_bigram import Bigram
from model_bagofwords import BagOfWords
from model_posbow import PosEmbedBagOfWords
from model_single_head_attn import CausalSelfAttention
from model_multihead_attn import OneLayerGPT
import data_loader as dl
from data.lookup import *
from utils import random_init

vocab_size = dl.vocab_size
batch_size = 32
context_size = 64
num_heads = 8
n_head = 64
n_embed = num_heads * n_head # 256
n_hidden = 4*n_embed
dropout = .2

num_epochs = 100
steps_per_epoch = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4

def get_xy(ds, bsz = batch_size):
    if ds == "train":
        ds = dl.train_data
    elif ds == "test":
        ds = dl.test_data
    return dl.get_batch(ds, bsz, context_size, device)

#m = Bigram(vocab_size)
#m = BagOfWords(vocab_size, n_embed)
#m = PosEmbedBagOfWords(vocab_size, n_embed, context_size)
#m = CausalSelfAttention(vocab_size, n_embed, context_size)
m = OneLayerGPT(vocab_size, context_size, n_embed, num_heads, n_hidden, dropout)
m = m.to(device)
optimizer = optim.AdamW(m.parameters(), lr=learning_rate)

@torch.no_grad
def estimate_losses(eval_bsz):
    m.eval()             # Put in eval mode (no gradient-descent)
    tr_loss = m(*get_xy("train", eval_bsz))[1].item()
    te_loss = m(*get_xy("test",  eval_bsz))[1].item()
    m.train()
    return tr_loss, te_loss

for epoch in range(num_epochs):
    print("Estimating Losses...")
    train_loss, test_loss = estimate_losses(20*batch_size)
    print(f"Epoch: {epoch:3d} Losses Train:{train_loss:5.2f} Test:{test_loss:5.2f}")

    context = torch.tensor([encode(random_init())], dtype=torch.long, device=device)
    gen = decode(m.generate(context, 100)[0].tolist())
    print(gen)

    for step in tqdm(range(steps_per_epoch)):
        x, y = get_xy("train")
        logits, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

