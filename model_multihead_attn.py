import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, context_size, n_embed, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(context_size, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_embed = n_embed

    def forward(self, context):
        B, T = context.shape
        dpu = context.device
        tok_emb = self.tok_embed(context)                               # B, T, N
        pos_emb = self.pos_embed(torch.arange(T, device=dpu))           #    T, N
        emb = tok_emb + pos_emb                                         # B, T, N
        out = self.dropout(emb)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, n_embed, num_heads, n_hidden, dropout):
        super().__init__()
        self.layernorm1 = LayerNorm(n_embed)
        self.mhcsattn = MultiHeadCausalSelfAttention(n_embed, num_heads, dropout)
        self.layernorm2 = LayerNorm(n_embed)
        self.mlp = MLP(n_embed, n_hidden, dropout)

    def forward(self, x):
        x = x + self.mhcsattn(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

    def forward(self, context):
        return context

class MLP(nn.Module):
    def __init__(self, n_embed, n_hidden, dropout):
        super().__init__()
        self.layer1 = nn.Linear(n_embed, n_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context):
        out = self.layer1(context)
        out = self.relu(out)
        out = self.layer2(out)
        return self.dropout(out)


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embed, num_heads, dropout):
        super().__init__()
        assert n_embed % num_heads == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.dropout_a = nn.Dropout(dropout)
        self.dropout_b = nn.Dropout(dropout)

        self.n_embed = n_embed
        self.num_heads = num_heads

    def forward(self, x):                                         # B, T, N
        B, T, N = x.shape
        h, n = self.num_heads, self.n_embed // self.num_heads
        dpu = x.device

        qkv = self.qkv(x)                                          # B, T, 3N
        q, k, v = qkv.split(self.n_embed, dim = -1)                # B, T, N
        q = q.view(B, T, h, n).transpose(1, 2)                # B, T, N ->
        k = k.view(B, T, h, n).transpose(1, 2)                # B, T, h, n ->
        v = v.view(B, T, h, n).transpose(1, 2)                # B, h, T, n
                                                              # B, h, n, T (for k)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(n)       # B, h, T, T
        mask = torch.tril(torch.ones(T, T, device=dpu)) == 0  #       T, T
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)                        # B, h, T, T

        attn = self.dropout_a(attn)
        out = attn @ v                 # (B, h, T, T) x (B, h, T, n) = (B, h, T, n)
                                                #  ^ This is the dissolving or summing dimension
                                                #    So soft-max should be along that dimension
        out = out.transpose(1, 2).contiguous().view(B, T, N)
        out = self.dropout_b(out)
        return out

class OneLayerGPT(nn.Module):
    def __init__(self, vocab_size, context_size, n_embed, num_heads, n_hidden, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, context_size, n_embed, dropout)
        self.attnblock = AttentionBlock(n_embed, num_heads, n_hidden, dropout)
        self.layernorm = LayerNorm(n_embed)
        self.unembedding = nn.Linear(n_embed, vocab_size)

        self.context_size = context_size

    def forward(self, context, targets=None):
        dpu = context.device
        emb = self.embedding(context)
        out = self.attnblock(emb)
        out = self.layernorm(out)

        if targets is not None:
            logits = self.unembedding(out)
            loss = F.cross_entropy(flat(logits), flat(targets))
        else:
            # Generate mode, so look at only the last time-step
            logits = self.unembedding(out[:, [-1], :])
            loss = None

        return logits, loss

    def generate(self, context, num_tokens):                       # B=1, T=1
        for _ in range(num_tokens):
            if context.shape[-1] > self.context_size:
                this_context = context[:, -self.context_size:]     # B=1, <=T, C
            else:
                this_context = context
            logits, _ = self(this_context)                         # B=1, 1, C
            logits = logits[:, -1, :]                              # B=1, C
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, 1)        # B=1, C=1
            context = torch.cat((context, preds), dim=-1)

        return context