import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat

class CausalSelfAttention(nn.Module):
    """
    Implements a Exponentially Weighted Bag of Words Model.
    """
    def __init__(self, vocab_sz, nembd, context_size):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_sz, nembd)
        self.pos_embed = nn.Embedding(context_size, nembd)
        self.keys = nn.Linear(nembd, nembd)
        self.queries = nn.Linear(nembd, nembd)
        self.values = nn.Linear(nembd, nembd)
        self.unembed = nn.Linear(nembd, vocab_sz)
        self.context_size = context_size
        self.nembd = nembd

    def forward(self, context, targets=None):  # (B, T)
        B, T = context.shape
        dpu = context.device

        tok_emb  = self.tok_embed(context)                          # (B, T, N)
        position = torch.arange(T, device=dpu)                      # (T)
        pos_emb  = self.pos_embed(position)                         # (T, N)
        embeding = tok_emb + pos_emb                                # (B, T, N)

        keys = self.keys(embeding)
        qrys = self.queries(embeding)
        vals = self.values(embeding)                               # (B, T, N)

        attn = (qrys @ keys.transpose(1, 2))/math.sqrt(self.nembd) # (B, T, N) @ (B, N, T) = (B, T, T)
        mask = torch.tril(torch.ones(T, T, device=dpu)).bool()
        attn = attn.masked_fill(mask == False, float('-inf'))
        attn = F.softmax(attn, dim=-1)                             # (B, T, T)
        logits = attn @ vals                                       # (B, T, N)
        logits   = self.unembed(logits)                           # (B, T, C)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(flat(logits), flat(targets))   # (B*T, C) , (B*T,)

        return logits, loss

    def generate(self, context, num_tokens):                # (B=1, T=1)
        for _ in range(num_tokens):
            if context.shape[-1] <= self.context_size:      # Trim to context_size
                this_context = context
            else:
                this_context = context[:, -self.context_size:]
            logits, _ = self(this_context)                    # (B=1, T, C)
            logits = logits[:, -1, :]                    # (B=1,    C)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, 1) # (B=1, 1)
            context = torch.cat((context, preds), dim=-1)

        return context