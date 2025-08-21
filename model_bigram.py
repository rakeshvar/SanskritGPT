import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat

class Bigram(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, vocab_sz)

    def forward(self, context, targets=None):           # (B, T) tensors
        logits = self.embedding(context)                # (B, T, vocab_sz)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(flat(logits), flat(targets))
        return logits, loss

    def generate(self, context, num_tokens):        # (B=1, T=1) tensor
        for i in range(num_tokens):
            logits, _ = self(context)          # Here you could pass just the last one (There is a lot of wasted computation as we are looking at all of the context for a bigram)
            logits = logits[:, -1, :]          # Care only about last one (B=1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, 1)           # (B=1, 1)
            context = torch.cat((context, preds), dim=-1)     # (B=1, ++1)

        return context
