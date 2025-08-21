import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat

class PosEmbedBagOfWords(nn.Module):
    """
    Implements a Exponentially Weighted Bag of Words Model.
    Latent Dimensionality for Word Embeddings.
    Positional Embedding also.
    """
    def __init__(self, vocab_sz, nembd, context_size):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_sz, nembd)
        self.pos_embed = nn.Embedding(context_size, nembd)
        self.unembed = nn.Linear(nembd, vocab_sz)
        self.context_size = context_size

    def forward(self, context, targets=None):  # (B, T)
        B, T = context.shape
        dpu = context.device

        tok_emb  = self.tok_embed(context)                          # (B, T, N)
        position = torch.arange(T, device=dpu)                      # (T)
        pos_emb  = self.pos_embed(position)                         # (T, N)
        embeding = tok_emb + pos_emb                                # (B, T, N)
        logits   = self.unembed(embeding)                           # (B, T, C)

        powers2 = 2 ** torch.arange(T, device=dpu)                  # (T,)
        mask    = torch.tril(torch.ones(T, T, device=dpu))          # (T, T)
        wts     = mask * powers2                                    #   "
        wts     = wts/wts.sum(1, keepdim=True)
        logits  = wts @ logits                                      # (B, T, C)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(flat(logits), flat(targets))

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