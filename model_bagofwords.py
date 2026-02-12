import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat

class BagOfWords(nn.Module):
    """
    Implements a weighted bag-of-words model.
    (Where closer entities have higher weights and farther ones lesser.)
    Also has a latent dimensionality for word Embeddings.
    """
    def __init__(self, vocab_sz, nembd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, nembd)
        self.unembed = nn.Linear(nembd, vocab_sz)

    def forward(self, context, targets=None):     # (B, T) tensors
        B, T = context.shape

        emb = self.embedding(context)            # (B, T, Nemb)
        logits = self.unembed(emb)               # (B, T, C)

        # Create a row of decaying wts.
        wts = torch.tril(torch.arange(1, T+1, dtype=torch.long,
                          device=context.device).unsqueeze(0).expand(T, T))
        wts = wts/wts.sum(1, keepdim=True)        # (T, T)
        logits = wts @ logits                     # B gets broadcasted

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(flat(logits), flat(targets))

        return logits, loss

    def generate(self, context, num_tokens):        # (B=1, T=1) tensor
        for i in range(num_tokens):
            logits, _ = self(context)
            logits = logits[:, -1, :]               # Care only about last one (B=1, vocab_sz)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, 1)    # B, 1
            context = torch.cat((context, preds), dim=-1)          # B, T+1

        return context
