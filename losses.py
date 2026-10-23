"""Symmetric InfoNCE contrastive loss (CLIP-style) between image and text embeddings.

Positive pairs are same-index (image[i], text[i]) within a batch - the caller is
responsible for only including one (patient, organ) item per index, so "same index"
means "same patient, same organ" (see dataset.py's PerOrganDataset - every batch item
already is exactly one such pair). Every other pairing within the batch is a negative,
which is why batch composition (how many distinct organs/patients end up in one batch)
matters for how informative the negatives are - see train.py's sampler.

Image-side similarity is "average of per-token cosine similarities", not "cosine of
the mean-pooled token" - these are mathematically different (cosine normalizes each
vector by its own norm before combining; mean-pool normalizes the *sum* once,
afterward). Chosen over mean-pool-then-cosine for being scale-robust per token (one
unusually large-norm token can't dominate the pooled direction) while keeping the same
dense-gradient training stability (every one of the 32 tokens gets gradient every
step, unlike a max-over-tokens scheme). It does NOT fix the "a single informative
token gets diluted by 31 irrelevant ones" concern - averaging 32 similarity scores has
the same dilution characteristic as averaging 32 vectors then taking one cosine; only
a max-over-tokens scheme avoids that, at the cost of sparse (single-token-per-step)
gradients - see the conversation for why max was rejected for this project.
"""
import torch
import torch.nn.functional as F


def info_nce_loss(image_embeds, text_embeds, temperature):
    """image_embeds: (B, 32, D) - Perceiver's per-token output (un-pooled), NOT
    pre-normalized. text_embeds: (B, D), NOT pre-normalized.
    temperature: a learnable nn.Parameter (scalar), matching fvlm's own
    self.temp = nn.Parameter(0.07 * torch.ones([])). Clamped to [0.001, 0.5] in-place
    before every use, exactly where fvlm does it (blip_pretrain.py:196) - not just
    once at init, since the parameter keeps moving during training.
    Returns the scalar loss (mean of image->text and text->image cross-entropy).
    """
    with torch.no_grad():
        temperature.clamp_(0.001, 0.5)

    image_embeds = F.normalize(image_embeds, dim=-1)  # (B, 32, D)
    text_embeds = F.normalize(text_embeds, dim=-1)     # (B, D)

    # sim_per_token[i, j, k] = cosine(image i's k-th token, text j) - full batch x
    # batch x 32 tensor (needed for InfoNCE's in-batch negatives, not just the
    # matching diagonal pairs), then averaged over the 32 tokens.
    sim_per_token = torch.einsum("ikd,jd->ijk", image_embeds, text_embeds)  # (B, B, 32)
    logits = sim_per_token.mean(dim=-1) / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2
