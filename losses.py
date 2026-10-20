"""Symmetric InfoNCE contrastive loss (CLIP-style) between image and text embeddings.

Positive pairs are same-index (image[i], text[i]) within a batch - the caller is
responsible for only including one (patient, organ) item per index, so "same index"
means "same patient, same organ" (see dataset.py's PerOrganDataset - every batch item
already is exactly one such pair). Every other pairing within the batch is a negative,
which is why batch composition (how many distinct organs/patients end up in one batch)
matters for how informative the negatives are - see train.py's sampler.
"""
import torch
import torch.nn.functional as F


def info_nce_loss(image_embeds, text_embeds, temperature):
    """image_embeds, text_embeds: (B, D), NOT pre-normalized - normalized here.
    temperature: a learnable nn.Parameter (scalar), matching fvlm's own
    self.temp = nn.Parameter(0.07 * torch.ones([])). Clamped to [0.001, 0.5] in-place
    before every use, exactly where fvlm does it (blip_pretrain.py:196) - not just
    once at init, since the parameter keeps moving during training.
    Returns the scalar loss (mean of image->text and text->image cross-entropy).
    """
    with torch.no_grad():
        temperature.clamp_(0.001, 0.5)

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.t() / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2
