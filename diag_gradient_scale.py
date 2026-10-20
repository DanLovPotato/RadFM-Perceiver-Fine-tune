"""One-off diagnostic: is the contrastive loss's gradient on Perceiver's parameters
much smaller than the generation loss's, such that the weighted sum
(0.2*contrastive + 0.8*generation) effectively drowns out the contrastive signal?

Computes each loss term's gradient on perceiver separately (two independent backward
passes on fresh forward passes, not the combined loss) and compares their norms.
Runs on GPU3, standalone - does not touch the live training run on GPU0/1/2.
"""
import sys

import torch

sys.path.insert(0, ".")
from dataset import PerOrganDataset, collate_skip_none
from generation import build_fc, build_lang_model, build_tokenizer, generation_loss
from losses import info_nce_loss
from model import PerceiverAlignmentModel

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES will restrict this to GPU3

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"

print("Building dataset...")
dataset = PerOrganDataset(
    DATA_ROOT + "/train_preprocessed",
    DATA_ROOT + "/train_region_mask",
    DATA_ROOT + "/radgenome_files/train_region_report.csv",
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_skip_none)
batch = next(iter(loader))
while batch is None:
    batch = next(iter(loader))

volume = batch["image"].to(device)
texts = batch["text"]
print("batch organs:", batch["organ"])

print("Building model...")
model = PerceiverAlignmentModel(device, init_temperature=0.07)
fc = build_fc().to(device)
lang_model = build_lang_model(device)
tokenizer = build_tokenizer()

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm() ** 2
    return total ** 0.5

# --- contrastive only ---
model.zero_grad()
fc.zero_grad()
image_tokens, image_pooled = model.encode_image(volume)
text_embeds = model.encode_text(texts)
contrastive = info_nce_loss(image_pooled, text_embeds, model.temp)
contrastive.backward()
contrastive_grad_norm = grad_norm(model.perceiver.parameters())
print(f"\ncontrastive loss = {contrastive.item():.4f}")
print(f"contrastive -> perceiver grad norm = {contrastive_grad_norm:.6f}")

# --- generation only (fresh forward pass, fresh graph) ---
model.zero_grad()
fc.zero_grad()
image_tokens2, image_pooled2 = model.encode_image(volume)
generation = generation_loss(image_tokens2, texts, fc, lang_model, tokenizer, device)
generation.backward()
generation_grad_norm = grad_norm(model.perceiver.parameters())
print(f"\ngeneration loss = {generation.item():.4f}")
print(f"generation -> perceiver grad norm = {generation_grad_norm:.6f}")

print(f"\nratio (generation_grad / contrastive_grad) = {generation_grad_norm / (contrastive_grad_norm + 1e-12):.2f}")
print(f"with weights 0.2/0.8, effective ratio = "
      f"{(0.8 * generation_grad_norm) / (0.2 * contrastive_grad_norm + 1e-12):.2f}")
