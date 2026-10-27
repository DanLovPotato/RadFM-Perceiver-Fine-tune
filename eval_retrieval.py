"""Held-out retrieval evaluation for comparing candidate (perceiver, fc, adapter)
checkpoint triples - standard CLIP-style image<->text retrieval accuracy (top-1/
top-5), computed on validation data never seen during this pretraining (train.py only
ever reads train_preprocessed/train_region_mask/train_region_report.csv).

fc is shared between the contrastive and generation branches and adapter only exists
for the contrastive branch (see model.py's module docstring) - evaluating a candidate
step means loading all three together (perceiver-stepX.pth, fc-stepX.pth,
adapter-stepX.pth), since they were trained jointly at that step.

Similarity matches losses.info_nce_loss exactly: each of the 32 image tokens (after
perceiver+fc) gets its own cosine similarity against the pooled text embedding, then
those 32 scores are averaged - not a mean-pooled single image vector compared once.

Single-pass design: the CT volume loading/preprocessing (CPU-bound, slow) and the
frozen vision_encoder (3D ViT) + BERT text_encoder forward passes are IDENTICAL across
every candidate checkpoint - only perceiver/fc/adapter differ. This does exactly one
pass over the data: per batch, run vision_encoder/BERT once, then cheaply apply each
checkpoint's perceiver+fc+adapter to that same cached patch_tokens/BERT output.

Usage: python3 eval_retrieval.py --checkpoints step10000 step30000 step50000 step69000
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PerOrganDataset, collate_skip_none
from model import PerceiverAlignmentModel, PerceiverResampler, PERCEIVER_NUM_LATENTS, VIS_DIM, LLM_HIDDEN_DIM
import torch.nn as nn

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
CKPT_DIR = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261021"


def _load_triple(ckpt_name, device):
    perceiver = PerceiverResampler(dim=VIS_DIM, num_latents=PERCEIVER_NUM_LATENTS)
    perceiver.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"perceiver-{ckpt_name}.pth"), map_location=device))
    fc = nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)
    fc.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"fc-{ckpt_name}.pth"), map_location=device))
    adapter = nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)
    adapter.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"adapter-{ckpt_name}.pth"), map_location=device))
    return (perceiver.to(device).eval(), fc.to(device).eval(), adapter.to(device).eval())


def _apply_image_side(perceiver_module, fc_module, patch_tokens):
    """Mirrors PerceiverAlignmentModel.encode_image exactly, but takes
    already-computed patch_tokens so the (frozen, shared-across-checkpoints)
    vision_encoder forward isn't repeated per checkpoint."""
    pt = patch_tokens.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 768)
    tokens = perceiver_module(pt)  # (B, 1, 32, 768)
    tokens = tokens[:, 0]  # (B, 32, 768)
    return fc_module(tokens)  # (B, 32, 4096)


@torch.no_grad()
def compute_embeddings(model, triples, loader, device, max_items):
    """One pass over loader. Returns {ckpt_name: (image_embeds (N,32,4096), text_embeds (N,4096))}."""
    per_ckpt = {name: {"image": [], "text": []} for name in triples}
    n = 0
    for batch in loader:
        if batch is None:
            continue
        volume = batch["image"].to(device)
        texts = batch["text"]

        patch_tokens, _ = model.vision_encoder(volume)  # frozen, shared
        tokens = model.bert_tokenizer(
            texts, padding="max_length", truncation=True, max_length=384, return_tensors="pt",
        ).to(device)
        bert_out = model.bert_text_encoder.forward_text(tokens)
        cls = bert_out.last_hidden_state[:, 0, :]  # (B, 768), frozen, shared

        for name, (perceiver_module, fc_module, adapter_module) in triples.items():
            image_embeds = _apply_image_side(perceiver_module, fc_module, patch_tokens)  # (B, 32, 4096)
            text_embeds = adapter_module(cls)  # (B, 4096)
            per_ckpt[name]["image"].append(image_embeds.cpu())
            per_ckpt[name]["text"].append(text_embeds.cpu())

        n += volume.size(0)
        if n >= max_items:
            break

    return {
        name: (torch.cat(d["image"], dim=0), torch.cat(d["text"], dim=0))
        for name, d in per_ckpt.items()
    }


def retrieval_accuracy(image_embeds, text_embeds):
    """Standard i2t/t2i top-1/top-5 retrieval accuracy, using the same similarity
    losses.info_nce_loss trains against: average of each image's 32 per-token
    cosine similarities to a given text, not a single pooled-vector cosine.
    image_embeds: (N, 32, 4096). text_embeds: (N, 4096)."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    sim_per_token = torch.einsum("ikd,jd->ijk", image_embeds, text_embeds)  # (N, N, 32)
    sim = sim_per_token.mean(dim=-1)  # (N, N)
    n = sim.size(0)
    targets = torch.arange(n)

    def topk_acc(scores, k):
        topk = scores.topk(k, dim=1).indices
        hits = (topk == targets.unsqueeze(1)).any(dim=1)
        return hits.float().mean().item()

    return {
        "i2t_top1": topk_acc(sim, 1),
        "i2t_top5": topk_acc(sim, min(5, n)),
        "t2i_top1": topk_acc(sim.t(), 1),
        "t2i_top5": topk_acc(sim.t(), min(5, n)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="e.g. step10000 step30000 - matches "
                              "perceiver-<name>.pth/fc-<name>.pth/adapter-<name>.pth "
                              "in the checkpoint dir")
    parser.add_argument("--max_items", type=int, default=400,
                         help="number of held-out (patient, organ) items to evaluate on")
    parser.add_argument("--cuda_device", default="3")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda")

    print("Building held-out validation dataset...")
    dataset = PerOrganDataset(
        DATA_ROOT + "/valid_preprocessed",
        DATA_ROOT + "/valid_region_mask",
        DATA_ROOT + "/radgenome_files/validation_region_report.csv",
    )
    # Fixed subset (no shuffle) so every checkpoint is scored on the exact same items.
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                         collate_fn=collate_skip_none)

    print("Building model (vision_encoder + BERT text_encoder, shared across all checkpoints)...")
    model = PerceiverAlignmentModel(device)

    print(f"Loading {len(args.checkpoints)} (perceiver, fc, adapter) checkpoint triples...")
    triples = {name: _load_triple(name, device) for name in args.checkpoints}

    print("Running single pass over held-out data (CT loading + ViT + BERT done once, "
          "perceiver+fc+adapter applied per checkpoint)...")
    embeds_by_ckpt = compute_embeddings(model, triples, loader, device, args.max_items)

    results = {}
    for ckpt_name, (image_embeds, text_embeds) in embeds_by_ckpt.items():
        metrics = retrieval_accuracy(image_embeds, text_embeds)
        results[ckpt_name] = metrics
        print(f"\n{ckpt_name} (n={image_embeds.size(0)}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Summary ===")
    print(f"{'checkpoint':<15} {'i2t_top1':>10} {'i2t_top5':>10} {'t2i_top1':>10} {'t2i_top5':>10}")
    for ckpt_name, metrics in results.items():
        print(f"{ckpt_name:<15} {metrics['i2t_top1']:>10.4f} {metrics['i2t_top5']:>10.4f} "
              f"{metrics['t2i_top1']:>10.4f} {metrics['t2i_top5']:>10.4f}")


if __name__ == "__main__":
    main()
