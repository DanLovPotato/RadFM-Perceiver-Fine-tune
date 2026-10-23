"""Held-out retrieval evaluation for comparing candidate Perceiver checkpoints -
standard CLIP-style image<->text retrieval accuracy (top-1/top-5), computed on
validation data never seen during this pretraining (train.py only ever reads
train_preprocessed/train_region_mask/train_region_report.csv).

Deliberately contrastive-only: fc isn't checkpointed alongside perceiver (train.py
only saves perceiver.state_dict() at each save_every_steps), so a candidate
perceiver's generation quality can't be fairly tested yet (whatever fc we'd pair it
with wasn't trained jointly with that specific point in perceiver's trajectory).
Retrieval accuracy only needs the (always-consistent, frozen) BERT text encoder, so it
sidesteps that problem entirely and still measures exactly what this pretraining was
for: image/text alignment quality.

Single-pass design: the CT volume loading/preprocessing (CPU-bound, slow) and the
frozen vision_encoder (3D ViT) + BERT text_encoder forward passes are IDENTICAL across
every candidate checkpoint - only perceiver differs. A naive per-checkpoint loop (the
original version of this script) re-does all of that expensive shared work once per
checkpoint for no reason. This version instead loads one perceiver module per
checkpoint up front, then does exactly one pass over the data: per batch, run
vision_encoder/BERT once, then cheaply apply each checkpoint's perceiver to that same
cached patch_tokens. ~4x fewer CT loads/ViT forwards for 4 checkpoints, with no
accuracy difference (the math is identical, just reordered).

Usage: python3 eval_retrieval.py --checkpoints step10000 step20000 step30000 step40000
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PerOrganDataset, collate_skip_none
from model import PerceiverAlignmentModel, PerceiverResampler, PERCEIVER_NUM_LATENTS, VIS_DIM

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
CKPT_DIR = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261020"


def _apply_perceiver(perceiver_module, patch_tokens):
    """Mirrors PerceiverAlignmentModel.encode_image's reshape/pool logic exactly,
    but takes already-computed patch_tokens so the (frozen, shared-across-checkpoints)
    vision_encoder forward isn't repeated per checkpoint."""
    pt = patch_tokens.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 768)
    tokens = perceiver_module(pt)  # (B, 1, 32, 768)
    tokens = tokens[:, 0]  # (B, 32, 768)
    return tokens.mean(dim=1)  # (B, 768)


def _load_perceiver(ckpt_path, device):
    perceiver = PerceiverResampler(dim=VIS_DIM, num_latents=PERCEIVER_NUM_LATENTS)
    perceiver.load_state_dict(torch.load(ckpt_path, map_location=device))
    return perceiver.to(device).eval()


@torch.no_grad()
def compute_embeddings(model, perceivers, loader, device, max_items):
    """One pass over loader. Returns (text_embeds, {ckpt_name: image_embeds})."""
    text_embeds = []
    image_embeds = {name: [] for name in perceivers}
    n = 0
    for batch in loader:
        if batch is None:
            continue
        volume = batch["image"].to(device)
        texts = batch["text"]

        patch_tokens, _ = model.vision_encoder(volume)  # frozen, shared
        t_embeds = model.encode_text(texts)  # frozen, shared
        text_embeds.append(F.normalize(t_embeds, dim=-1).cpu())

        for name, perceiver_module in perceivers.items():
            pooled = _apply_perceiver(perceiver_module, patch_tokens)
            image_embeds[name].append(F.normalize(pooled, dim=-1).cpu())

        n += volume.size(0)
        if n >= max_items:
            break
    text_embeds = torch.cat(text_embeds, dim=0)
    image_embeds = {name: torch.cat(embeds, dim=0) for name, embeds in image_embeds.items()}
    return text_embeds, image_embeds


def retrieval_accuracy(image_embeds, text_embeds):
    """Standard i2t/t2i top-1/top-5 retrieval accuracy. Index i's image and index i's
    text are the one true positive pair (every other index is a negative), matching
    how PerOrganDataset emits one item per (patient, organ)."""
    sim = image_embeds @ text_embeds.t()  # (N, N)
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
                         help="e.g. step10000 step20000 - matches "
                              "perceiver-<name>.pth in the checkpoint dir")
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

    print(f"Loading {len(args.checkpoints)} perceiver checkpoints...")
    perceivers = {}
    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(CKPT_DIR, f"perceiver-{ckpt_name}.pth")
        perceivers[ckpt_name] = _load_perceiver(ckpt_path, device)

    print("Running single pass over held-out data (CT loading + ViT + BERT done once, "
          "perceiver applied per checkpoint)...")
    text_embeds, image_embeds_by_ckpt = compute_embeddings(
        model, perceivers, loader, device, args.max_items
    )

    results = {}
    for ckpt_name, image_embeds in image_embeds_by_ckpt.items():
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
