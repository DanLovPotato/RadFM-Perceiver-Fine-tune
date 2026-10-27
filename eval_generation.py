"""Held-out report-generation quality evaluation for candidate (perceiver, fc)
checkpoint pairs - actually generates text (not just computes next-token loss, like
train.py's generation_loss does) via the frozen LLaMA-2-7B-chat decoder, then scores
it against the ground-truth organ report with the same metrics Reg2RG's own
evaluation/hf_nlg_evaluation_region.py uses (BLEU-1..4, METEOR, ROUGE-1/2/L, CIDEr).

Needs a MATCHED (perceiver, fc) pair from the same training step - fc is shared
between branches (see model.py's module docstring), so a perceiver trained at step X
needs step X's own fc to produce image embeddings LLaMA was actually co-trained
against; pairing it with a different step's fc (or a randomly-initialized one) isn't
a fair test. train.py checkpoints both together at every save_every_steps for exactly
this reason.

Single-pass design (mirrors eval_retrieval.py): CT loading/preprocessing and the
frozen vision_encoder forward are identical across every candidate checkpoint, so
this does them once per batch and only re-runs the cheap perceiver+fc forward and the
(expensive, autoregressive) generation call per checkpoint. BERT/the contrastive
branch are irrelevant here, so only vision_encoder is built (not the full
PerceiverAlignmentModel, which would also build frozen BERT for nothing).

Usage: python3 eval_generation.py --checkpoints step140000 step145000 --max_items 50
"""
import argparse
import os

import evaluate
import torch
import torch.nn as nn
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader

from dataset import PerOrganDataset, collate_skip_none
from generation import build_lang_model, build_tokenizer, IMAGE_TOKEN_COUNT
from model import build_vision_encoder, PerceiverResampler, PERCEIVER_NUM_LATENTS, VIS_DIM, LLM_HIDDEN_DIM

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
CKPT_DIR = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/outputs/radfm_perceiver_align/run_20261021"


def _load_pair(ckpt_name, device):
    perceiver = PerceiverResampler(dim=VIS_DIM, num_latents=PERCEIVER_NUM_LATENTS)
    perceiver.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"perceiver-{ckpt_name}.pth"), map_location=device))
    fc = nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)
    fc.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"fc-{ckpt_name}.pth"), map_location=device))
    return perceiver.to(device).eval(), fc.to(device).eval()


@torch.no_grad()
def generate_reports(vision_encoder, pairs, lang_model, tokenizer, loader, device, max_items, max_new_tokens):
    """One pass over loader. Returns {ckpt_name: (predictions, references, organs)}."""
    per_ckpt = {name: {"pred": [], "ref": [], "organ": []} for name in pairs}
    n = 0
    for batch in loader:
        if batch is None:
            continue
        volume = batch["image"].to(device)
        texts = batch["text"]
        batch_organs = batch["organ"]

        patch_tokens, _ = vision_encoder(volume)  # (B, N, 768), frozen, shared
        pt = patch_tokens.unsqueeze(1).unsqueeze(1)

        for name, (perceiver, fc) in pairs.items():
            tokens = perceiver(pt)[:, 0]  # (B, 32, 768)
            image_embeds = fc(tokens)  # (B, 32, 4096)

            attention_mask = torch.ones(
                image_embeds.size(0), IMAGE_TOKEN_COUNT, dtype=torch.long, device=device
            )
            generated_ids = lang_model.generate(
                inputs_embeds=image_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            per_ckpt[name]["pred"].extend(decoded)
            per_ckpt[name]["ref"].extend(texts)
            per_ckpt[name]["organ"].extend(batch_organs)

        n += volume.size(0)
        print(f"  generated {n}/{max_items} (x{len(pairs)} checkpoints)")
        if n >= max_items:
            break

    return {
        name: (d["pred"][:max_items], d["ref"][:max_items], d["organ"][:max_items])
        for name, d in per_ckpt.items()
    }


def compute_nlg_metrics(predictions, references):
    """Mirrors Reg2RG's own evaluation/hf_nlg_evaluation_region.py metric set."""
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    refs_listed = [[r] for r in references]

    overall = {}
    for k in range(1, 5):
        result = bleu.compute(predictions=predictions, references=refs_listed, max_order=k, smooth=True)
        overall[f"BLEU-{k}"] = result["bleu"]
    overall["METEOR"] = meteor.compute(predictions=predictions, references=refs_listed)["meteor"]
    rouge_result = rouge.compute(predictions=predictions, references=refs_listed)
    overall["ROUGE-1"] = rouge_result["rouge1"]
    overall["ROUGE-2"] = rouge_result["rouge2"]
    overall["ROUGE-L"] = rouge_result["rougeL"]

    cider_scorer = Cider()
    gts_dict = {str(i): r for i, r in enumerate(refs_listed)}
    res_dict = {str(i): [p] for i, p in enumerate(predictions)}
    cider_score, _ = cider_scorer.compute_score(gts_dict, res_dict)
    overall["CIDEr"] = cider_score

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="e.g. step140000 step145000 - matches "
                              "perceiver-<name>.pth and fc-<name>.pth per checkpoint")
    parser.add_argument("--max_items", type=int, default=30,
                         help="generation is autoregressive and expensive - keep this small")
    parser.add_argument("--max_new_tokens", type=int, default=150)
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
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                         collate_fn=collate_skip_none)

    print("Building vision_encoder (frozen, shared across all checkpoints)...")
    vision_encoder = build_vision_encoder().to(device)

    print(f"Loading {len(args.checkpoints)} (perceiver, fc) checkpoint pairs...")
    pairs = {name: _load_pair(name, device) for name in args.checkpoints}

    print("Building frozen LLaMA-2-7B-chat decoder (shared across all checkpoints)...")
    lang_model = build_lang_model(device)
    tokenizer = build_tokenizer()

    print(f"Generating reports for up to {args.max_items} held-out items per checkpoint...")
    results_by_ckpt = generate_reports(
        vision_encoder, pairs, lang_model, tokenizer, loader, device,
        args.max_items, args.max_new_tokens,
    )

    all_metrics = {}
    for ckpt_name, (predictions, references, organs) in results_by_ckpt.items():
        print(f"\n=== Sample generations: {ckpt_name} ===")
        for i in range(min(2, len(predictions))):
            print(f"[{organs[i]}]")
            print(f"  GT:   {references[i][:150]}")
            print(f"  Pred: {predictions[i][:150]}")

        metrics = compute_nlg_metrics(predictions, references)
        all_metrics[ckpt_name] = metrics
        print(f"\n{ckpt_name} (n={len(predictions)}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Summary ===")
    header = ["checkpoint"] + list(next(iter(all_metrics.values())).keys())
    print(" ".join(f"{h:>10}" for h in header))
    for ckpt_name, metrics in all_metrics.items():
        row = [ckpt_name] + [f"{v:.4f}" for v in metrics.values()]
        print(" ".join(f"{c:>10}" for c in row))


if __name__ == "__main__":
    main()
