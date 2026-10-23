"""Recomputes fvlm's organ_report_embeddings.npz, but through THIS project's trained
adapter (768->4096, aligned to fc's/Reg2RG's LLaMA-2-7B-chat space) instead of fvlm's
own text_proj (768->256) - see the conversation: once the mentor's "everything aligns
to fc" design was settled on, the knowledge bank has to move to fc's 4096-d space too,
since fvlm's original 256-d bank was built for a completely different (fvlm's own)
vision/text embedding space that this project no longer uses for the contrastive
branch (adapter replaced it).

Mirrors fvlm/KE_prepare_unique/2_compute_text_embeddings.py's structure/output format
exactly (same unique_reports.json input, same organ_report_embeddings.npz output
shape) - only the text-side projection differs (adapter instead of text_proj), and the
output is 4096-d per row instead of 256-d.

Only runs once a real (non-random) adapter checkpoint exists - training produces one
at every save_every_steps and at the end (see train.py). Usage:

    python3 compute_text_embeddings_4096.py --adapter_checkpoint /path/to/adapter-stepX.pth \\
        --output_path /path/to/organ_report_embeddings_4096.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import build_adapter, build_bert_text_encoder

import json
import sys

_ARCHIVE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Reg2RG-debug-current-full_correct_F_20260920_archive",
    "src",
)
if _ARCHIVE_SRC not in sys.path:
    sys.path.insert(0, _ARCHIVE_SRC)
from Dataset.radgenome_dataset_train import REGIONS as ORGANS  # noqa: E402

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
UNIQUE_REPORTS_JSON = os.path.join(DATA_ROOT, "EK_files_train_unique", "unique_reports.json")

BATCH_SIZE = 64


@torch.inference_mode()
def compute_embeddings_for_texts(bert_tokenizer, bert_text_encoder, adapter, texts, batch_size, device):
    """Mirrors fvlm's compute_embeddings_for_texts() - same batching/ordering
    contract (row i corresponds to texts[i]), but [CLS] -> adapter (4096-d) instead
    of [CLS] -> text_proj (256-d)."""
    feats = np.zeros((len(texts), adapter.out_features), dtype=np.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc="embed"):
        batch_texts = texts[start:start + batch_size]
        tokens = bert_tokenizer(
            batch_texts, padding="max_length", truncation=True,
            max_length=384, return_tensors="pt",
        ).to(device)
        output = bert_text_encoder.forward_text(tokens)
        cls = output.last_hidden_state[:, 0, :]  # (B, 768)
        text_feat = F.normalize(adapter(cls), dim=-1)  # (B, 4096)
        feats[start:start + batch_size, :] = text_feat.cpu().numpy()
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_checkpoint", required=True,
                         help="e.g. .../adapter-step40000.pth or adapter-final.pth")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="only compute the first N texts per organ - for a quick "
                              "smoke test, not for the real run")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda")

    print("Building frozen BERT text encoder...")
    bert_tokenizer, bert_text_encoder = build_bert_text_encoder(device)

    print(f"Loading trained adapter from {args.adapter_checkpoint}...")
    adapter = build_adapter().to(device)
    adapter.load_state_dict(torch.load(args.adapter_checkpoint, map_location=device))
    adapter.eval()

    with open(UNIQUE_REPORTS_JSON, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    assert set(unique_reports.keys()) == set(ORGANS), (
        f"unique_reports.json organs don't match current REGIONS: "
        f"json={sorted(unique_reports.keys())} vs REGIONS={sorted(ORGANS)}"
    )

    save_dict = {"organs": np.array(ORGANS)}
    for organ in ORGANS:
        entries = unique_reports[organ]
        if args.max_samples is not None:
            entries = entries[:args.max_samples]
        texts = [entry["text"] for entry in entries]
        print(f"{organ}: {len(texts)} unique texts")
        feats = compute_embeddings_for_texts(
            bert_tokenizer, bert_text_encoder, adapter, texts, BATCH_SIZE, device,
        )
        assert not np.isnan(feats).any(), f"{organ} embeddings contain NaN"
        save_dict[f"{organ}_feats"] = feats

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, **save_dict)
    print(f"Saved {args.output_path}")
    for organ in ORGANS:
        print(f"  {organ}_feats shape: {save_dict[f'{organ}_feats'].shape}")


if __name__ == "__main__":
    main()
