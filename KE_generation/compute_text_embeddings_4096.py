"""KE step 1: embed every unique text from step 0 (get_unique_reports.py) through
THIS project's text side - frozen BERT [CLS] -> trained adapter (768->4096) -> L2
norm - so the text bank lives in the same 4096-d fc space the image side
(perceiver -> fc) was trained against.

Output npz: "organs" + one "{organ}_feats" array per organ, shape (n_unique, 4096).
Row i of "{organ}_feats" is the embedding of unique_reports.json[organ][i]["text"]
(no reordering anywhere), which is what steps 2/3 rely on to map rows back to
patients.

Usage:

    CUDA_VISIBLE_DEVICES=1 python3 compute_text_embeddings_4096.py \\
        --unique_reports /path/to/unique_reports.json \\
        --adapter_checkpoint /path/to/adapter-stepX.pth \\
        --output_path /path/to/organ_report_embeddings_unique_4096.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ORGANS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_adapter, build_bert_text_encoder  # noqa: E402

BATCH_SIZE = 64


@torch.inference_mode()
def compute_embeddings_for_texts(bert_tokenizer, bert_text_encoder, adapter, texts, batch_size, device):
    """Row i corresponds to texts[i]. Same tokenization as model.encode_text (the
    training path): max_length=384, [CLS] -> adapter."""
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
    parser.add_argument("--unique_reports", required=True, help="step 0's unique_reports.json")
    parser.add_argument("--adapter_checkpoint", required=True,
                         help="e.g. .../adapter-step150000.pth or adapter-final.pth")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="only compute the first N texts per organ - for a quick "
                              "smoke test, not for the real run")
    args = parser.parse_args()

    # Must be set in the shell, not here - torch queues CUDA init at import time, so
    # setting it from inside the script is too late (same as train.py).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES in the shell before launching this script.")
    device = torch.device("cuda")

    with open(args.unique_reports, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    assert set(unique_reports.keys()) == set(ORGANS), (
        f"unique_reports.json organs {sorted(unique_reports.keys())} != ORGANS {sorted(ORGANS)}"
    )

    print("Building frozen BERT text encoder...")
    bert_tokenizer, bert_text_encoder = build_bert_text_encoder(device)

    print(f"Loading trained adapter from {args.adapter_checkpoint}...")
    adapter = build_adapter().to(device)
    adapter.load_state_dict(torch.load(args.adapter_checkpoint, map_location=device))
    adapter.eval()

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

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(args.output_path, **save_dict)
    print(f"Saved {args.output_path}")
    for organ in ORGANS:
        print(f"  {organ}_feats shape: {save_dict[f'{organ}_feats'].shape}")


if __name__ == "__main__":
    main()
