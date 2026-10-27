"""KE steps 1b + 3: for every patient of a split, per organ, pick top-k TRAIN reports to
use as references, and write organ_annotation.json - the file Reg2RG's
organ_annotation_path reads.

Image side: CT x organ mask (Reg2RG's own crop, via PerOrganDataset) -> frozen ViT ->
perceiver -> fc -> 32 tokens (4096-d). Text side: step 1's bank ({organ}_feats).
Similarity matches training exactly (losses.info_nce_loss): the mean over 32 tokens
of cosine(token, text). With every token L2-normalized and text already
L2-normalized, that equals q @ t where q = mean of the normalized tokens - so q is
NOT re-normalized after the mean (that would be a different similarity).

Two-stage retrieval (--mode two_stage, the default). Plain nearest-5 over all
sentences hands normal patients mostly abnormal reports: after dedup, abnormal
sentences outnumber normal ones 2-24x per organ (normal phrasing is shared by
thousands, abnormal phrasing is nearly one per patient), so some abnormal sentence is
always closest. Instead:
  1b) prototypes per organ (step 0b's bank_labels.json): Pn / Pa = plain mean of the
      NORMAL / ABNORMAL sentence vectors, using only sentences really written in the
      reports - the "<organ> shows no significant
      abnormalities." template we fill in for organs without a sentence is left out: it
      never appears in training (training and Reg2RG only use organs that have a
      sentence), so it sits far from every image and dragged Pn away (breast/thyroid
      Pn was ~99% template, which put every patient on the ABNORMAL side).
  a)  zero-shot decision, NO threshold: NORMAL if the image is closer to Pn than to Pa
      (cos(q, Pn) >= cos(q, Pa)). Nothing is fitted on any dataset's statistics, so it
      applies unchanged to new patients/sites. On val organs that have a sentence this
      matched a train-fitted threshold (balanced acc 0.68 vs 0.67).
  b)  NORMAL   -> top-k among NORMAL sentences, plain cosine;
      ABNORMAL -> top-k among ABNORMAL sentences, both sides mean-centered first
      (image mean M = mean train image vector per organ, text mean = bank row mean),
      which spreads picks over more distinct findings instead of a few "hub" sentences.
  Organs in NO_DECISION_ORGANS skip a)/b) and use plain top-k over all sentences:
  esophagus / trachea (the decision is near random there, AUC ~0.6) and breast /
  thyroid (only ~4.5% of scans have a sentence - too few to build or check a decision).
M is computed by the train run and saved (--image_mean); the val run loads it, so the
train run must finish first. --mode plain = plain top-k for every organ.

Train: query = train scans. For each query, every bank row written by ANY scan of the
same person is masked out - not just the query scan's own row. Ids look like
train_10006_b_1 (person 10006, scan b), and 2637 train people have several scans whose
images and reports are near-identical, so a person's other scan would be the easiest
(leaked) hit. A masked row is masked only for this query, not removed from the bank
(e.g. the template stays retrievable for everyone else). Unmasked rows contain no scan
of this person at all, so `patient_ids[0]` of a hit row is always another person.
Val: query = validation patients, bank = the SAME train text bank, nothing masked
(val patients aren't in it). Validation has no bank of its own: every neighbor is a
train patient, and Reg2RG's validation bank_npy_path is the same train npz step 2
writes - only organ_annotation_path differs between splits.

Output, one record per patient (sorted ids) - the layout Reg2RG's my_embedding_layer.py reads:

    {"train": [                                  # "validation" for --split val
        {"id": "train_10000_a_1", "image_path": "train_10000_a_1.nii.gz",
         "abdomen": "<this patient's abdomen text>",
         "abdomen_indices": [{"id": "train_14600_a_1", "organ": "abdomen"}, ... x top_k],
         ... a "<organ>" / "<organ>_indices" pair per organ
        }, ...]}

A (patient, organ) with no usable image (mask missing or all-zero) gets an empty
`<organ>_indices` list. Image queries are cached to --query_cache since encoding is
the hours-long part; a rerun with the same cache only redoes the cheap retrieval.

Usage:

    CUDA_VISIBLE_DEVICES=1 python3 build_organ_annotation.py --split train \\
        --ckpt_dir /path/to/run_dir --ckpt_name step150000 \\
        --unique_reports /path/to/unique_reports.json \\
        --unique_feats /path/to/organ_report_embeddings_unique_4096.npz \\
        --bank_labels /path/to/bank_labels.json \\
        --prototypes /path/to/prototypes.npz \\
        --image_mean /path/to/train/image_mean.npz \\
        --output_path /path/to/train/organ_annotation.json \\
        --query_cache /path/to/train/image_queries.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import ORGANS, SPLITS, load_split, person_id, sample_pid, template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (  # noqa: E402
    LLM_HIDDEN_DIM, PERCEIVER_NUM_LATENTS, VIS_DIM, PerceiverResampler, build_vision_encoder,
)


def load_perceiver_fc(ckpt_dir, ckpt_name, device):
    """Matched (perceiver, fc) pair from the same step - see eval_generation.py."""
    perceiver = PerceiverResampler(dim=VIS_DIM, num_latents=PERCEIVER_NUM_LATENTS)
    perceiver.load_state_dict(torch.load(os.path.join(ckpt_dir, f"perceiver-{ckpt_name}.pth"), map_location=device))
    fc = nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)
    fc.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fc-{ckpt_name}.pth"), map_location=device))
    return perceiver.to(device).eval(), fc.to(device).eval()


class PatientOrgansDataset(Dataset):
    """One item = one patient: the CT is read ONCE and all of that patient's organs are
    cropped in a single mask_nii_img_to_tensor call - exactly how Reg2RG's own
    RadGenomeDataset_Train.__getitem__ calls it (all masks at once). Each organ's crop
    only depends on the CT and that organ's own mask (deterministic CropForeground ->
    Resize -> ToTensor per organ), so the crops are identical to cropping one organ per
    call, while reading each CT ~10x less often.

    Organs are enumerated by mask file, not by csv Sentence: organs without a finding
    still have the template text in the bank, so they need an image query too.
    """

    def __init__(self, base, patients):
        self.base = base            # Reg2RG's RadGenomeDataset_Train (for mask_nii_img_to_tensor)
        self.patients = patients    # [(pid, img_file, {organ: mask_file}), ...]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        pid, img_file, mask_files = self.patients[index]
        try:
            mask_img_tensors, _ = self.base.mask_nii_img_to_tensor(img_file, mask_files)
        except UnboundLocalError:
            # mask_nii_img_to_tensor's own edge case: crashes when EVERY mask is empty
            # (see dataset.py) - no usable organ for this patient
            return []
        # organs whose mask is all-zero are skipped by mask_nii_img_to_tensor itself
        return [(pid, organ, mask_img_tensors[organ][0]) for organ in mask_files if organ in mask_img_tensors]


def collate_patients(batch):
    items = [item for patient_items in batch for item in patient_items]
    if not items:
        return None
    return {
        "patient_id": [pid for pid, _, _ in items],
        "organ": [organ for _, organ, _ in items],
        "image": torch.stack([volume for _, _, volume in items], dim=0),  # (n, 3, 256, 256, 64)
    }


@torch.no_grad()
def compute_image_queries(dataset, mask_folder, query_ids, ckpt_dir, ckpt_name,
                          batch_size, num_workers, gpu_chunk, device):
    """Returns (queries {organ: float32 (n_query, 4096)}, valid {organ: bool (n_query,)}).
    batch_size = patients per DataLoader batch; gpu_chunk = max organ crops per forward."""
    query_pos = {pid: i for i, pid in enumerate(query_ids)}

    patients = []
    for sample in dataset._base.samples:
        pid = sample_pid(sample)
        if pid not in query_pos:
            continue
        mask_files = {}
        for organ in ORGANS:
            mask_file = os.path.join(mask_folder, f"seg_{pid}", f"{organ}.nii.gz")
            if os.path.exists(mask_file):
                mask_files[organ] = mask_file
        if mask_files:
            patients.append((pid, sample["image"], mask_files))
    n_items = sum(len(m) for _, _, m in patients)
    print(f"{len(patients)} patients, {n_items} (patient, organ) items to encode")

    loader = DataLoader(PatientOrgansDataset(dataset._base, patients), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate_patients)

    vision_encoder = build_vision_encoder().to(device)
    perceiver, fc = load_perceiver_fc(ckpt_dir, ckpt_name, device)

    queries = {organ: np.zeros((len(query_ids), LLM_HIDDEN_DIM), dtype=np.float32) for organ in ORGANS}
    valid = {organ: np.zeros(len(query_ids), dtype=bool) for organ in ORGANS}
    for batch in tqdm(loader, desc="encode images (patients)"):
        if batch is None:
            continue
        for start in range(0, len(batch["organ"]), gpu_chunk):
            volume = batch["image"][start:start + gpu_chunk].to(device)
            patch_tokens, _ = vision_encoder(volume)  # (B, N, 768)
            tokens = perceiver(patch_tokens.unsqueeze(1).unsqueeze(1))[:, 0]  # (B, 32, 768)
            image_embeds = fc(tokens)  # (B, 32, 4096)
            # mean of per-token cosines == (mean of normalized tokens) @ normalized text
            q = F.normalize(image_embeds.float(), dim=-1).mean(dim=1).cpu().numpy()  # (B, 4096)
            for b, (pid, organ) in enumerate(zip(batch["patient_id"][start:start + gpu_chunk],
                                                 batch["organ"][start:start + gpu_chunk])):
                i = query_pos[pid]
                queries[organ][i] = q[b]
                valid[organ][i] = True
    return queries, valid


def build_excluded_rows(unique_reports, query_ids):
    """organ -> (query_idx, row) int64 arrays: for each query scan, every bank row
    written by any scan of the same person (its own row included)."""
    query_pos = {pid: i for i, pid in enumerate(query_ids)}
    excluded = {}
    for organ in ORGANS:
        person_rows = {}
        for row, entry in enumerate(unique_reports[organ]):
            for pid in entry["patient_ids"]:
                person_rows.setdefault(person_id(pid), set()).add(row)
        missing = [pid for pid in query_ids if person_id(pid) not in person_rows]
        assert not missing, f"{organ}: train patients missing from unique_reports.json, e.g. {missing[:3]}"
        pairs = [(query_pos[pid], row) for pid in query_ids for row in person_rows[person_id(pid)]]
        excluded[organ] = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return excluded


# Organs where two_stage falls back to plain top-k over all sentences (see module docstring).
NO_DECISION_ORGANS = {"esophagus", "trachea and bronchie", "breast", "thyroid"}


def build_prototypes(bank_feats, unique_reports, bank_labels):
    """organ -> {"normal": Pn, "abnormal": Pa}, L2-normalized plain means of the NORMAL /
    ABNORMAL sentence vectors. The filled-in template sentence is excluded (never seen in
    training). Not weighted by how many patients wrote a sentence: with the template out,
    weighting made no difference on val (balanced acc 0.73 vs 0.74), so keep it simple."""
    prototypes = {}
    for organ in ORGANS:
        feats = bank_feats[organ]
        is_normal = np.array(bank_labels[organ], dtype=bool)
        real = np.array([e["text"] != template(organ) for e in unique_reports[organ]])
        protos = {}
        for name, rows in (("normal", is_normal & real), ("abnormal", ~is_normal & real)):
            p = feats[rows].mean(0)
            protos[name] = p / np.linalg.norm(p)
        prototypes[organ] = protos
    return prototypes


@torch.no_grad()
def retrieve_top_k(queries, valid, bank_feats, excluded, top_k, device, mode,
                   bank_labels=None, prototypes=None, image_mean=None, chunk=4096):
    """Returns (top, decisions):
    top       - organ -> int64 (n_query, top_k) bank rows, -1 for invalid queries/empty slots
    decisions - organ -> {"score": float32 (n_query,), "normal": bool (n_query,)} for organs
                that use the zero-shot decision (score = cos(q, Pn) - cos(q, Pa), normal = score >= 0)."""
    top, decisions = {}, {}
    for organ in tqdm(ORGANS, desc="top-k"):
        txt = torch.from_numpy(bank_feats[organ]).to(device)  # (n_bank, 4096)
        n_query = queries[organ].shape[0]
        k = min(top_k, txt.shape[0] - 1)
        out = np.full((n_query, top_k), -1, dtype=np.int64)
        decide = mode == "two_stage" and organ not in NO_DECISION_ORGANS
        if decide:
            normal_rows = torch.tensor(bank_labels[organ], dtype=torch.bool, device=device)
            p_n = torch.from_numpy(prototypes[organ]["normal"]).to(device)
            p_a = torch.from_numpy(prototypes[organ]["abnormal"]).to(device)
            txt_c = F.normalize(txt - txt.mean(0), dim=-1)
            m = torch.from_numpy(image_mean[organ]).to(device)
            scores = np.zeros(n_query, dtype=np.float32)
        for start in range(0, n_query, chunk):
            q = torch.from_numpy(queries[organ][start:start + chunk]).to(device)
            if not decide:
                sim = q @ txt.t()  # (chunk, n_bank)
            else:
                q_n = F.normalize(q, dim=-1)
                score = q_n @ p_n - q_n @ p_a  # (chunk,)
                scores[start:start + chunk] = score.cpu().numpy()
                sim_normal = (q @ txt.t()).masked_fill(~normal_rows, float("-inf"))
                sim_abnormal = (F.normalize(q - m, dim=-1) @ txt_c.t()).masked_fill(normal_rows, float("-inf"))
                sim = torch.where((score >= 0)[:, None], sim_normal, sim_abnormal)
            if excluded is not None:
                pairs = excluded[organ]
                pairs = pairs[(pairs[:, 0] >= start) & (pairs[:, 0] < start + chunk)]
                pairs = torch.from_numpy(pairs).to(device)
                sim[pairs[:, 0] - start, pairs[:, 1]] = float("-inf")
            values, indices = sim.topk(k, dim=1)
            indices[torch.isinf(values)] = -1  # fewer than k candidates left in the pool
            out[start:start + chunk, :k] = indices.cpu().numpy()
        out[~valid[organ]] = -1
        top[organ] = out
        if decide:
            decisions[organ] = {"score": scores, "normal": scores >= 0}
    return top, decisions


def build_records(query_ids, captions, top, unique_reports):
    records = []
    for i, pid in enumerate(query_ids):
        record = {"id": pid, "image_path": f"{pid}.nii.gz"}
        for organ in ORGANS:
            entries = unique_reports[organ]
            record[organ] = captions[pid][organ]
            record[f"{organ}_indices"] = [
                {"id": entries[idx]["patient_ids"][0], "organ": organ}
                for idx in top[organ][i].tolist() if idx >= 0
            ]
        records.append(record)
    return records


def verify(records, bank_ids, top_k, exclude_self):
    n_empty = 0
    for record in records:
        for organ in ORGANS:
            neighbors = record[f"{organ}_indices"]
            n_empty += not neighbors
            assert len(neighbors) <= top_k
            for neighbor in neighbors:
                assert neighbor["organ"] == organ
                assert neighbor["id"] in bank_ids, f"{neighbor['id']} is not a train bank patient"
                if exclude_self:
                    assert person_id(neighbor["id"]) != person_id(record["id"]), (
                        f"{record['id']} retrieved {neighbor['id']} (same person)"
                    )
    print(f"verified: {len(records)} records, every neighbor is a bank patient"
          f"{', none from the same person' if exclude_self else ''}; "
          f"{n_empty} empty <organ>_indices lists (no usable mask)")


def compare_reference(records, reference_path, json_key):
    """Optional: report how the new file lines up with another annotation file.
    Informational only, nothing is asserted."""
    with open(reference_path, "r", encoding="utf-8") as f:
        ref = json.load(f)[json_key]
    ref_ids = [r["id"] for r in ref]
    new_ids = [r["id"] for r in records]
    print(f"reference {reference_path}: {len(ref_ids)} records vs new {len(new_ids)}; "
          f"same order: {ref_ids == new_ids}; only in new: {sorted(set(new_ids) - set(ref_ids))[:10]}; "
          f"only in reference: {sorted(set(ref_ids) - set(new_ids))[:10]}")
    same_keys = all(list(r.keys()) == list(records[0].keys()) for r in ref)
    print(f"  reference records have the same key layout: {same_keys}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--ckpt_dir", required=True,
                         help="training output dir holding perceiver-<name>.pth / fc-<name>.pth")
    parser.add_argument("--ckpt_name", required=True,
                         help="e.g. step150000 - loads perceiver-<name>.pth/fc-<name>.pth; must be "
                              "the same step as the adapter used for --unique_feats")
    parser.add_argument("--unique_reports", required=True, help="step 0's unique_reports.json (train)")
    parser.add_argument("--unique_feats", required=True, help="step 1's {organ}_feats npz (train)")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", default="two_stage", choices=["two_stage", "plain"],
                         help="two_stage: zero-shot normal/abnormal, then pick within that class "
                              "(see module docstring); plain: nearest top-k over all sentences")
    parser.add_argument("--bank_labels", default=None, help="step 0b's bank_labels.json (two_stage)")
    parser.add_argument("--prototypes", default=None,
                         help="two_stage: prototypes npz - written here if missing, loaded if present")
    parser.add_argument("--image_mean", default=None,
                         help="two_stage: train image mean per organ (for centering) - the train "
                              "run writes it, the val run loads it")
    parser.add_argument("--decisions_out", default=None,
                         help="optional: save each query's zero-shot score / normal decision (npz)")
    parser.add_argument("--query_cache", default=None,
                         help="load image queries from here if it exists, else compute and save them here")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2,
                         help="patients per DataLoader batch (each patient = up to 10 organ crops)")
    parser.add_argument("--num_workers", type=int, default=16,
                         help="CPU processes reading/cropping CTs - this, not the GPU, is the bottleneck")
    parser.add_argument("--gpu_chunk", type=int, default=16,
                         help="max organ crops per GPU forward")
    parser.add_argument("--reference_annotation", default=None,
                         help="optional: an existing organ_annotation.json to compare ids/layout against")
    parser.add_argument("--max_patients", type=int, default=None,
                         help="only the first N patients - smoke test only")
    args = parser.parse_args()

    # Must be set in the shell, not here - torch queues CUDA init at import time, so
    # setting it from inside the script is too late (same as train.py).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES in the shell before launching this script.")
    device = torch.device("cuda")
    split_cfg = SPLITS[args.split]

    dataset, query_ids, captions = load_split(args.split)
    if args.max_patients is not None:
        query_ids = query_ids[: args.max_patients]
    print(f"{args.split}: {len(query_ids)} query patients")

    with open(args.unique_reports, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    npz = np.load(args.unique_feats)
    assert list(npz["organs"]) == ORGANS, "unique_feats organ order doesn't match ORGANS"
    bank_feats = {organ: npz[f"{organ}_feats"] for organ in ORGANS}
    for organ in ORGANS:
        assert bank_feats[organ].shape[0] == len(unique_reports[organ]), (
            f"{organ}: unique_feats has {bank_feats[organ].shape[0]} rows, "
            f"unique_reports.json has {len(unique_reports[organ])} - one of them is stale, rerun steps 0/1"
        )
    bank_ids = {pid for entry in unique_reports[ORGANS[0]] for pid in entry["patient_ids"]}

    if args.query_cache and os.path.exists(args.query_cache):
        cache = torch.load(args.query_cache)
        assert cache["query_ids"] == query_ids and cache["ckpt_name"] == args.ckpt_name, (
            f"{args.query_cache} was built for a different patient list or checkpoint"
        )
        queries, valid = cache["queries"], cache["valid"]
        print(f"loaded image queries from {args.query_cache}")
    else:
        queries, valid = compute_image_queries(
            dataset, split_cfg["mask_folder"], query_ids, args.ckpt_dir, args.ckpt_name,
            args.batch_size, args.num_workers, args.gpu_chunk, device,
        )
        if args.query_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.query_cache)), exist_ok=True)
            torch.save({"query_ids": query_ids, "ckpt_name": args.ckpt_name,
                        "queries": queries, "valid": valid}, args.query_cache)
            print(f"saved image queries to {args.query_cache}")
    for organ in ORGANS:
        print(f"  {organ}: {valid[organ].sum()}/{len(query_ids)} valid image queries")

    bank_labels = prototypes = image_mean = None
    if args.mode == "two_stage":
        assert args.bank_labels and args.prototypes and args.image_mean, (
            "--mode two_stage needs --bank_labels, --prototypes and --image_mean"
        )
        with open(args.bank_labels, "r", encoding="utf-8") as f:
            bank_labels = json.load(f)
        for organ in ORGANS:
            assert len(bank_labels[organ]) == len(unique_reports[organ]), (
                f"{organ}: bank_labels.json doesn't match unique_reports.json - rerun step 0b"
            )

        if os.path.exists(args.prototypes):
            npz_p = np.load(args.prototypes)
            prototypes = {o: {"normal": npz_p[f"{o}_normal"], "abnormal": npz_p[f"{o}_abnormal"]} for o in ORGANS}
            print(f"loaded prototypes from {args.prototypes}")
        else:
            prototypes = build_prototypes(bank_feats, unique_reports, bank_labels)
            np.savez(args.prototypes, **{f"{o}_{c}": prototypes[o][c] for o in ORGANS for c in ("normal", "abnormal")})
            print(f"saved prototypes to {args.prototypes}")

        if args.split == "train":
            image_mean = {o: queries[o][valid[o]].mean(0) for o in ORGANS}
            os.makedirs(os.path.dirname(os.path.abspath(args.image_mean)), exist_ok=True)
            np.savez(args.image_mean, **image_mean)
            print(f"saved train image mean to {args.image_mean}")
        else:
            assert os.path.exists(args.image_mean), (
                f"{args.image_mean} not found - it is written by the train run (run_train.sh), "
                "which has to finish before the val run"
            )
            npz_m = np.load(args.image_mean)
            image_mean = {o: npz_m[o] for o in ORGANS}

    exclude_self = args.split == "train"
    excluded = build_excluded_rows(unique_reports, query_ids) if exclude_self else None
    top, decisions = retrieve_top_k(queries, valid, bank_feats, excluded, args.top_k, device, args.mode,
                                    bank_labels, prototypes, image_mean)
    if decisions:
        print("zero-shot decision (share of valid queries judged NORMAL):")
        for organ in ORGANS:
            if organ in decisions:
                print(f"  {organ}: {decisions[organ]['normal'][valid[organ]].mean():.1%}")
            else:
                print(f"  {organ}: no decision (plain top-k)")
        if args.decisions_out:
            np.savez(args.decisions_out, query_ids=np.array(query_ids),
                     **{f"{o}_{k}": decisions[o][k] for o in decisions for k in ("score", "normal")},
                     **{f"{o}_valid": valid[o] for o in ORGANS})
            print(f"saved decisions to {args.decisions_out}")

    records = build_records(query_ids, captions, top, unique_reports)
    verify(records, bank_ids, args.top_k, exclude_self)
    if args.reference_annotation:
        compare_reference(records, args.reference_annotation, split_cfg["json_key"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({split_cfg["json_key"]: records}, f, ensure_ascii=False)
    print(f"Saved {args.output_path}: {len(records)} records")


if __name__ == "__main__":
    main()
