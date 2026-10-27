"""KE step 2: expand step 1's deduplicated text bank ({organ}_feats, one row per unique
text) into the dense per-patient format Reg2RG's archive loads
(my_embedding_layer.py only accepts `feats [N_patients, N_organs, dim]` +
`patient_ids` + `organs`):

    feats[p, o] = {organ}_feats[row of patient p's text under organ o]

Identical text -> identical embedding, so this gives exactly what embedding all
24126 x 10 texts one by one would, without step 1 recomputing duplicates.

`patient_ids` is the sorted train patient list (every patient appears in every
organ's unique_reports.json list - organs without a finding carry the template text,
see config.py). Optional --reference_npz checks that `patient_ids` / `organs` /
(N, organs) match another bank exactly (only the last dim may differ) - useful when
replacing a bank Reg2RG already uses.

Usage:

    python3 build_dense_bank.py \\
        --unique_reports /path/to/unique_reports.json \\
        --unique_feats /path/to/organ_report_embeddings_unique_4096.npz \\
        --output_path /path/to/organ_report_embeddings.npz
"""
import argparse
import json
import os

import numpy as np

from config import ORGANS


def build_row_index(unique_reports):
    """organ -> {patient_id: row in that organ's unique bank}. Each patient must
    appear in exactly one row per organ (one text per (patient, organ))."""
    row_index = {}
    for organ in ORGANS:
        pid_to_row = {}
        for row, entry in enumerate(unique_reports[organ]):
            for pid in entry["patient_ids"]:
                assert pid not in pid_to_row, f"{organ}: {pid} appears in more than one unique row"
                pid_to_row[pid] = row
        row_index[organ] = pid_to_row
    return row_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_reports", required=True, help="step 0's unique_reports.json")
    parser.add_argument("--unique_feats", required=True, help="step 1's {organ}_feats npz")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                         help="float32 is ~4GB at 4096-d; float16 halves that, but Reg2RG "
                              "holds the bank as a register_buffer on every GPU")
    parser.add_argument("--reference_npz", default=None,
                         help="optional: an existing bank whose patient_ids/organs/(N, organs) "
                              "the new one must match exactly")
    args = parser.parse_args()

    with open(args.unique_reports, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    row_index = build_row_index(unique_reports)

    patient_ids = sorted(row_index[ORGANS[0]])
    for organ in ORGANS:
        assert set(row_index[organ]) == set(patient_ids), (
            f"{organ}: patient set differs from {ORGANS[0]}'s - every patient should have "
            "text for every organ (template if no finding)"
        )
    print(f"{len(patient_ids)} patients, organs={ORGANS}")

    npz = np.load(args.unique_feats)
    assert list(npz["organs"]) == ORGANS, "unique_feats organ order doesn't match ORGANS"
    dim = npz[f"{ORGANS[0]}_feats"].shape[1]
    feats = np.empty((len(patient_ids), len(ORGANS), dim), dtype=np.dtype(args.dtype))
    for o, organ in enumerate(ORGANS):
        organ_feats = npz[f"{organ}_feats"]
        assert organ_feats.shape[0] == len(unique_reports[organ]), (
            f"{organ}: unique_feats has {organ_feats.shape[0]} rows, unique_reports.json has "
            f"{len(unique_reports[organ])} - one of them is stale, rerun steps 0/1"
        )
        rows = np.array([row_index[organ][pid] for pid in patient_ids], dtype=np.int64)
        feats[:, o, :] = organ_feats[rows]
    print(f"feats: {feats.shape} {feats.dtype}")

    norms = np.linalg.norm(feats.astype(np.float32), axis=-1)
    assert not np.isnan(feats).any(), "feats contain NaN"
    assert np.allclose(norms, 1.0, atol=1e-2), f"feats not L2-normalized: norms in [{norms.min()}, {norms.max()}]"

    patient_ids_arr = np.array(patient_ids)
    organs_arr = np.array(ORGANS)
    if args.reference_npz:
        ref = np.load(args.reference_npz)
        assert set(ref.files) == {"feats", "patient_ids", "organs"}, f"reference keys {ref.files}"
        assert np.array_equal(ref["patient_ids"], patient_ids_arr), "patient_ids differ from reference"
        assert np.array_equal(ref["organs"], organs_arr), "organs differ from reference"
        assert ref["feats"].shape[:2] == feats.shape[:2], "feats (N, organs) differ from reference"
        print(f"matches reference {args.reference_npz}: same keys / patient_ids / organs / "
              f"(N, organs); dim {ref['feats'].shape[-1]} -> {dim}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(args.output_path, feats=feats, patient_ids=patient_ids_arr, organs=organs_arr)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
