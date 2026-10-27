"""KE step 0: deduplicate train report texts per organ.

Many (patient, organ) texts are word-for-word identical (above all the template
"<organ> shows no significant abnormalities."), and identical text always gives an
identical embedding - so step 1 only embeds each distinct text once, and step 2
expands the result back to one vector per (patient, organ). Pure string grouping, no
model involved.

Output unique_reports.json:

    {
        "abdomen": [
            {"text": "...", "patient_ids": ["train_10000_a_1", ...]},   # everyone who wrote this text
            ...
        ],
        ...   one list per organ; row i here == row i of {organ}_feats in step 1's npz
    }

Usage:

    python3 get_unique_reports.py --output_path /path/to/unique_reports.json
"""
import argparse
import json
import os

from config import ORGANS, load_split, template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    _, patient_ids, captions = load_split("train")
    print(f"{len(patient_ids)} train patients, {len(ORGANS)} organs")

    # dict keeps first-seen order, so rows are reproducible run to run
    grouped = {organ: {} for organ in ORGANS}
    for pid in patient_ids:
        for organ in ORGANS:
            grouped[organ].setdefault(captions[pid][organ], []).append(pid)

    unique_reports = {
        organ: [{"text": text, "patient_ids": pids} for text, pids in text_to_pids.items()]
        for organ, text_to_pids in grouped.items()
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(unique_reports, f, ensure_ascii=False)
    print(f"Saved {args.output_path}")

    total = 0
    for organ in ORGANS:
        entries = unique_reports[organ]
        total += len(entries)
        n_template = sum(len(e["patient_ids"]) for e in entries if e["text"] == template(organ))
        print(f"  {organ}: {len(entries)} unique texts ({n_template} patients use the template)")
    print(f"  total: {total} unique texts (from {len(patient_ids) * len(ORGANS)})")


if __name__ == "__main__":
    main()
