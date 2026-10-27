"""KE step 0b: label every bank sentence (step 0's unique_reports.json) as normal or
abnormal, so step 3 can do two-stage retrieval (decide normal/abnormal first, then
only pick among sentences of that class).

Labels come only from the dataset's own vqa_abnormality csv (RadGenome's automatic
"which findings, per scan and anatomy" extraction):

1. A (patient, organ) is ABNORMAL if its row lists any finding other than
   "no findings"; no row, or only "no findings", means NORMAL.
2. Negation filter: the extraction sometimes turns a negated sentence into a finding -
   "Pericardial effusion was not detected." becomes "pericardial effusion". A finding
   is dropped when, in the patient's report for that organ, every mention of it is
   directly followed by "was/were/is/are not detected/observed/seen/...". Measured on
   train: this hits 58% of heart's "abnormal" labels and <1% of every other organ's.
3. A bank sentence's label = majority vote of the patients who wrote it.

Output bank_labels.json: {organ: [true/false, ...]} - true = NORMAL, row i matches
unique_reports.json[organ][i].

Usage:

    python3 label_bank.py --unique_reports /path/to/unique_reports.json \\
        --output_path /path/to/bank_labels.json
"""
import argparse
import json
import os
import re

import pandas as pd

from config import ORGANS, SPLITS

# "X was not detected", "X-thickening were not observed", ... right after the finding
_POST_NEGATION = re.compile(
    r"^[\w\s\-/,]{0,40}?\b(was|were|is|are)\s+(not|no longer)\s+"
    r"(detected|observed|seen|present|noted|identified|found|evident|visualized)"
)


def negated_in_report(finding, report):
    """True if the finding appears in the report and every mention of it is directly
    followed by a 'was not detected'-type phrase."""
    term = finding.lower()
    hits = 0
    for sentence in re.split(r"(?<=[.;])\s+", report.lower()):
        i = sentence.find(term)
        if i < 0:
            continue
        hits += 1
        if not _POST_NEGATION.search(sentence[i + len(term):]):
            return False
    return hits > 0


def abnormal_findings(split, captions):
    """{(patient_id, organ): [findings]} for every ABNORMAL (patient, organ) of the
    split, after the negation filter. Anything not in the dict is NORMAL.
    captions = {patient_id: {organ: report text}} (config.load_split's, or rebuilt
    from unique_reports.json for train)."""
    df = pd.read_csv(SPLITS[split]["vqa_csv"])
    df = df[df["Anatomy"].isin(ORGANS)]
    abnormal = {}
    for volume, organ, text in zip(df["Volumename"], df["Anatomy"], df["Abnormality"]):
        pid = volume.removesuffix(".nii.gz")
        if pid not in captions:
            continue
        findings = [x.strip() for x in str(text).split(",")
                    if x.strip() and x.strip().lower() != "no findings"]
        report = captions[pid][organ]
        findings = [f for f in findings if not negated_in_report(f, report)]
        if findings:
            abnormal.setdefault((pid, organ), []).extend(findings)
    return abnormal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_reports", required=True, help="step 0's unique_reports.json (train)")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    with open(args.unique_reports, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    captions = {}
    for organ in ORGANS:
        for entry in unique_reports[organ]:
            for pid in entry["patient_ids"]:
                captions.setdefault(pid, {})[organ] = entry["text"]

    abnormal = abnormal_findings("train", captions)

    labels = {}
    print(f"{'organ':22s} {'normal patients':>15s} {'normal sentences':>16s} {'abnormal sentences':>18s}")
    for organ in ORGANS:
        row_normal = []
        for entry in unique_reports[organ]:
            n_normal = sum((pid, organ) not in abnormal for pid in entry["patient_ids"])
            row_normal.append(n_normal * 2 > len(entry["patient_ids"]))
        labels[organ] = row_normal
        n_pat = sum(len(e["patient_ids"]) for e in unique_reports[organ])
        n_pat_normal = sum((pid, organ) not in abnormal for e in unique_reports[organ] for pid in e["patient_ids"])
        n_rows_normal = sum(row_normal)
        print(f"{organ:22s} {n_pat_normal / n_pat:15.1%} {n_rows_normal:16d} {len(row_normal) - n_rows_normal:18d}")
        assert 0 < n_rows_normal < len(row_normal), f"{organ}: needs both normal and abnormal sentences"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
