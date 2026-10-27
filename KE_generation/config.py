"""Settings shared by every KE_generation step - edit this file to change paths.

- DATA_ROOT: where the data lives (the checkpoint dir is set in run_train.sh/run_test.sh)
- ORGANS: the 10 organs
- SPLITS: per split (train / val), the CT folder, mask folder, report csv and the
  vqa_abnormality csv (normal/abnormal labels, see label_bank.py)
- load_split(): which patients, and which report sentence per (patient, organ)

load_split() is the one place that decides "which patients, which text", so steps
0-3 can never disagree about it. Patients come from this project's own
PerOrganDataset (Reg2RG's RadGenomeDataset_Train indexing) - the same data the
adapter/perceiver/fc were trained on: train 24126, validation 1564. Text per
(patient, organ) is the raw csv Sentence, exactly as the adapter saw it in training;
an organ with no Sentence row gets the template "<organ> shows no significant
abnormalities.", so every patient has text for every organ (the bank needs a vector
at every [patient, organ]).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import PerOrganDataset  # noqa: E402

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"

# Same set as Reg2RG's REGIONS (pleura listed last here). Reg2RG looks organs up by
# name, so the order only decides the layout of the npz's organ axis.
ORGANS = [
    "abdomen", "bone", "breast", "esophagus", "heart",
    "lung", "mediastinum", "thyroid", "trachea and bronchie", "pleura",
]

SPLITS = {
    "train": {
        "json_key": "train",
        "data_folder": os.path.join(DATA_ROOT, "train_preprocessed"),
        "mask_folder": os.path.join(DATA_ROOT, "train_region_mask"),
        "csv_file": os.path.join(DATA_ROOT, "radgenome_files", "train_region_report.csv"),
        "vqa_csv": os.path.join(DATA_ROOT, "radgenome_files", "train_vqa_abnormality.csv"),
    },
    "val": {
        "json_key": "validation",
        "data_folder": os.path.join(DATA_ROOT, "valid_preprocessed"),
        "mask_folder": os.path.join(DATA_ROOT, "valid_region_mask"),
        "csv_file": os.path.join(DATA_ROOT, "radgenome_files", "validation_region_report.csv"),
        "vqa_csv": os.path.join(DATA_ROOT, "radgenome_files", "validation_vqa_abnormality.csv"),
    },
}


def template(organ):
    return f"{organ} shows no significant abnormalities."


def sample_pid(sample):
    return os.path.basename(sample["image"]).removesuffix(".nii.gz")


def person_id(pid):
    """Scan id -> person id: train_10006_b_1 -> train_10006 (b = which scan of that
    person). One person can have several scans with near-identical images/reports."""
    return pid.rsplit("_", 2)[0]


def load_split(split):
    """Returns (dataset, patient_ids, captions):
    dataset     - PerOrganDataset for the split (step 3 reuses it for image loading)
    patient_ids - sorted list of patient ids
    captions    - {patient_id: {organ: text}}, every organ present
    """
    cfg = SPLITS[split]
    dataset = PerOrganDataset(cfg["data_folder"], cfg["mask_folder"], cfg["csv_file"])
    captions = {}
    for sample in dataset._base.samples:
        pid = sample_pid(sample)
        assert pid not in captions, f"duplicate patient id {pid}"
        texts = {}
        for organ in ORGANS:
            sentence = sample[organ][1] if organ in sample else None
            if isinstance(sentence, str) and sentence.strip():
                texts[organ] = sentence.strip()
            else:
                texts[organ] = template(organ)
        captions[pid] = texts
    return dataset, sorted(captions), captions
