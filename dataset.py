"""Per-organ (image, report) dataset for the Perceiver/BERT contrastive + generation
pretraining. One item = one (patient, organ) pair, not one (patient) with all organs
bundled together like Reg2RG's own RadGenomeDataset_Train - that class is built for
Reg2RG's multi-region generation prompt format (full/mask branches, its own LLM
tokenizer, combined-report text), none of which this pretraining needs.

Reuses RadGenomeDataset_Train's sample indexing (load_accession_sentences/
prepare_samples) and its mask_nii_img_to_tensor cropping/HU-window/3-channel-repeat
pipeline by composition (wrapping an instance of it), not reimplementation - the
Perceiver must see the exact same input distribution during this pretraining that it
will see once loaded back into Reg2RG's self.perceiver, so the crop must come from the
same code, not a reimplementation that could silently drift.

Imports RadGenomeDataset_Train from the Reg2RG *archive* (no-fvlm) codebase
specifically - see /home/wisc/dxiang23/.claude/projects/.../memory/
project_reg2rg_base_pivot.md - since that archive is where the trained Perceiver is
going back into, not the fvlm-modified working copy.
"""
import os
import sys

import torch
from torch.utils.data import Dataset

_ARCHIVE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Reg2RG-debug-current-full_correct_F_20260920_archive",
    "src",
)
sys.path.insert(0, _ARCHIVE_SRC)
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train, REGIONS  # noqa: E402


class PerOrganDataset(Dataset):
    """Flattens RadGenomeDataset_Train's per-patient samples into one item per
    (patient, organ) pair. Each item: a 3-channel cropped organ sub-volume (shape
    (3, 256, 256, 64), same preprocessing Reg2RG's own region branch uses) and that
    organ's report sentence.
    """

    def __init__(self, data_folder, mask_folder, csv_file):
        # text_tokenizer/image_padding_tokens/region_padding_tokens are only used by
        # RadGenomeDataset_Train's __getitem__/text_add_*_tokens methods, which this
        # class never calls - safe to pass None. __init__ itself never touches them.
        self._base = RadGenomeDataset_Train(
            text_tokenizer=None,
            image_padding_tokens=None,
            region_padding_tokens=None,
            data_folder=data_folder,
            mask_folder=mask_folder,
            csv_file=csv_file,
        )

        self.flat_index = []  # list of (sample_idx, organ_name)
        for sample_idx, sample in enumerate(self._base.samples):
            for organ in REGIONS:
                if organ in sample:
                    self.flat_index.append((sample_idx, organ))

        print(f"PerOrganDataset: {len(self._base.samples)} patients -> "
              f"{len(self.flat_index)} (patient, organ) items")

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, index):
        sample_idx, organ = self.flat_index[index]
        sample = self._base.samples[sample_idx]
        img_file = sample["image"]
        mask_file, report_text = sample[organ]
        patient_id = os.path.basename(img_file).removesuffix(".nii.gz")

        # Pass only this one organ's mask - mask_nii_img_to_tensor loops over
        # whatever's in mask_paths, so this crops just the organ we need (it still
        # always also builds the global-image tensor internally, which we discard -
        # that's wasted but harmless; keeping the call identical to Reg2RG's own usage
        # matters more than trimming that one extra step).
        #
        # mask_nii_img_to_tensor has a real edge-case bug when called with a single
        # organ whose mask is entirely empty: it defines hu_min/hu_max only *inside*
        # the per-organ loop, after the "skip if mask is all-zero" check, then uses
        # them again *outside* the loop unconditionally. Reg2RG's own usage always
        # passes all ~10 organs at once, so some other organ's mask being non-empty
        # masks this bug in practice; our single-organ-per-call usage hits it directly
        # whenever that one organ happens to be empty. Not our place to patch the
        # archive's own method (it's the target codebase, not ours to edit) - catch it
        # here instead and skip the sample, same as any other bad-sample case.
        try:
            mask_img_tensors, _ = self._base.mask_nii_img_to_tensor(
                img_file, {organ: mask_file}
            )
        except UnboundLocalError:
            return None
        if organ not in mask_img_tensors:
            # mask_nii_img_to_tensor skips organs whose mask is all-zero (see its
            # `if np.sum(mask_data) == 0: continue`) - shouldn't happen since
            # prepare_samples already only lists organs with a caption, but guard
            # against it rather than crash the whole batch on one bad sample.
            return None

        volume = mask_img_tensors[organ][0]  # drop the batch dim: (3, 256, 256, 64)

        return {
            "patient_id": patient_id,
            "organ": organ,
            "image": volume,
            "text": report_text,
        }


def collate_skip_none(batch):
    """Drops any None items (see PerOrganDataset.__getitem__'s all-zero-mask guard)
    before the default collation, so one bad sample doesn't crash the whole batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "organ": [item["organ"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "text": [item["text"] for item in batch],
    }
